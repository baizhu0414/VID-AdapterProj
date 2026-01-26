# 依赖: pip install torch torchvision diffusers transformers tqdm numpy pillow
import os
import math
import random
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.ops import roi_align
from torchvision.utils import save_image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

# --------------------------- 超参（核心调整） ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2  # 降低batch size减少显存占用
VAE_SCALE_FACTOR = 8
LATENT_CHANNELS = 4
D_txt = 768  # CLIP text encoder维度
D_obj = 512
H = 8
LAMBDA_OBJ = 0.5  # Attn_obj权重
LAMBDA_GLOBAL = 0.3  # 新增：Attn_global权重
LAMBDA_REGION = 0.5
LAMBDA_TRIPLET = 0.3  # 三元组损失权重，需匹配负样本逻辑
LAMBDA_ATTN = 0.2
GATE_INIT = 0.2
EPS_MASK = -1e9
NUM_BANDS = 8
P_FOURIER = 4 * 2 * NUM_BANDS
SAVE_DIR = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/3drealcar_dataset/save_vehicle_adapter"
LOG_CSV = os.path.join(SAVE_DIR, "train_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 256
N_OBJ = 1  # 简化为单目标，减少维度复杂度
D_id = 768
VEHICLE_TYPES = ["Sedan","SUV","MPV","Pickup","Van","Big Truck","Sports Car","Bus"]
type2idx = {v:i for i,v in enumerate(VEHICLE_TYPES)}

# --------------------------- 工具函数 ---------------------------
def fourier_feats_sin_cos(box_coords: torch.Tensor, num_bands: int = NUM_BANDS) -> torch.Tensor:
    # 修复：兼容2维输入（自动扩展为3维）
    if len(box_coords.shape) == 2:
        box_coords = box_coords.unsqueeze(1)  # [B,4] → [B,1,4]（N=1）
    B, N, _ = box_coords.shape
    device = box_coords.device
    scales = (2 ** torch.arange(num_bands, device=device).float()) * math.pi
    coords = box_coords.unsqueeze(-1) * scales.view(1,1,1,-1)
    sin = torch.sin(coords)
    cos = torch.cos(coords)
    fourier = torch.cat([sin, cos], dim=-1)
    fourier = fourier.view(B, N, -1)
    return fourier

def sanitize_key(name: str) -> str:
    return name.replace(".", "__").replace("/", "_").replace(":", "_")

# --------------------------- 核心模块 ---------------------------
class MLPObj(nn.Module):
    def __init__(self, D_id:int, P_box:int, C_type:int, D_obj:int):
        super().__init__()
        self.proj_id = nn.Linear(D_id, D_obj)
        self.proj_box = nn.Linear(P_box, D_obj)
        self.proj_type = nn.Linear(C_type, D_obj)
        self.mlp = nn.Sequential(
            nn.Linear(3 * D_obj, D_obj),
            nn.ReLU(),
            nn.Linear(D_obj, D_obj)
        )
    def forward(self, id_emb, box_fourier, type_onehot, box_mask):
        # 强制所有输入转为3维
        if len(id_emb.shape) != 3:
            id_emb = id_emb.view(-1, 1, id_emb.shape[-1])
        if len(box_fourier.shape) != 3:
            box_fourier = box_fourier.view(-1, 1, box_fourier.shape[-1])
        if len(type_onehot.shape) != 3:
            type_onehot = type_onehot.view(-1, 1, type_onehot.shape[-1])
        if len(box_mask.shape) == 2:
            box_mask = box_mask.unsqueeze(-1)  # [B,N] → [B,N,1]
        
        z_id = self.proj_id(id_emb)
        z_box = self.proj_box(box_fourier)
        z_type = self.proj_type(type_onehot)
        
        cat = torch.cat([z_id, z_box, z_type], dim=-1)
        out = self.mlp(cat)
        out = out * box_mask  # box_mask已经是[B,N,1]，可广播
        
        # 强制输出为3维（核心修复！）
        out = out.view(out.shape[0], -1, out.shape[-1])
        return out


class ObjectTransformer(nn.Module):
    def __init__(self, D_obj:int, n_layers:int=1, nhead:int=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=D_obj, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(D_obj)

    def forward(self, obj_feats, obj_mask):
        # 终极兜底：强制转为3维（不管输入是几维）
        if len(obj_feats.shape) < 3:
            obj_feats = obj_feats.unsqueeze(1)  # 2维→3维
        elif len(obj_feats.shape) > 3:
            # 4维及以上：合并前N-1维为B，保留最后两维为N,D
            obj_feats = obj_feats.view(-1, obj_feats.shape[-2], obj_feats.shape[-1])
        
        # 现在obj_feats一定是3维
        B,N,D = obj_feats.shape
        
        # 核心修复：先将obj_mask压缩到2维 [B,N]（去掉多余的维度）
        obj_mask = obj_mask.squeeze(-1)  # 3维[1,1,1] → 2维[1,1]
        if len(obj_mask.shape) == 1:
            obj_mask = obj_mask.unsqueeze(1)  # [B] → [B,1]
        obj_mask = obj_mask.expand(B, N)
        
        # 修复3：mask_expanded维度修正（避免维度不匹配）
        mask_expanded = (1.0 - obj_mask).bool()  # [B,N]
        try:
            out = self.transformer(obj_feats, src_key_padding_mask=mask_expanded)
        except Exception as e:
            print(f"Transformer执行失败：{e}，使用原特征")
            out = obj_feats
        return self.ln(out + obj_feats)


class MultiLayerKVProjector(nn.Module):
    def __init__(self, layers:List[str], D_obj:int, D_txt:int, H:int, per_layer_dims:dict=None):
        super().__init__()
        self.layers = layers
        self.D_txt = D_txt
        self.H = H
        self.per_layer_dims = per_layer_dims or {}
        self.default_d_h = D_txt // H
        
        # 提前注册所有投影层
        self.proj_k = nn.ModuleDict()
        self.proj_v = nn.ModuleDict()
        for l in layers:
            info = self.per_layer_dims.get(l, None)
            if info is not None:
                tgt_heads = int(info.get('heads', H))
                tgt_dh = int(info.get('d_h', self.default_d_h))
                out_dim = tgt_heads * tgt_dh
            else:
                out_dim = H * self.default_d_h
            self.proj_k[l] = nn.Linear(D_obj, out_dim)
            self.proj_v[l] = nn.Linear(D_obj, out_dim)

    def forward(self, obj_feats):
        B,N,_ = obj_feats.shape
        Ks, Vs = {}, {}
        for l in self.layers:
            K = self.proj_k[l](obj_feats)
            V = self.proj_v[l](obj_feats)
            info = self.per_layer_dims.get(l, None)
            if info is not None:
                tgt_heads = int(info.get('heads'))
                tgt_dh = int(info.get('d_h'))
            else:
                tgt_heads = self.H
                tgt_dh = self.default_d_h
            K = K.view(B, N, tgt_heads, tgt_dh).permute(0,2,1,3).contiguous()
            V = V.view(B, N, tgt_heads, tgt_dh).permute(0,2,1,3).contiguous()
            Ks[l] = K
            Vs[l] = V
        return Ks, Vs


class GateNet(nn.Module):
    def __init__(self, D_obj:int, R_dim:int, H:int, init_val:float=GATE_INIT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_obj + R_dim, D_obj), 
            nn.ReLU(), 
            nn.Linear(D_obj, H)
        )
        b = math.log(init_val / (1.0 - init_val))
        self.net[-1].bias.data.fill_(b)

    def forward(self, obj_feats, rel_feats):
        x = torch.cat([obj_feats, rel_feats], dim=-1)
        g_pre = self.net(x)
        g = torch.sigmoid(g_pre).unsqueeze(-1)
        return g

# --------------------------- 核心修复：compute_obj_attention ---------------------------
def compute_obj_attention(Q:torch.Tensor, K_obj:torch.Tensor, V_obj:torch.Tensor, 
                         obj_mask:torch.Tensor, gate:torch.Tensor, eps_mask:float=EPS_MASK):
    B, T_q, D_q = Q.shape
    Bk, H, N, d_h = K_obj.shape
    
    # 安全检查 + 强制适配
    if B != Bk:
        K_obj = K_obj[:B]
        V_obj = V_obj[:B]
    if D_q != H * d_h:
        # 强制映射Q到目标维度（指定device和dtype，避免类型不匹配）
        proj = nn.Linear(D_q, H * d_h, device=Q.device, dtype=Q.dtype)
        Q = proj(Q)
        D_q = H * d_h
    
    # Q reshape: [B, H, T_q, d_h]
    Q_h = Q.view(B, T_q, H, d_h).permute(0,2,1,3).contiguous()
    
    # 计算注意力分数
    logits = torch.matmul(Q_h, K_obj.transpose(-1,-2)) / math.sqrt(d_h)
    
    # mask处理（规范obj_mask维度为[B,N]）
    obj_mask = obj_mask.view(B, -1)  # 强制obj_mask为2维[B,N]
    if obj_mask.shape[1] != N:
        obj_mask = obj_mask[:, :N]  # 截断到目标N，避免维度不匹配
    mask_logits = (1.0 - obj_mask).unsqueeze(1).unsqueeze(2) * eps_mask  # [B,1,1,N]
    logits = logits + mask_logits.to(logits.device)
    
    # softmax
    alpha = torch.softmax(logits, dim=-1)  # alpha维度：[B, H, T_q, N]
    
    # gate应用（核心修复：修正gate维度顺序和扩展逻辑）
    gate = gate.view(B, -1, H).unsqueeze(2)  # 重塑为[B,1,1,H]
    gate = gate.permute(0, 3, 1, 2).contiguous()  # 关键修复：调整维度顺序
    gate = gate.expand(B, H, T_q, N)
    alpha = alpha * gate
    
    # 归一化
    alpha_sum = alpha.sum(dim=-1, keepdim=True) + 1e-8
    alpha = alpha / alpha_sum
    
    # 计算输出
    attn_out = torch.matmul(alpha, V_obj)

    # 兼容性处理（简化版，优先挤压冗余维度）
    try:
        attn_out = attn_out.squeeze()
        if attn_out.dim() == 3:
            attn_out = attn_out.unsqueeze(1)
        elif attn_out.dim() < 3:
            raise RuntimeError(f"attn_out维度过低: {attn_out.dim()}维，shape={attn_out.shape}")
        if attn_out.dim() == 4:
            attn_out = attn_out.permute(0,2,1,3).reshape(B, T_q, D_q)
        else:
            attn_out = attn_out.view(B, T_q, D_q)
    except Exception as e:
        print("compute_obj_attention shape error:")
        print("Q.shape", Q.shape)
        print("K_obj.shape", K_obj.shape)
        print("V_obj.shape", V_obj.shape)
        print("alpha.shape", alpha.shape)
        print("attn_out.shape(before handling)", attn_out.shape if 'attn_out' in locals() else None)
        raise
    
    return attn_out, alpha


# --------------------------- Adapter总封装 ---------------------------
class VehicleAdapter(nn.Module):
    def __init__(self, D_id:int, P_box:int, C_type:int, D_obj:int, layers:List[str], D_txt:int, H:int, per_layer_dims:dict=None):
        super().__init__()
        self.layers = layers
        self.mlpobj = MLPObj(D_id, P_box, C_type, D_obj)
        self.obj_transformer = ObjectTransformer(D_obj, n_layers=1, nhead=4)
        self.kv_proj = MultiLayerKVProjector(layers, D_obj, D_txt, H, per_layer_dims=per_layer_dims)
        self.gate = GateNet(D_obj, R_dim=16, H=H)
        self.rel_proj = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16))
        
        # ModuleDict初始化正确，仅调用时需注意
        self.per_layer_global_proj = nn.ModuleDict()
        self.per_layer_kproj = nn.ModuleDict()
        self.per_layer_vproj = nn.ModuleDict()
        self.per_layer_gateproj = nn.ModuleDict()
        
        for l in layers:
            layer_dim = per_layer_dims.get(l, {})
            hidden_size = layer_dim.get('hidden_size', D_txt)
            self.per_layer_kproj[l] = nn.Linear(D_obj, hidden_size)
            self.per_layer_vproj[l] = nn.Linear(D_obj, hidden_size)
            self.per_layer_global_proj[l] = nn.Linear(hidden_size, hidden_size)

        # 新增：提前初始化 gate 投影层，使用键名 g_<sanitized_layer>
        for l in layers:
            info = per_layer_dims.get(l, {})
            tgt_heads = int(info.get('heads', H))
            gname = f"g_{sanitize_key(l)}"
            self.per_layer_gateproj[gname] = nn.Linear(int(H), tgt_heads)

        # 缓存
        self.last_Ks = None
        self.last_Vs = None
        self.last_g = None
        self.last_mask = None
        self.last_alpha = {}
        self.layer_name_map = {}

    def forward(self, id_embs, box_coords, types_onehot, box_mask):
        # 修复1：强制所有输入转为3维（B,N,D）
        if len(id_embs.shape) == 2:
            id_embs = id_embs.unsqueeze(1)
        if len(types_onehot.shape) == 2:
            types_onehot = types_onehot.unsqueeze(1)
        if len(box_mask.shape) == 1:
            box_mask = box_mask.unsqueeze(1)
        
        B = id_embs.shape[0]
        box_fourier = fourier_feats_sin_cos(box_coords)
        
        # MLP处理
        obj_feats = self.mlpobj(id_embs, box_fourier, types_onehot, box_mask)
        
        # 核心修复：强制obj_feats为3维
        if len(obj_feats.shape) != 3:
            obj_feats = obj_feats.view(obj_feats.shape[0], -1, obj_feats.shape[-1])
        
        # Transformer处理
        obj_feats_inter = self.obj_transformer(obj_feats, box_mask)
        
        # 关系特征
        B,N,_ = obj_feats_inter.shape
        rel_feats = self.rel_proj(torch.zeros(B,N,8, device=obj_feats_inter.device))
        
        # Gate生成
        g = self.gate(obj_feats_inter, rel_feats)
        
        # KV投影
        Ks, Vs = self.kv_proj(obj_feats_inter)
        
        # 缓存
        self.last_Ks = Ks
        self.last_Vs = Vs
        self.last_g = g
        self.last_mask = box_mask
        self.last_alpha = {l: None for l in Ks.keys()}
        
        return Ks, Vs, g, box_mask
    

# --------------------------- ROI裁剪 ---------------------------
def roi_crop_and_resize(images: torch.Tensor, boxes: torch.Tensor, output_size=(256,256)):
    # 修复点：增加维度检查和修正，确保images为4维
    if len(images.shape) == 5:
        images = images[:, :, 0, :, :]
    elif len(images.shape) != 4:
        raise ValueError(f"images维度错误，期望4维，实际{len(images.shape)}维：{images.shape}")
    
    B, C, H, W = images.shape
    B2, N, _ = boxes.shape
    assert B == B2, f"批次大小不匹配：images={B}, boxes={B2}"
    device = images.device
    
    # 标准化box坐标到[0,1] -> [0, W/H]（适配roi_align）
    boxes_abs = boxes.clone()
    boxes_abs[..., 0] *= W
    boxes_abs[..., 1] *= H
    boxes_abs[..., 2] *= W
    boxes_abs[..., 3] *= H
    
    # 构造roi_align输入：(B*N, 4)
    rois = []
    for b in range(B):
        for n in range(N):
            rois.append([b, boxes_abs[b,n,0], boxes_abs[b,n,1], boxes_abs[b,n,2], boxes_abs[b,n,3]])
    rois = torch.tensor(rois, dtype=torch.float32, device=device)
    
    # ROI裁剪
    crops = roi_align(images, rois, output_size=output_size, spatial_scale=1.0)
    crops = crops.view(B, N, C, output_size[0], output_size[1])
    return crops


# --------------------------- 数据集（核心修改：负样本逻辑） ---------------------------
class VehicleDataset(Dataset):
    def __init__(self, root_dir: str, transform, type2idx: dict, size=(IMG_SIZE, IMG_SIZE)):
        super().__init__()
        self.root = Path(root_dir)
        self.anns_dir = self.root / "anns"
        self.masks_dir = self.root / "masks"
        self.transform = transform
        self.type2idx = type2idx
        self.idx2type = {v:k for k,v in type2idx.items()}
        self.size = size
        self.C_type = len(type2idx)
        
        # 样本列表和映射
        self.samples = []
        self.car_folder_map = {}
        self.sample_id_map = {}
        
        # 核心映射：文件夹→类型字符串、类型字符串→文件夹列表（用于负样本筛选）
        self.car_folder_to_type_str = {}
        self.type_str_to_car_folders = {}
        
        # 加载标注
        for json_path in self.anns_dir.glob("*.json"):
            car_folder = json_path.stem
            with open(json_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
            
            folder_type_str = None
            for img_result in ann_data["image_results"]:
                img_path = img_result["original_image_path"]
                for box_idx, box_result in enumerate(img_result["box_results"]):
                    # 读取类型字符串
                    type_str = box_result["anno"]["type"]
                    type_idx = self.type2idx.get(type_str, 0)
                    type_onehot = [0]*self.C_type
                    type_onehot[type_idx] = 1
                    
                    # 构建样本
                    sample = {
                        "car_folder": car_folder,
                        "image_path": img_path,
                        "box": box_result["box"],
                        "type_str": type_str,
                        "type_onehot": type_onehot,
                        "id_embs": box_result["embeddings"],
                        "mask_path": box_result["mask_path"],
                        "caption": box_result["anno"].get("caption", ""),
                        "sample_id": f"{img_path}_{box_idx}"
                    }
                    self.samples.append(sample)
                    
                    # 更新文件夹-样本映射
                    if car_folder not in self.car_folder_map:
                        self.car_folder_map[car_folder] = []
                    self.car_folder_map[car_folder].append(sample)
                    self.sample_id_map[sample["sample_id"]] = sample
                    
                    # 记录文件夹类型
                    if folder_type_str is None:
                        folder_type_str = type_str
            
            # 缓存文件夹类型（兜底）
            if folder_type_str is None:
                folder_type_str = random.choice(list(self.type2idx.keys()))
            self.car_folder_to_type_str[car_folder] = folder_type_str
            
            # 更新类型-文件夹映射
            if folder_type_str not in self.type_str_to_car_folders:
                self.type_str_to_car_folders[folder_type_str] = []
            if car_folder not in self.type_str_to_car_folders[folder_type_str]:
                self.type_str_to_car_folders[folder_type_str].append(car_folder)
        
        # ========== 新增：过滤无效文件夹 ==========
        # 过滤：只保留存在于 car_folder_map 中的文件夹（即有样本的文件夹）
        for type_str in list(self.type_str_to_car_folders.keys()):
            valid_folders = [f for f in self.type_str_to_car_folders[type_str] if f in self.car_folder_map]
            self.type_str_to_car_folders[type_str] = valid_folders
            # 如果过滤后为空，删除该类型（避免后续选到空列表）
            if not self.type_str_to_car_folders[type_str]:
                del self.type_str_to_car_folders[type_str]
        
        # 整理文件夹列表
        self.car_folders = list(self.car_folder_map.keys())
        
        # ========== 新增：校验日志 ==========
        print(f"加载完成：共{len(self.samples)}个样本，{len(self.car_folders)}个车辆文件夹，{len(self.type_str_to_car_folders)}种车辆类型")
        for i, folder in enumerate(list(self.car_folder_map.keys())[:5]):
            print(f"  文件夹 {folder}: {len(self.car_folder_map[folder])} 个样本")
        
        # 校验：打印各类型的有效文件夹数
        print("\n【类型-有效文件夹数】校验：")
        for type_str, folders in self.type_str_to_car_folders.items():
            print(f"  {type_str}: {len(folders)} 个有效文件夹")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 1. 获取锚点样本
        anchor_sample = self.samples[idx]
        anchor_car_folder = anchor_sample["car_folder"]
        anchor_sample_id = anchor_sample["sample_id"]
        anchor_type_str = self.car_folder_to_type_str.get(anchor_car_folder, anchor_sample["type_str"])
        
        # 2. 读取锚点图像和掩码
        anchor_image = Image.open(anchor_sample["image_path"]).convert("RGB")
        anchor_image = self.transform(anchor_image)
        
        mask_path = anchor_sample["mask_path"]
        if not Path(mask_path).is_absolute():
            mask_path = self.root / mask_path
        mask_image = Image.open(mask_path).convert("L")
        mask_image = T.Resize(self.size)(mask_image)
        mask_tensor = T.ToTensor()(mask_image).squeeze(0)
        
        # 3. 找正样本（同文件夹，排除自身）
        folder_samples = self.car_folder_map.get(anchor_car_folder, [])
        if not folder_samples:
            print(f"[ERROR] 文件夹 {anchor_car_folder} 无样本，使用当前样本作为正样本")
            pos_sample = anchor_sample
        else:
            pos_candidates = [s for s in folder_samples if s["sample_id"] != anchor_sample_id]
            pos_sample = random.choice(pos_candidates) if pos_candidates else random.choice(folder_samples)
        
        # 打印正样本调试信息
        print(f"[DEBUG] 索引 {idx}：文件夹 {anchor_car_folder} 样本数={len(folder_samples)}，候选正样本数={len(pos_candidates) if 'pos_candidates' in locals() else 0}")
        
        pos_image = Image.open(pos_sample["image_path"]).convert("RGB")
        pos_image = self.transform(pos_image)
        
        # 4. 找负样本（核心优化：优先同类型、非当前文件夹，增加存在性校验）
        # 4.1 筛选同类型的其他有效文件夹（必须在 car_folder_map 中）
        same_type_folders = [f for f in self.type_str_to_car_folders.get(anchor_type_str, []) 
                             if f != anchor_car_folder and f in self.car_folder_map]

        if same_type_folders:
            neg_car_folder = random.choice(same_type_folders)
            neg_samples = self.car_folder_map[neg_car_folder]
            neg_sample = random.choice(neg_samples)
            print(f"[DEBUG] 索引 {idx}：锚点类型={anchor_type_str}，选取同类型负样本文件夹={neg_car_folder}")
        else:
            # 兜底1：无同类型有效文件夹，选其他任意有效文件夹
            neg_car_folders = [f for f in self.car_folders if f != anchor_car_folder and f in self.car_folder_map]
            if neg_car_folders:
                neg_car_folder = random.choice(neg_car_folders)
                neg_samples = self.car_folder_map[neg_car_folder]
                neg_sample = random.choice(neg_samples)
                print(f"[WARNING] 索引 {idx}：锚点类型={anchor_type_str} 无同类型负样本，选取其他文件夹={neg_car_folder}")
            else:
                # 兜底2：仅当所有样本都在同一文件夹时，选其他样本
                other_candidates = [s for s in self.samples if s["sample_id"] != anchor_sample_id]
                neg_sample = random.choice(other_candidates) if other_candidates else anchor_sample
                print(f"[WARNING] 索引 {idx}：无可用负样本文件夹，选取其他样本")
        
        neg_image = Image.open(neg_sample["image_path"]).convert("RGB")
        neg_image = self.transform(neg_image)
        
        # 5. 整理张量（保证维度匹配）
        anchor_box = torch.tensor(anchor_sample["box"], dtype=torch.float32).unsqueeze(0)
        anchor_id_embs = torch.tensor(anchor_sample["id_embs"], dtype=torch.float32).unsqueeze(0)
        anchor_type_onehot = torch.tensor(anchor_sample["type_onehot"], dtype=torch.float32).unsqueeze(0)
        anchor_box_mask = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        pos_id_embs = torch.tensor(pos_sample["id_embs"], dtype=torch.float32).unsqueeze(0)
        neg_id_embs = torch.tensor(neg_sample["id_embs"], dtype=torch.float32).unsqueeze(0)
        
        return {
            "image": anchor_image,
            "pos_image": pos_image,
            "neg_image": neg_image,
            "id_embs": anchor_id_embs,
            "pos_embs": pos_id_embs,
            "neg_embs": neg_id_embs,
            "boxes_coords": anchor_box,
            "box_mask": anchor_box_mask,
            "types": anchor_type_onehot,
            "masks": mask_tensor,
            "caption": anchor_sample["caption"],
            "type_str": anchor_type_str
        }

# --------------------------- 模拟车辆编码器 ---------------------------
def load_transreid_config(cfg_path):
    """加载TransReID的yaml配置文件（适配官方格式）"""
    import yaml
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 封装为可属性访问的字典（适配TransReID的cfg结构）
    class ConfigDict(dict):
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
    
    cfg = ConfigDict(cfg)
    def recursive_convert(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = ConfigDict(v)
                recursive_convert(d[k])
    recursive_convert(cfg)
    return cfg


def load_vehicle_encoder(cfg_path:str, pretrained_path:str, device="cuda"):
    # 新增：路径校验（提前报错）
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"TransReID配置文件不存在：{cfg_path}")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"TransReID预训练权重不存在：{pretrained_path}")
    
    try:
        # 修复1：适配官方TransReID的导入路径
        from transreid.models.build import build_model
        from transreid.config import get_cfg
        
        # 修复2：加载并解析配置文件（官方标准流程）
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.freeze()
        
        # 修复3：构建模型（特征提取模式，关闭分类头）
        model = build_model(cfg, num_classes=0)  # num_classes=0 关闭分类头，仅特征提取
        model = model.to(device)
        
        # 修复4：优化权重加载（处理module.前缀）
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # 去除分布式训练的module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # 宽松加载（忽略分类头等不匹配层）
        model.load_state_dict(new_state_dict, strict=False)
        
        # 修复5：正确获取特征维度
        feat_dim = getattr(model, 'feat_dim', None) or getattr(model, 'num_features', None) or 768
        model.eval()  # 特征提取需设为eval模式
        return model, int(feat_dim)
    
    except ImportError as e:
        # 降级方案1：尝试自定义路径的TransReID
        try:
            from model.backbones.vit_pytorch import TransReID  # 自定义路径
            cfg = load_transreid_config(cfg_path)  # 使用新增的配置加载函数
            model = TransReID(
                img_size=cfg.INPUT.SIZE_TEST,
                patch_size=16,
                stride_size=cfg.MODEL.STRIDE_SIZE,
                in_chans=3,
                num_classes=0,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                camera=7 if cfg.MODEL.SIE_CAMERA else 0,
                view=3 if cfg.MODEL.SIE_VIEW else 2,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                local_feature=False,
                sie_xishu = cfg.MODEL.SIE_COE if (hasattr(cfg.MODEL, 'SIE_COE') and cfg.MODEL.SIE_COE is not None) else 1.0
            ).to(device)
            # 加载权重（处理module.前缀）
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            return model, 768
        except Exception as e2:
            print(f"Warning: 自定义TransReID也加载失败({e2})，使用MockVehicleEncoder")
            # 最终降级：模拟编码器（保证接口一致）
            class _MockVehicleEncoder(nn.Module):
                def __init__(self, D_id=768):
                    super().__init__()
                    self.feature_extractor = nn.Sequential(
                        nn.Conv2d(3, 64, 3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(64, D_id)
                    )
                def forward(self, x, cam_label=None, view_label=None):
                    # 与真实TransReID接口完全对齐
                    return self.feature_extractor(x)
            vehicle_model = _MockVehicleEncoder(D_id=D_id).to(device)
            return vehicle_model, D_id
    except Exception as e:
        print(f"Warning: failed to load transreid model ({e}), falling back to MockVehicleEncoder.")
        class _MockVehicleEncoder(nn.Module):
            def __init__(self, D_id=768):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, D_id)
                )
            def forward(self, x, cam_label=None, view_label=None):
                return self.feature_extractor(x)
        vehicle_model = _MockVehicleEncoder(D_id=D_id).to(device)
        return vehicle_model, D_id

# --------------------------- 推断UNet层维度 ---------------------------
def infer_unet_layer_dims(unet, default_heads=8):
    layer_map = {}
    for name, module in unet.named_modules():
        if "attn2" in name and "processor" in name:
            if hasattr(module, "heads"):
                heads = module.heads
                hidden_size = module.to_q.weight.shape[0] if hasattr(module, "to_q") else 320
                layer_map[name] = {
                    "hidden_size": hidden_size,
                    "heads": heads,
                    "d_h": hidden_size // heads,
                    "is_cross": True
                }
    if not layer_map:
        layer_map = {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": {"hidden_size": 320, "heads": 40, "d_h": 8, "is_cross": True},
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": {"hidden_size": 640, "heads": 80, "d_h": 8, "is_cross": True},
        }
    return layer_map

# --------------------------- Attention Processor ---------------------------
class AdapterAttentionProcessor(nn.Module):
    def __init__(self, original_processor, adapter:VehicleAdapter, layer_name:str, layer_dim_info:dict=None):
        super().__init__()
        self.original_processor = original_processor
        self.adapter = adapter
        self.layer_name = layer_name
        self.layer_dim_info = layer_dim_info
        self.is_cross = "attn2" in layer_name

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # 原始文本注意力
        try:
            attn_text = self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            if isinstance(attn_text, tuple):
                attn_text = attn_text[0]
        except:
            attn_text = hidden_states

        # 获取adapter缓存
        Ks = self.adapter.last_Ks
        Vs = self.adapter.last_Vs
        g = self.adapter.last_g
        mask = self.adapter.last_mask
        if Ks is None or Vs is None or g is None or mask is None:
            return attn_text

        # 获取当前层维度
        if self.layer_dim_info is None:
            return attn_text
        tgt_heads = self.layer_dim_info["heads"]
        tgt_dh = self.layer_dim_info["d_h"]
        tgt_hidden = self.layer_dim_info["hidden_size"]

        # 获取当前层KV
        safe_key = self.adapter.layer_name_map.get(self.layer_name, sanitize_key(self.layer_name))
        K_obj = Ks.get(safe_key, list(Ks.values())[0]).to(hidden_states.device)
        V_obj = Vs.get(safe_key, list(Vs.values())[0]).to(hidden_states.device)

        # 维度适配
        B, T_q, D_q = hidden_states.shape
        if D_q != tgt_heads * tgt_dh:
            proj = nn.Linear(D_q, tgt_heads * tgt_dh).to(hidden_states.device)
            hidden_states = proj(hidden_states)
            D_q = tgt_heads * tgt_dh

        # 调整KV维度
        Bk, H_a, N, d_a = K_obj.shape
        if (H_a != tgt_heads) or (d_a != tgt_dh):
            if safe_key in self.adapter.per_layer_kproj:
                kproj = self.adapter.per_layer_kproj[safe_key].to(hidden_states.device)
                vproj = self.adapter.per_layer_vproj[safe_key].to(hidden_states.device)
                
                K_perm = K_obj.permute(0,2,1,3).contiguous().view(Bk*N, H_a*d_a)
                V_perm = V_obj.permute(0,2,1,3).contiguous().view(Bk*N, H_a*d_a)
                
                K_out = kproj(K_perm).view(Bk, N, tgt_heads, tgt_dh).permute(0,2,1,3).contiguous()
                V_out = vproj(V_perm).view(Bk, N, tgt_heads, tgt_dh).permute(0,2,1,3).contiguous()
                
                K_obj = K_out
                V_obj = V_out

        # 映射gate到目标heads
        g_orig = g
        try:
            Bg, Ng, Hg, _ = g_orig.shape
        except Exception:
            g_orig = g_orig.unsqueeze(-1)
            Bg, Ng, Hg, _ = g_orig.shape
        if Hg != tgt_heads:
            gname = f"g_{sanitize_key(self.layer_name)}"
            if gname not in self.adapter.per_layer_gateproj:
                self.adapter.per_layer_gateproj[gname] = nn.Linear(Hg, tgt_heads).to(hidden_states.device)
            gproj = self.adapter.per_layer_gateproj[gname].to(hidden_states.device)
            g_s = g_orig.squeeze(-1)
            g_flat = g_s.view(Bg * Ng, Hg)
            g_out = gproj(g_flat).view(Bg, Ng, tgt_heads).unsqueeze(-1)
        else:
            g_out = g_orig

        # 计算对象注意力
        attn_obj_out, alpha = compute_obj_attention(
            hidden_states, K_obj, V_obj, 
            mask.to(hidden_states.device), g_out.to(hidden_states.device)
        )
        self.adapter.last_alpha[safe_key] = alpha.detach().cpu()
        
        # 全局投影
        if safe_key in self.adapter.per_layer_global_proj:
            global_proj = self.adapter.per_layer_global_proj[safe_key].to(hidden_states.device)
        else:
            global_proj = nn.Linear(D_q, D_q).to(hidden_states.device)
        attn_global = global_proj(hidden_states)
        
        # 特征融合
        final = attn_text + LAMBDA_GLOBAL * attn_global + LAMBDA_OBJ * attn_obj_out
        return final
    

# --------------------------- 损失计算 ---------------------------
triplet_loss_fn = nn.TripletMarginLoss(margin=0.3, p=2, reduction='mean')

def unet_infer(vae, unet, scheduler, images, text_cond, device):
    B = images.shape[0]
    
    # VAE编码
    with torch.no_grad():
        enc_out = vae.encode(images)
        z0 = enc_out.latent_dist.mode().to(device)
    
    # 随机加噪
    timestep = torch.randint(1, scheduler.config.num_train_timesteps, (1,), device=device).item()
    timesteps = torch.full((B,), timestep, device=device, dtype=torch.long)
    eps = torch.randn_like(z0)
    z_t = scheduler.add_noise(z0, eps, timesteps)
    
    # UNet去噪
    unet_out = unet(z_t, timesteps, encoder_hidden_states=text_cond)
    eps_pred = unet_out.sample if hasattr(unet_out, "sample") else unet_out[0]
    
    # 去噪得到z0'
    z0_prime = []
    for i in range(B):
        step_out = scheduler.step(eps_pred[i:i+1], timestep, z_t[i:i+1])
        z0_prime.append(step_out.prev_sample)
    z0_prime = torch.cat(z0_prime, dim=0)
    
    # VAE解码
    with torch.no_grad():
        img_pred = vae.decode(z0_prime / vae.config.scaling_factor).sample
    
    # 扩散损失
    L_diff = F.mse_loss(eps_pred, eps)
    
    return img_pred, z0_prime, L_diff


def compute_losses(vae, unet, scheduler, adapter, 
                  images, pos_images, neg_images, text_cond,
                  boxes_coords, box_mask, masks, vehicle_encoder,
                  orig_id_embs, pos_id_embs, neg_id_embs):
    device = images.device
    B = images.shape[0]
    
    # 标准化box_mask
    if box_mask is None:
        box_mask = torch.zeros(B, 1, dtype=torch.float32, device=device)
    else:
        while box_mask.dim() > 2 and box_mask.size(-1) == 1:
            box_mask = box_mask.squeeze(-1)
        if box_mask.dim() == 1:
            box_mask = box_mask.unsqueeze(1)
        box_mask = box_mask.to(device)
    
    # 两次UNet推理
    img1_pred, z0_prime_1, L_diff_1 = unet_infer(vae, unet, scheduler, images, text_cond, device)
    img2_pred, z0_prime_2, L_diff_2 = unet_infer(vae, unet, scheduler, pos_images, text_cond, device)
    L_diff = (L_diff_1 + L_diff_2) / 2
    
    # 计算L_region（区域损失）
    mask_2d = masks[:, 0, :, :]
    mask_3d = mask_2d.unsqueeze(1)
    img1_pred_masked = img1_pred * mask_3d
    
    img1_pred_crops = roi_crop_and_resize(img1_pred_masked, boxes_coords)
    B2, N, C, h, w = img1_pred_crops.shape
    img1_pred_crops_flat = img1_pred_crops.view(B2*N, C, h, w)
    cam_label = torch.zeros(img1_pred_crops_flat.shape[0], dtype=torch.long).to(device)
    view_label = torch.zeros(img1_pred_crops_flat.shape[0], dtype=torch.long).to(device)
    id_pred_1 = vehicle_encoder(img1_pred_crops_flat, cam_label=cam_label, view_label=view_label).view(B2, N, -1)
    
    mask_expand = box_mask.unsqueeze(-1).float()
    num_valid = mask_expand.sum()
    if num_valid > 0:
        L_region = F.mse_loss(id_pred_1 * mask_expand, orig_id_embs * mask_expand, reduction='sum') / (num_valid + 1e-8)
    else:
        L_region = torch.tensor(0.0, device=device)
    
    # 计算L_tri（三元组损失，依赖负样本）
    img2_pred_masked = img2_pred * mask_3d
    img2_pred_crops = roi_crop_and_resize(img2_pred_masked, boxes_coords)
    img2_pred_crops_flat = img2_pred_crops.view(B2*N, C, h, w)
    cam_label2 = torch.zeros(img2_pred_crops_flat.shape[0], dtype=torch.long).to(device)
    view_label2 = torch.zeros(img2_pred_crops_flat.shape[0], dtype=torch.long).to(device)
    id_pred_2 = vehicle_encoder(img2_pred_crops_flat, cam_label=cam_label2, view_label=view_label2).view(B2, N, -1)
    
    neg_img_masked = neg_images * mask_3d
    neg_img_crops = roi_crop_and_resize(neg_img_masked, boxes_coords)
    neg_img_crops_flat = neg_img_crops.view(B2*N, C, h, w)
    cam_label_neg = torch.zeros(neg_img_crops_flat.shape[0], dtype=torch.long).to(device)
    view_label_neg = torch.zeros(neg_img_crops_flat.shape[0], dtype=torch.long).to(device)
    id_neg = vehicle_encoder(neg_img_crops_flat, cam_label=cam_label_neg, view_label=view_label_neg).view(B2, N, -1)
    
    valid_mask = box_mask.bool()
    if valid_mask.sum() > 0:
        anchor = id_pred_1[valid_mask].mean(dim=0, keepdim=True)
        positive = id_pred_2[valid_mask].mean(dim=0, keepdim=True)
        negative = id_neg[valid_mask].mean(dim=0, keepdim=True)
        L_tri = triplet_loss_fn(anchor, positive, negative)
    else:
        L_tri = torch.tensor(0.0, device=device)
    
    # 总损失
    loss_total = L_diff + LAMBDA_REGION * L_region + LAMBDA_TRIPLET * L_tri
    
    return loss_total, L_diff, L_region, L_tri


# --------------------------- 主训练循环 ---------------------------
def train_loop(
    data_root,
    vehicle_encoder_cfg,
    vehicle_encoder_pth,
    epochs=1,
    lr=2e-4,
    save_every=10000
):
    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    
    # 加载预训练模型
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    
    # 冻结预训练模型
    for p in vae.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False
    for p in unet.parameters(): p.requires_grad = False

    # 加载车辆编码器
    vehicle_encoder, D_id = load_vehicle_encoder(vehicle_encoder_cfg, vehicle_encoder_pth, device=DEVICE)
    for p in vehicle_encoder.parameters(): p.requires_grad = False

    # 推断UNet层维度
    layer_dim_map = infer_unet_layer_dims(unet, default_heads=8)

    # 构建层名称映射
    original_keys = list(unet.attn_processors.keys()) if hasattr(unet, "attn_processors") else []
    layer_name_map = {}
    safe_keys = []
    for orig in original_keys:
        safe = sanitize_key(orig)
        layer_name_map[orig] = safe
        safe_keys.append(safe)
    
    # 构建Adapter
    C_type = len(type2idx)
    per_layer_dims = {}
    for orig_name, safe in layer_name_map.items():
        info = layer_dim_map.get(orig_name)
        if info is not None:
            per_layer_dims[safe] = {
                'heads': int(info['heads']), 
                'd_h': int(info['d_h']),
                'hidden_size': int(info['hidden_size'])
            }
    
    adapter = VehicleAdapter(
        D_id=D_id, 
        P_box=P_FOURIER, 
        C_type=C_type, 
        D_obj=D_obj,
        layers=safe_keys, 
        D_txt=D_txt, 
        H=H,
        per_layer_dims=per_layer_dims
    ).to(DEVICE)
    adapter.layer_name_map = layer_name_map

    # 设置Attention Processor
    new_processors = {}
    for layer_name in unet.attn_processors.keys():
        orig_proc = unet.attn_processors[layer_name]
        layer_info = layer_dim_map.get(layer_name, None)
        new_processors[layer_name] = AdapterAttentionProcessor(orig_proc, adapter, layer_name, layer_info).to(DEVICE)
    unet.set_attn_processor(new_processors)

    # 数据集
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), 
        T.ToTensor(), 
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = VehicleDataset(data_root, transform, type2idx)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        adapter.parameters(), 
        lr=lr, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    total_steps = max(1, epochs * len(dataloader))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # 日志
    csv_file = open(LOG_CSV, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step","loss_total","L_diff","L_tri","L_region"])
    csv_file.flush()

    # 训练循环
    step = 0
    adapter.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # 加载数据
            images = batch["image"].to(DEVICE)
            pos_images = batch["pos_image"].to(DEVICE)
            neg_images = batch["neg_image"].to(DEVICE)
            id_embs = batch["id_embs"].to(DEVICE)
            pos_embs = batch["pos_embs"].to(DEVICE)
            neg_embs = batch["neg_embs"].to(DEVICE)
            boxes_coords = batch["boxes_coords"].to(DEVICE)
            box_mask = batch["box_mask"].to(DEVICE)
            types = batch["types"].to(DEVICE)
            masks = batch["masks"].to(DEVICE)
            
            # 文本条件
            captions = batch.get("caption", None)
            if captions is None:
                texts = ["a photo of a car"] * BATCH
            else:
                texts = list(captions) if isinstance(captions, (list, tuple)) else [captions]

            inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                text_cond = text_encoder(**inputs).last_hidden_state
            
            # Adapter前向
            Ks, Vs, g, mask = adapter(id_embs, boxes_coords, types, box_mask)
            
            # 计算损失
            loss_total, L_diff, L_region, L_tri = compute_losses(
                vae, unet, scheduler, adapter,
                images, pos_images, neg_images, text_cond,
                boxes_coords, box_mask, masks, vehicle_encoder,
                id_embs, pos_embs, neg_embs
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            
            # 日志
            log_data = [step, loss_total.item(), L_diff.item(), L_tri.item(), L_region.item()]
            csv_writer.writerow(log_data)
            csv_file.flush()
            
            # 进度条
            pbar.set_postfix({
                "loss": f"{loss_total.item():.4f}",
                "diff": f"{L_diff.item():.4f}",
                "tri": f"{L_tri.item():.4f}",
                "region": f"{L_region.item():.4f}"
            })
            
            # 保存模型
            if step % save_every == 0 and step > 0:
                torch.save({
                    'adapter_state_dict': adapter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step
                }, os.path.join(SAVE_DIR, f"adapter_step{step}.pth"))
            
            step += 1

    # 清理
    csv_file.close()
    torch.save(adapter.state_dict(), os.path.join(SAVE_DIR, "adapter_final.pth"))
    print(f"\n训练完成！最终模型保存至 {os.path.join(SAVE_DIR, 'adapter_final.pth')}")


# --------------------------- main 入口 ---------------------------
if __name__ == "__main__":
    DATA_ROOT = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/3drealcar_dataset/processed_output"
    VEH_CFG = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/configs/VeRi/vit_transreid_stride.yml"
    VEH_PTH = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/vit_transreid_veri.pth"
    
    train_loop(
        DATA_ROOT, 
        VEH_CFG, 
        VEH_PTH, 
        epochs=10,
        lr=5e-5, 
        save_every=10000
    )