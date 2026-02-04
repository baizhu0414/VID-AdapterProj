"""
双损失设计：扩散损失 + 车辆ID一致性损失
MLP + Transformer 融合ID/box/type特征
对齐IP-Adapter原生实现：文本Token+图像Prompt Token拼接注入注意力
核心要求：1.直接加载anns中embeddings；2.去噪图无mask提标准ID

对其IP-Adapter的注意力处理器，替换交叉注意力层
"""

import os
import math
import json
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 扩散/Transformer相关库
import accelerate
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer

from model.backbones.vit_pytorch import TransReID


# -------------------------- 全局配置 & 超参数（可根据需求调整） --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BANDS = 6  # box傅里叶编码频率数 → 4*6*2=48维
BATCH_SIZE = 8
TOTAL_EPOCHS = 50 # 轮次数
SAVE_INTERVAL = 10  # 每N轮保存一次模型
LEARNING_RATE = 5e-5 # 如果fp16不稳定则调低学习率
WEIGHT_DECAY = 1e-3 # 降低对小参数的压制
GRAD_CLIP_NORM = 0.5
SAVE_DIR = Path("./saved_vehicle_model")  # 模型保存目录
SAVE_PIC_DIR = Path("./saved_vehicle_pics")  # 生成图片保存目录
SAVE_DIR.mkdir(exist_ok=True, parents=True)
SAVE_PIC_DIR.mkdir(exist_ok=True, parents=True)
SEED = 42  # 随机种子（保证复现）
NUM_TOKENS = 4  # IP-Adapter图像提示Token数（原生默认4）
TRANSFORMER_NHEAD = 8  # 特征融合Transformer头数 768/8=96
TRANSFORMER_LAYERS = 2  # 特征融合Transformer层数

# 车辆类型（8类，与预处理脚本一致）
VEHICLE_TYPES = ["Sedan","SUV","MPV","Pickup","Van","Big Truck","Sports Car","Bus"]
type2idx = {v:i for i,v in enumerate(VEHICLE_TYPES)}

# 路径配置（请确认本地路径正确）
DATASET_ROOT = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/3drealcar_dataset/processed_output/anns"
VEH_CFG = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/configs/VeRi/vit_transreid_stride.yml"
VEH_PTH = "/media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/vit_transreid_veri.pth"

# -------------------------- 工具函数：随机种子/box傅里叶编码 --------------------------
def set_seed(seed=SEED):
    """固定所有随机种子，保证训练复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fourier_feats_sin_cos(box_coords: torch.Tensor, num_bands: int = NUM_BANDS) -> torch.Tensor:
    """
    box坐标傅里叶编码（正余弦），输出48维特征
    输入: box_coords [4] 单box / [B,4] 批量box
    输出: fourier [48] / [B,48] 编码特征
    """
    if box_coords.dim() == 1:
        box_coords = box_coords.unsqueeze(0)
    B = box_coords.shape[0]
    device = box_coords.device
    scales = (2 ** torch.arange(num_bands, device=device).float()) * math.pi
    coords = box_coords.unsqueeze(-1) * scales
    coords = torch.clamp(coords, -10*math.pi, 10*math.pi)  # 裁剪数值，防止sin/cos溢出
    sin = torch.sin(coords)
    cos = torch.cos(coords)
    fourier = torch.cat([sin, cos], dim=-1).view(B, -1)
    return fourier.squeeze(0) if B == 1 else fourier

# -------------------------- TransReID相关：车辆ID特征提取（冻结，无梯度） --------------------------
def get_cfg():
    """加载yaml配置并封装为可属性访问的对象（适配TransReID）"""
    with open(VEH_CFG, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 配置封装类
    class ConfigDict(dict):
        def __getattr__(self, key): return self[key]
        def __setattr__(self, key, value): self[key] = value
    cfg = ConfigDict(cfg)
    # 递归转换
    def recursive_convert(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = ConfigDict(v)
                recursive_convert(d[k])
    recursive_convert(cfg)
    return cfg

def build_transforms(cfg):
    """构建TransReID专用预处理（基于张量，支持批量处理，无需PIL）"""
    normalize = transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    test_transform = transforms.Compose([
        # 替换PIL的Resize为张量的Resize，antialias=True提升画质
        transforms.Resize(cfg.INPUT.SIZE_TEST, antialias=True),
        normalize,  # 直接对张量做归一化，无需ToTensor（VAE解码已是张量）
    ])
    return test_transform


def build_model(cfg, num_classes=0):
    """构建TransReID模型并加载预训练权重（无分类头，仅特征提取）"""
    model = TransReID(
        img_size=cfg.INPUT.SIZE_TEST,
        patch_size=16,
        stride_size=cfg.MODEL.STRIDE_SIZE,
        in_chans=3,
        num_classes=num_classes,
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
        sie_xishu=cfg.MODEL.SIE_COE if hasattr(cfg.MODEL, 'SIE_COE') else 1.0
    )
    # 加载权重（去除module.前缀，宽松加载）【修复torch.load警告】
    checkpoint = torch.load(VEH_PTH, map_location='cpu', weights_only=True)  # 新增weights_only=True
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


def get_vehicle_id_emb(img_tensor, vehicle_encoder, transform, device=DEVICE):
    """
    标准流程提取车辆ID特征（无mask操作），适配VAE解码张量/PIL图片
    :param img_tensor: [B,3,H,W] VAE解码张量（[-1,1]）/ PIL图片列表
    :return: [B,768] 归一化ID特征（保留x_hat的梯度，反传至上游）
    """
    # 处理PIL图片输入
    if isinstance(img_tensor, list) and all(isinstance(im, Image.Image) for im in img_tensor):
        img_tensor = torch.stack([transform(im) for im in img_tensor], dim=0).to(device)
    # 处理VAE解码张量（[-1,1]→[0,1]，再做TransReID标准化）
    else:
        img_tensor = (img_tensor + 1.0) / 2.0  # 像素范围转换，保留梯度
        # bs, c, h, w = img_tensor.shape
        # processed = []
        # for i in range(bs):
        #     # 转PIL时保留梯度（cpu()不阻断梯度，detach()才会）
        #     img_pil = transforms.ToPILImage()(img_tensor[i].cpu())
        #     processed.append(transform(img_pil).to(device))
        # img_tensor = torch.stack(processed, dim=0)
        img_tensor = transform(img_tensor)
    
    # 移除torch.no_grad()！TransReID已冻结，梯度不会回传至它，但会保留x_hat的梯度
    cam_label = torch.zeros(img_tensor.size(0), dtype=torch.long, device=device)
    view_label = torch.zeros(img_tensor.size(0), dtype=torch.long, device=device)
    feat = vehicle_encoder(x=img_tensor, cam_label=cam_label, view_label=view_label)
    feat = F.normalize(feat, p=2, dim=1)  # L2归一化，保留梯度
    return feat.squeeze()  # [B,768] 带梯度的张量


def load_vehicle_encoder(device=DEVICE):
    """加载冻结的TransReID编码器+专用预处理"""
    if not os.path.exists(VEH_CFG) or not os.path.exists(VEH_PTH):
        raise FileNotFoundError(f"TransReID文件不存在：{VEH_CFG} | {VEH_PTH}")
    cfg = get_cfg()
    model = build_model(cfg, num_classes=0)
    transform = build_transforms(cfg)
    # 严格冻结：禁止梯度更新
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] TransReID加载完成 | 特征维度768 | 预处理尺寸{cfg.INPUT.SIZE_TEST}")
    return model, transform, cfg

# -------------------------- 数据集类：直接加载anns中的embeddings作为原图ID --------------------------
class VehicleDataset(Dataset):
    def __init__(self, anns_dir, size=(256, 256)):
        self.anns_dir = Path(anns_dir)
        # 训练用图像/掩码变换（和TransReID预处理区分）
        self.transform = transforms.Compose([
            transforms.Resize(size, antialias=True),
            transforms.ToTensor(),  # [0,255]→[0,1]
        ])
        self.samples = self._load_samples()

    def _load_samples(self):
        """加载所有样本，直接读取anns中的embeddings"""
        samples = []
        for json_path in tqdm(self.anns_dir.glob("*.json"), desc="加载数据集（读取embeddings）"):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for img in data["image_results"]:
                img_path = Path(img["original_image_path"])
                for br in img["box_results"]:
                    # 基础特征提取
                    box = torch.tensor(br["box"], dtype=torch.float32)
                    type_str = br["anno"]["type"]
                    type_idx = type2idx.get(type_str, 0)
                    type_onehot = torch.eye(len(VEHICLE_TYPES))[type_idx].float()
                    mask_path = Path(br["mask_path"])
                    id_embs = torch.tensor(br["embeddings"], dtype=torch.float32)  # 直接加载ID特征
                    
                    if img_path.exists():
                        samples.append({
                            "image_path": img_path, "mask_path": mask_path,
                            "box": box, "type_onehot": type_onehot, "id_embs": id_embs
                        })
        print(f"[INFO] 数据集加载完成 | 共{len(samples)}个样本")
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        W, H = image.size  # 获取图像实际宽高（或直接用256，因为数据集已Resize）
        box = s["box"]  # [x1,y1,x2,y2]
        # box归一化到[0,1]：x/W, y/H
        box = box / torch.tensor([W, H, W, H], dtype=torch.float32)
        box_fourier = fourier_feats_sin_cos(box)  # 归一化后再编码
        image = self.transform(image) # 补上你遗漏的图像预处理
        return {
            "image": image, "id_emb": s["id_embs"],
            "box_fourier": box_fourier, "type_onehot": s["type_onehot"]
        }

# -------------------------- 核心模型：特征融合+IP-Adapter投影+原生注意力处理器 --------------------------
class ObjectFeatEncoder(nn.Module):
    """MLP+Transformer融合ID/box/type特征，输出对齐UNet的cross_attention_dim
    修复：充分利用Transformer所有交互特征，仅一次L2归一化，移除裁剪
    """
    def __init__(self, id_dim=768, box_dim=48, type_dim=8, cross_attention_dim=768,
                 nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_LAYERS):
        super().__init__()
        # MLP投影层：将各特征投影到统一维度（带偏置，对齐原生）
        self.id_proj = nn.Linear(id_dim, cross_attention_dim, bias=True)  # ID主导投影
        self.box_proj = nn.Linear(box_dim, cross_attention_dim, bias=True)
        self.type_proj = nn.Linear(type_dim, cross_attention_dim, bias=True)
        self.proj_norm = nn.LayerNorm(cross_attention_dim)
        
        # Transformer Encoder：捕捉特征间关联（对齐SDXL的Transformer配置）
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=cross_attention_dim, nhead=nhead, dim_feedforward=cross_attention_dim*4,
            batch_first=True, norm_first=True, dropout=0.1, activation="gelu", bias=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(cross_attention_dim)
        # 可学习权重：融合Transformer聚合特征和原始ID特征，保证ID主导（初始70%ID）
        self.id_fusion_weight = nn.Parameter(torch.tensor(0.7))  # 可学习，归一化到[0,1]

    def forward(self, id_emb, box_fourier, type_onehot):
        """
        输入：id_emb[B,768] | box_fourier[B,48] | type_onehot[B,8]
        输出：feat[B,1,768] 融合特征（ID主导，含box/type交互信息）
        """
        # 1. 各特征独立投影+增加序列维度 [B,768] → [B,1,768]
        x_id = self.id_proj(id_emb).unsqueeze(1)
        x_box = self.box_proj(box_fourier).unsqueeze(1)
        x_type = self.type_proj(type_onehot).unsqueeze(1)
        
        # 2. 拼接为3个Token，做层归一化 [B,1,768]×3 → [B,3,768]
        tokens = torch.cat([x_id, x_box, x_type], dim=1)
        tokens = self.proj_norm(tokens)
        
        # 3. Transformer编码特征交互 → [B,3,768]
        tokens = self.transformer_encoder(tokens)
        
        # 4. 平均汇聚所有Token的交互特征（充分利用ID/box/type的交互信息）→ [B,1,768]
        token_agg = torch.mean(tokens, dim=1, keepdim=True)
        
        # 5. ID主导融合：原始ID特征 + Transformer聚合特征（加权，可学习）
        w = torch.sigmoid(self.id_fusion_weight)  # 权重归一化到[0,1]，避免负数
        feat = w * x_id + (1 - w) * token_agg  # w越大，ID特征占比越高
        
        # 6. 最终归一化+L2归一化（仅一次，保证特征范数为1，提升稳定性）
        feat = self.final_norm(feat)
        feat = F.normalize(feat, p=2, dim=-1, eps=1e-8)
        
        return feat


class ImageProjModel(nn.Module):
    """对齐IP-Adapter原生+InstantID：融合特征→图像Prompt Token（MLP结构）
    修复：移除冗余L2归一化，显式声明偏置，贴近原生实现
    """
    def __init__(self, cross_attention_dim=768, num_tokens=NUM_TOKENS):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        # 原生IP-Adapter的MLP投影结构（显式带偏置，GELU激活，两层线性）
        self.proj = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim * num_tokens, bias=True),
            nn.GELU(),
            nn.Linear(cross_attention_dim * num_tokens, cross_attention_dim * num_tokens, bias=True),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)  # 对每个Token做层归一化

    def forward(self, feat):
        """
        输入：feat[B,1,768] 融合特征（ObjectFeatEncoder输出）
        输出：ip_tokens[B,NUM_TOKENS,768] 图像Prompt Token（适配IP-Adapter）
        """
        B = feat.shape[0]
        # 挤压序列维度 → [B,768]
        feat_flat = feat.squeeze(1)
        # MLP投影 → 重塑为Token序列 [B,768*N] → [B,N,768]
        ip_tokens = self.proj(feat_flat).view(B, self.num_tokens, self.cross_attention_dim)
        # 对每个Token做层归一化（提升数值稳定性）
        ip_tokens = self.norm(ip_tokens)
        
        return ip_tokens


class IPAttnProcessor(AttnProcessor2_0):
    """IP-Adapter + InstantID 注意力处理器（显式分离文本/图像 Token）
    改进：
      1. image token 数量固定 NUM_TOKENS，由上游 ImageProjModel 输出保证
      2. 文本 token 与 image token 自动匹配，无需 runtime pad
      3. K/V 分支完全解耦，scale 可学习
    """
    def __init__(self, hidden_size, cross_attention_dim=768, scale=1.0, num_tokens=NUM_TOKENS):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens  # 上游 ImageProjModel 输出 token 数量
        self.scale = nn.Parameter(torch.tensor(scale))  # 可学习图像分支权重
        self.to_k_ip = None
        self.to_v_ip = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        hidden_states: UNet 当前 hidden states [B, seq_len, hidden_size]
        encoder_hidden_states: 拼接文本 token + image token [B, seq_len_total, cross_attention_dim]
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, seq_len, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype

        # -------------------------------
        # 1. 计算 Query
        # -------------------------------
        query = attn.head_to_batch_dim(attn.to_q(hidden_states)) * attn.scale  # [B*num_heads, seq_len, head_dim]

        # -------------------------------
        # 2. 分离文本 token / image token
        #    image token 数量固定 self.num_tokens
        # -------------------------------
        if encoder_hidden_states.shape[1] >= self.num_tokens:
            text_hs  = encoder_hidden_states[:, :-self.num_tokens, :]
            image_hs = encoder_hidden_states[:, -self.num_tokens:, :]
        else:
            # Batch 内文本 token 太少，但 image token 固定，由 ImageProjModel 输出保证
            text_hs  = encoder_hidden_states
            # 构造 image token 全零（理论上不应出现，因为上游输出保证 NUM_TOKENS）
            image_hs = torch.zeros(B, self.num_tokens, self.cross_attention_dim,
                                   device=device, dtype=dtype)

        text_hs = text_hs.to(device=device, dtype=dtype)
        image_hs = image_hs.to(device=device, dtype=dtype)

        # -------------------------------
        # 3. 懒初始化图像分支 K/V 投影
        # -------------------------------
        if self.to_k_ip is None or self.to_v_ip is None:
            hidden_dim = attn.to_k.out_features
            self.to_k_ip = nn.Linear(self.cross_attention_dim, hidden_dim, bias=False).to(device, dtype=dtype)
            self.to_v_ip = nn.Linear(self.cross_attention_dim, hidden_dim, bias=False).to(device, dtype=dtype)

        # -------------------------------
        # 4. 文本分支 K/V
        # -------------------------------
        key_text   = attn.head_to_batch_dim(attn.to_k(text_hs))
        value_text = attn.head_to_batch_dim(attn.to_v(text_hs))

        # -------------------------------
        # 5. 图像分支 K/V
        # -------------------------------
        key_image   = attn.head_to_batch_dim(self.to_k_ip(image_hs))
        value_image = attn.head_to_batch_dim(self.to_v_ip(image_hs))

        # -------------------------------
        # 6. 拼接 K/V（scale_sig 可学习）
        # -------------------------------
        scale_sig = torch.sigmoid(self.scale)
        key   = torch.cat([key_text, key_image * scale_sig], dim=1)
        value = torch.cat([value_text, value_image * scale_sig], dim=1)

        # -------------------------------
        # 7. 注意力分数
        # -------------------------------
        attn_scores = torch.matmul(query, key.transpose(-2, -1))

        # -------------------------------
        # 8. 注意力掩码
        # -------------------------------
        if attention_mask is not None:
            if attention_mask.shape[-1] != key.shape[2]:
                pad_len = key.shape[2] - attention_mask.shape[-1]
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0.0)
            attention_mask = attention_mask.view(B, 1, 1, -1)
            attn_scores = attn_scores + attention_mask

        # -------------------------------
        # 9. 注意力概率 + dropout
        # -------------------------------
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=attn.dropout, training=attn.training, inplace=False)

        # -------------------------------
        # 10. 注意力加权求和 + reshape
        # -------------------------------
        hidden_states = torch.matmul(attn_probs, value).transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(B, seq_len, self.hidden_size)

        # -------------------------------
        # 11. 输出投影
        # -------------------------------
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_ip_attn(unet, num_tokens=NUM_TOKENS):
    """给UNet注册IP-Adapter原生注意力处理器（仅替换交叉注意力attn2）"""
    attn_procs = {}
    replaced_num = 0  # 统计替换数量
    for name in unet.attn_processors.keys():
        # 修正匹配条件：包含attn2 + 以processor结尾（兼容所有diffusers版本）
        if "attn2" in name and name.endswith("processor"):
            replaced_num +=1
            # 根据UNet各层结构获取hidden_size（原生IP-Adapter逻辑）
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])  # 更鲁棒的块ID提取
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            # 初始化处理器
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=unet.config.cross_attention_dim,
                scale=random.uniform(0.8, 1.2),  # 随机初始化0.8~1.2，提升灵活性
                num_tokens=num_tokens
            )
        else:  # 自注意力使用默认处理器
            attn_procs[name] = AttnProcessor2_0()
    unet.set_attn_processor(attn_procs)
    print(f"[INFO] UNet注册IP-Adapter处理器完成 | 成功替换{replaced_num}个交叉注意力层")
    return unet


# -------------------------- 损失函数：扩散损失 + 车辆ID一致性损失（无mask） --------------------------
# 替换原compute_loss函数中的ID损失计算部分，将余弦相似度损失改为对比损失（InfoNCE）,新增对比损失+动态加权
def compute_contrastive_loss(feat1, feat2, temperature=0.15):
    """InfoNCE对比损失，适合特征匹配，数值稳定性高"""
    feat1 = F.normalize(feat1, p=2, dim=1, eps=1e-8)
    feat2 = F.normalize(feat2, p=2, dim=1, eps=1e-8)
    sim = torch.matmul(feat1, feat2.T) / temperature
    batch_size = feat1.shape[0]
    label = torch.arange(batch_size, device=feat1.device)
    loss = F.cross_entropy(sim, label)
    return loss


def compute_loss(noise_pred, noise, latents, noisy_latents, t, vae, 
                 vehicle_encoder, vehicle_transform, id_emb_original,
                 noise_scheduler, accelerator, global_step, dataloader):
    """
    双损失计算：
    1. 扩散重建损失：UNet预测噪声与真实噪声的MSE
    2. ID一致性损失：去噪图无mask提ID 匹配 原图anns中的ID
    【核心修复】绕过原生step的if-else，手动批量计算z0_hat，解决布尔判断歧义
    """
    # 1. 扩散重建损失（不变）
    diff_loss = F.mse_loss(noise_pred, noise)

    # 2. 车辆ID一致性损失（无mask操作）
    with torch.no_grad():
        # 基础修复：确保t是一维张量+设备对齐
        t = t.squeeze().to(noisy_latents.device)
        bsz = noisy_latents.shape[0]
        
        # ========== 终极修复：手动批量计算z0_hat，替代原生step的if-else逻辑 ==========
        # 获取Scheduler的核心张量（已移到GPU）
        alphas_cumprod = noise_scheduler.alphas_cumprod
        one = noise_scheduler.one
        # 计算批量的prev_t，避免原生step的if-else歧义
        prev_t = t - 1
        # 用torch.where替代if-else，批量处理prev_t >=0的判断（核心！）
        alpha_prod_t = alphas_cumprod[t] if t.ndim == 0 else alphas_cumprod[t].view(bsz, 1, 1, 1)
        alpha_prod_t_prev = torch.where(
            prev_t >= 0,
            alphas_cumprod[prev_t] if prev_t.ndim ==0 else alphas_cumprod[prev_t].view(bsz,1,1,1),
            one.view(1,1,1,1)
        )
        beta_prod_t = 1 - alpha_prod_t
        # 手动批量计算去噪后的z0_hat（和原生step的pred_original_sample公式完全一致）
        z0_hat = (noisy_latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        # 裁剪z0_hat（和原生Scheduler的clip_denoised逻辑一致，避免值溢出）
        if noise_scheduler.config.clip_sample:  # 核心：clip_denoised → clip_sample
            clip_range = noise_scheduler.config.clip_sample_range  # 用官方裁剪范围参数
            z0_hat = torch.clamp(z0_hat, -clip_range, clip_range)
        # ==========================================================================

        # VAE解码到像素空间x_hat（[-1,1]，和原生逻辑一致）
        x_hat = vae.decode(z0_hat / 0.18215).sample  # 去噪后的生成图
    
    if torch.any(torch.isnan(x_hat)) or torch.any(torch.isinf(x_hat)):
        print(f"Step {global_step}: x_hat出现NaN/Inf")
        exit()
    # 去噪图无mask提取ID特征（保留梯度，反传至UNet/IP-Adapter）
    id_emb_recon = get_vehicle_id_emb(x_hat, vehicle_encoder, vehicle_transform, device=accelerator.device)
    
    if torch.any(torch.isnan(id_emb_recon)) or torch.any(torch.isinf(id_emb_recon)):
        print(f"Step {global_step}: id_emb_recon出现NaN/Inf，x_hat均值={x_hat.mean().item()}")
        torch.save({"x_hat": x_hat, "z0_hat": z0_hat}, "nan_id_emb.pth")
        exit()
    # 维度对齐+计算ID损失（不变）
    id_emb_original = id_emb_original.squeeze() if id_emb_original.dim() > 2 else id_emb_original
    id_emb_recon = id_emb_recon.squeeze() if id_emb_recon.dim() > 2 else id_emb_recon
    
    # ===================== 新增代码：保存x_hat + 计算ID特征余弦相似度 =====================
    if global_step % 2000 == 0 and accelerator.is_main_process:
        # 1. 计算ID特征平均余弦相似度（验证模型是否学到ID关联，越高越好）
        cos_sim = F.cosine_similarity(id_emb_recon, id_emb_original, dim=1, eps=1e-8)
        avg_cos_sim = cos_sim.mean().item()
        print(f"\n[Step {global_step}] ID特征平均余弦相似度：{avg_cos_sim:.4f}（目标：逐步提升至0.8+）")
        
        # 2. 保存第一张去噪图x_hat到本地，直观查看生成效果
        x_hat_vis = x_hat[0].detach().cpu()  # 取批次中第一张图，剥离梯度并移到CPU
        x_hat_vis = (x_hat_vis + 1.0) / 2.0  # 像素范围从[-1,1]转换为[0,1]（PIL要求）
        x_hat_vis = torch.clamp(x_hat_vis, 0.0, 1.0)  # 防止数值溢出
        x_hat_pil = transforms.ToPILImage()(x_hat_vis)  # 转PIL图像
        x_hat_pil.save(SAVE_PIC_DIR/f"x_hat_step_{global_step}.png")  # 保存
        print(f"[Step {global_step}] 去噪图已保存：x_hat_step_{global_step}.png")
    # ====================================================================================
    
    # id_loss = F.mse_loss(id_emb_recon, id_emb_original) # 这个损失不好收敛
    # 替换compute_loss函数中的ID损失计算行
    epoch = global_step // len(dataloader)  # 从全局步数推导当前epoch
    # 线性递增：前20轮从0.05升到0.2，20轮后保持0.2（平缓适配，避免跳升）
    id_weight = min(0.05 + epoch * (0.15 / 20), 0.2)
    id_loss = compute_contrastive_loss(id_emb_recon, id_emb_original)
    # cos_sim = F.cosine_similarity(id_emb_recon, id_emb_original, dim=1, eps=1e-8)  # 加eps防除零
    # cos_sim = torch.clamp(cos_sim, -1.0 + 1e-8, 1.0 - 1e-8)  # 裁剪范围，避免超出[-1,1]
    # id_loss = 1 - cos_sim  # 损失越小，余弦相似度越高，ID特征越匹配

    # 3. 总损失（ID损失加权，可调整系数）
    total_loss = diff_loss + id_weight * id_loss
    # 多卡训练损失同步（不变）
    return accelerator.gather(total_loss).mean(), diff_loss, id_loss


# -------------------------- 模型加载 & 冻结：仅训练自定义组件 --------------------------
def load_all_models(device=DEVICE):
    """加载所有模型，严格冻结非自定义组件，返回可训练参数"""
    # 1. 加载SD1.5基础模型（冻结）
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    
    # ========== 核心修复：将Scheduler所有张量移到训练设备（GPU） ==========
    for attr in dir(noise_scheduler):
        if not attr.startswith('_'):
            val = getattr(noise_scheduler, attr)
            if isinstance(val, torch.Tensor):
                setattr(noise_scheduler, attr, val.to(device))
    # ======================================================================
    
    # 2. 加载冻结的车辆ID编码器
    vehicle_encoder, vehicle_transform, _ = load_vehicle_encoder(device=device)
    
    # 3. 初始化自定义模型+注册IP-Adapter处理器
    cross_attention_dim = unet.config.cross_attention_dim  # SD1.5=768
    obj_encoder = ObjectFeatEncoder(cross_attention_dim=cross_attention_dim).to(device)
    image_proj = ImageProjModel(cross_attention_dim=cross_attention_dim).to(device)
    unet = register_ip_attn(unet, num_tokens=NUM_TOKENS)
    
    # 4. 严格冻结非自定义组件（核心！仅训练自定义部分）
    freeze_models = [vae, text_encoder, vehicle_encoder]
    for model in freeze_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    # UNet仅解冻注意力处理器的scale参数，其余全冻结
    for name, param in unet.named_parameters():
        param.requires_grad_(False)
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor):
            proc.scale.requires_grad_(True)
    
    # 5. 收集所有可训练参数
    trainable_params = list(obj_encoder.parameters()) + list(image_proj.parameters())
    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor):
            trainable_params.append(proc.scale)
    print(f"[INFO] 可训练参数总数：{sum(p.numel() for p in trainable_params)/1e6:.2f} M")
    
    return (vae, unet, tokenizer, text_encoder, noise_scheduler, 
            vehicle_encoder, vehicle_transform, obj_encoder, image_proj, trainable_params)


# 在main函数中，模型加载完成后添加钩子
def detect_nan_hook(grad):
    if torch.any(torch.isnan(grad)) or torch.any(torch.isinf(grad)):
        print("检测到梯度出现NaN/Inf，终止训练！")
        torch.save({"grad": grad}, "nan_grad.pth")
        exit()
    return grad


# -------------------------- 主训练函数 --------------------------
def main():
    # 1. 初始化加速器（混合精度/多卡适配）
    accelerator = Accelerator(mixed_precision="no", device_placement=True) # bf16
    device = accelerator.device
    set_seed()  # 固定随机种子

    # 2. 加载所有模型+可训练参数
    (vae, unet, tokenizer, text_encoder, noise_scheduler,
     vehicle_encoder, vehicle_transform, obj_encoder, image_proj, trainable_params) = load_all_models(device)

    # 3. 加载数据集+DataLoader（跨平台兼容num_workers）
    dataset = VehicleDataset(DATASET_ROOT)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4 if os.name == "posix" else 0, pin_memory=True, drop_last=True
    )

    # 4. 初始化优化器（仅优化可训练参数）
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY, 
        betas=(0.9, 0.999),
        eps=1e-8,  # 新增，提升数值稳定性
        foreach=False  # 关闭foreach，避免多卡下的梯度异常
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    # 5. 加速器准备（多卡/混合精度适配）
    (vae, unet, text_encoder,
    obj_encoder, image_proj, optimizer, lr_scheduler, dataloader) = accelerator.prepare(
        vae, unet, text_encoder,
        obj_encoder, image_proj, optimizer, lr_scheduler, dataloader
    )
    
    # 新增：一次性注册梯度NaN/Inf检测钩子（仅注册一次，关键修正！）
    print(f"[INFO] 注册梯度检测钩子，共{len(trainable_params)}个可训练参数")
    for p in trainable_params:
        p.register_hook(detect_nan_hook)

    # 6. 断点恢复
    start_epoch = 0
    global_step = 0
    latest_ckpt = sorted(SAVE_DIR.glob("vehicle_model_epoch_*.pth"))
    if latest_ckpt:
        ckpt_path = latest_ckpt[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        obj_encoder.load_state_dict(ckpt["obj_encoder"])
        image_proj.load_state_dict(ckpt["image_proj"])
        # optimizer.load_state_dict(ckpt["optimizer"])
        
        # 新增：加载IP注意力scale参数（关键！断点恢复训练连续）
        if "ip_attn_scales" in ckpt:
            ip_attn_scales = ckpt["ip_attn_scales"]
            for name, proc in unet.attn_processors.items():
                if isinstance(proc, IPAttnProcessor) and name in ip_attn_scales:
                    proc.scale.data = ip_attn_scales[name].to(device)
            print(f"[INFO] 成功加载{len(ip_attn_scales)}个IP注意力scale参数")
            
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("global_step", 0)
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print(f"[INFO] 从断点恢复训练 | epoch={start_epoch} | global_step={global_step}")

    # 8. 训练循环
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        obj_encoder.train()
        image_proj.train()
        total_loss, total_diff_loss, total_id_loss = 0.0, 0.0, 0.0

        # 进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}", disable=not accelerator.is_main_process)
        for batch in pbar:
            # 加载批次数据
            images = batch["image"].to(device)
            id_emb_original = batch["id_emb"].to(device)
            box_fourier = batch["box_fourier"].to(device)
            type_onehot = batch["type_onehot"].to(device)

            # VAE编码得到潜空间特征
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.mean * 0.18215

            # 给潜空间加噪
            noise = torch.randn_like(latents, device=device)
            bsz = latents.shape[0]
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)

            # 文本编码（空caption，仅用CLIP基础编码）
            with torch.no_grad():
                text_inputs = tokenizer(
                    [""]*bsz, padding="do_not_pad", truncation=True, return_tensors="pt"
                ).to(device)
                text_embs = text_encoder(text_inputs.input_ids).last_hidden_state

            # 自定义特征融合 + 生成IP Token + 拼接文本+图像Token（核心修正）
            feat = obj_encoder(id_emb_original, box_fourier, type_onehot)
            ip_tokens = image_proj(feat)
            encoder_hidden_states = torch.cat([text_embs, ip_tokens], dim=1)  # 拼接后传入UNet

            # UNet预测噪声（无ip_tokens参数，解决报错）
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred = torch.clamp(noise_pred, -10.0, 10.0)  # 裁剪噪声预测，限制FP16/32的合理范围
            if torch.any(torch.isnan(noise_pred)) or torch.any(torch.isinf(noise_pred)):
                print(f"Step {global_step}: noise_pred出现NaN/Inf")
                exit()

            # 计算双损失
            loss, diff_loss, id_loss = compute_loss(
                noise_pred, noise, latents, noisy_latents, t, vae,
                vehicle_encoder, vehicle_transform, id_emb_original,
                noise_scheduler, accelerator, global_step, dataloader
            )

            # 反向传播+梯度裁剪+优化器更新
            accelerator.backward(loss)
            # 新增：每100步打印核心参数的梯度范数，确认梯度传递.正常现象：梯度范数应在1e-4 ~ 1e-1之间，若为 0 则说明梯度消失，若远大于 1 则说明梯度爆炸
            if global_step % 1000 == 0 and accelerator.is_main_process:
                grad_norms = {}
                # 打印obj_encoder的ID投影层梯度
                if obj_encoder.id_proj.weight.grad is not None:
                    grad_norms["id_proj_grad"] = obj_encoder.id_proj.weight.grad.norm().item()
                # 打印IP scale的平均梯度
                scale_grads = [proc.scale.grad.norm().item() for proc in unet.attn_processors.values() if isinstance(proc, IPAttnProcessor) and proc.scale.grad is not None]
                if scale_grads:
                    grad_norms["avg_scale_grad"] = sum(scale_grads) / len(scale_grads)
                # 打印ID融合权重的梯度
                if obj_encoder.id_fusion_weight.grad is not None:
                    grad_norms["id_fusion_grad"] = obj_encoder.id_fusion_weight.grad.norm().item()
                print(f"Step {global_step} 梯度范数：{grad_norms}")
            # 新增：检测并置零异常梯度（NaN/Inf）
            for p in trainable_params:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            accelerator.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
            optimizer.step()
            # 新增：裁剪注意力scale参数，限制数值范围
            for proc in unet.attn_processors.values():
                if isinstance(proc, IPAttnProcessor):
                    proc.scale.data = torch.clamp(proc.scale.data, 0.01, 3.0) # 初始化(0.8,1.2)，裁剪(0.1,2.0)
            optimizer.zero_grad()

            # 损失统计
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            total_id_loss += id_loss.item()
            global_step += 1

            # 进度条更新
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff_loss": f"{diff_loss.item():.4f}",
                "id_loss": f"{id_loss.item():.4f}"
            })

        # 打印epoch平均损失
        avg_loss = total_loss / len(dataloader)
        avg_diff = total_diff_loss / len(dataloader)
        avg_id = total_id_loss / len(dataloader)
        if accelerator.is_main_process:
            print(f"\n[EPOCH {epoch+1}] 平均总损失：{avg_loss:.4f} | 扩散损失：{avg_diff:.4f} | ID损失：{avg_id:.4f}")
            lr_scheduler.step() # 学习率更新
            # 保存模型（仅主进程保存）
            if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == TOTAL_EPOCHS:
                save_dict = {
                    "obj_encoder": accelerator.unwrap_model(obj_encoder).state_dict(),
                    "image_proj": accelerator.unwrap_model(image_proj).state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    # 保存所有IP注意力的scale参数（断点恢复用）
                    "ip_attn_scales": {
                        name: proc.scale.data for name, proc in accelerator.unwrap_model(unet).attn_processors.items()
                        if isinstance(proc, IPAttnProcessor)
                    },
                    "epoch": epoch+1,
                    "global_step": global_step,
                    "cross_attention_dim": unet.config.cross_attention_dim,
                    "num_tokens": NUM_TOKENS,
                    "lr_scheduler": lr_scheduler.state_dict()  # 仅保存调度器状态，体积可忽略
                }
                save_path = SAVE_DIR / f"vehicle_model_epoch_{epoch+1}.pth"
                torch.save(save_dict, save_path)
                print(f"[INFO] 模型保存完成 | {save_path}\n")

if __name__ == "__main__":
    main()