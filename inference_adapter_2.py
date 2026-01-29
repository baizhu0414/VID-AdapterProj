import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
# 彻底关闭所有无关警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from diffusers import StableDiffusionPipeline

# 从训练脚本导入核心组件和超参
from train_vehicle_adapter import (
    VehicleAdapter, AdapterAttentionProcessor,
    infer_unet_layer_dims, sanitize_key,
    D_id, P_FOURIER, D_obj, D_txt, H, type2idx
)

# --------------------------- 全局配置（与训练脚本严格一致） ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
C_TYPE = len(type2idx)
# 训练脚本硬编码的BATCH=2，推理时必须对齐
TRAIN_BATCH_SIZE = 2

# --------------------------- 加载适配器权重（兼容训练的state_dict） ---------------------------
def load_adapter_state(adapter, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'adapter_state_dict' in state:
        sd = state['adapter_state_dict']
    else:
        sd = state
    adapter.load_state_dict(sd, strict=False)
    adapter.to(device, dtype=torch.float32)
    adapter.eval()
    return adapter

# --------------------------- 从JSON加载特征并构造B=2的输入（核心修复） ---------------------------
def load_and_pad_feat_to_batch2(json_path, device):
    """
    加载特征并复制为batch size=2，匹配训练脚本的硬编码B=2
    返回：所有特征的shape均为[2, 1, D]或[2,4]（B=2）
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON特征文件不存在：{json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # 提取单目标特征
    first_image = json_data["image_results"][0]
    first_box = first_image["box_results"][0]
    
    # 基础特征（B=1）
    id_emb_1 = torch.tensor(first_box["embeddings"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,768]
    box_coords_1 = torch.tensor(first_box["box"], dtype=torch.float32, device=device).unsqueeze(0)  # [1,4]
    type_onehot_1 = torch.tensor(first_box["anno"]["type_onehot"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,8]
    box_mask_1 = torch.tensor([[1.0]], dtype=torch.float32, device=device)  # [1,1]（2维）
    
    # --------------------------- 核心修复：复制特征构造B=2 ---------------------------
    id_emb = torch.cat([id_emb_1, id_emb_1], dim=0)  # [2,1,768]
    box_coords = torch.cat([box_coords_1, box_coords_1], dim=0)  # [2,4]
    type_onehot = torch.cat([type_onehot_1, type_onehot_1], dim=0)  # [2,1,8]
    box_mask = torch.cat([box_mask_1, box_mask_1], dim=0)  # [2,1]（总元素数=2，匹配view(2,-1)）
    # --------------------------------------------------------------------------------
    
    # 打印特征信息
    print(f"✅ 从JSON加载特征并构造B=2成功")
    print(f"  车辆类型：{first_box['anno']['type']} | 归一化框坐标：{np.round(first_box['box'], 4)}")
    print(f"  ID特征维度：{id_emb.shape} | Box维度：{box_coords.shape} | Mask维度：{box_mask.shape}")
    return id_emb, box_coords, type_onehot, box_mask

# --------------------------- 核心推理函数 ---------------------------
def infer(adapter_path, json_feat_path, prompt, out_path, num_samples=1, device=DEVICE):
    # 1. 加载Stable Diffusion Pipeline（低版本兼容）
    print("🔧 加载Stable Diffusion模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        requires_safety_checker=False
    )
    # 仅对PyTorch模型做设备/精度转换
    pipe.unet = pipe.unet.to(device, dtype=torch.float32)
    pipe.vae = pipe.vae.to(device, dtype=torch.float32)
    pipe.text_encoder = pipe.text_encoder.to(device, dtype=torch.float32)
    # 关闭梯度 + 显存优化
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    # 2. 初始化车辆适配器（与训练严格一致）
    print("🔧 初始化车辆适配器...")
    layer_dim_map = infer_unet_layer_dims(pipe.unet, default_heads=8)
    original_keys = list(pipe.unet.attn_processors.keys())
    layer_name_map = {orig: sanitize_key(orig) for orig in original_keys}
    safe_keys = [sanitize_key(orig) for orig in original_keys]
    # 构建层维度字典
    per_layer_dims = {}
    for orig_name, safe in layer_name_map.items():
        info = layer_dim_map.get(orig_name)
        if info is not None:
            per_layer_dims[safe] = {
                'heads': int(info['heads']),
                'd_h': int(info['d_h']),
                'hidden_size': int(info['hidden_size'])
            }
    # 创建适配器
    adapter = VehicleAdapter(
        D_id=D_id,
        P_box=P_FOURIER,
        C_type=C_TYPE,
        D_obj=D_obj,
        layers=safe_keys,
        D_txt=D_txt,
        H=H,
        per_layer_dims=per_layer_dims
    )
    # 加载权重
    adapter = load_adapter_state(adapter, adapter_path, device)
    # 补全层名称映射（训练时的关键配置）
    adapter.layer_name_map = layer_name_map

    # 3. 挂载适配器注意力处理器
    print("🔧 挂载适配器注意力处理器...")
    new_processors = {}
    for layer_name in pipe.unet.attn_processors.keys():
        orig_proc = pipe.unet.attn_processors[layer_name]
        layer_info = layer_dim_map.get(layer_name, None)
        new_processors[layer_name] = AdapterAttentionProcessor(
            orig_proc, adapter, layer_name, layer_info
        ).to(device, dtype=torch.float32)
    pipe.unet.set_attn_processor(new_processors)

    # 4. 加载B=2的特征并初始化适配器缓存（关键！匹配训练硬编码）
    print(f"🔧 加载推理特征并构造B=2：{json_feat_path}")
    id_emb, box_coords, type_onehot, box_mask = load_and_pad_feat_to_batch2(json_feat_path, device)
    # 适配器前向：用B=2的特征初始化KV/gate/mask缓存
    with torch.no_grad():
        Ks, Vs, g, mask = adapter(id_emb, box_coords, type_onehot, box_mask)
    # 确认缓存初始化成功
    assert adapter.last_Ks is not None and adapter.last_Vs is not None, "适配器缓存初始化失败！"

    # 5. 图像生成（生成时仍用单样本，取缓存的第一个样本）
    print(f"🚀 开始生成图像：prompt='{prompt}'，样本数={num_samples}")
    generator = torch.Generator(device=device).manual_seed(42)
    images = []
    for i in range(num_samples):
        with torch.no_grad():
            # 生成时batch size=1，适配器会自动使用缓存的前1个样本特征
            img = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                height=512,
                width=512
            ).images[0]
        images.append(img)
        print(f"✅ 生成完成：第{i+1}/{num_samples}张")

    # 6. 保存图像
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for i, img in enumerate(images):
        save_path = out_path if num_samples == 1 else out_path.replace('.png', f'_{i}.png')
        img.save(save_path)
        print(f"💾 图像保存：{save_path}")
    print(f"\n🎉 推理完成！所有图像已保存至 {os.path.dirname(out_path)}")

# --------------------------- 命令行参数解析 ---------------------------
def cli():
    parser = argparse.ArgumentParser(description="车辆特征引导的SD推理适配器（兼容训练B=2硬编码）")
    # 必选参数
    parser.add_argument("--adapter", required=True, help="适配器权重路径（如adapter_final.pth）")
    parser.add_argument("--json", required=True, help="推理特征JSON文件路径")
    # 可选参数
    parser.add_argument("--prompt", default="a photo of a car, high resolution, realistic", help="生成提示词")
    parser.add_argument("--out", default="./output/gen_car.png", help="输出图像路径")
    parser.add_argument("--num_samples", type=int, default=1, help="生成样本数量（显存不足请设为1）")
    parser.add_argument("--device", default="cuda", help="运行设备（cuda/cpu）")
    
    args = parser.parse_args()
    # 执行推理
    infer(
        adapter_path=args.adapter,
        json_feat_path=args.json,
        prompt=args.prompt,
        out_path=args.out,
        num_samples=args.num_samples,
        device=args.device
    )


# ========== 程序入口（无修改） ==========
if __name__ == '__main__':
    """
    python inference_adapter_2.py \
        --adapter /media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/3drealcar_dataset/save_vehicle_adapter/adapter_final.pth \
        --json /media/c303-2/225b3449-be93-436c-9dca-9188d2e145c21/cuiy/3dgsEnhancedDiffusion/TransReID/3drealcar_dataset/processed_output/anns/2024_01_11_15_03_39.json \
        --prompt "a photo of a black SUV on the highway, 8k, realistic, sunny day" \
        --out ./output/black_suv.png \
        --num_samples 3 \
        --device cuda
    """
    cli()