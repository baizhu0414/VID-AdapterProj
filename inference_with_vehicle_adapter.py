import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae"
).to(DEVICE)

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet"
).to(DEVICE)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14"
).to(DEVICE)

scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
)

vae.eval()
unet.eval()
text_encoder.eval()

from collections import OrderedDict

# ========== 推断 UNet cross-attention 层 ==========
layer_dim_map = infer_unet_layer_dims(unet)
original_keys = list(unet.attn_processors.keys())

layer_name_map = {}
safe_keys = []
for orig in original_keys:
    safe = sanitize_key(orig)
    layer_name_map[orig] = safe
    safe_keys.append(safe)

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
    D_id=768,
    P_box=P_FOURIER,
    C_type=len(type2idx),
    D_obj=512,
    layers=safe_keys,
    D_txt=768,
    H=8,
    per_layer_dims=per_layer_dims
).to(DEVICE)

adapter.layer_name_map = layer_name_map

# ====== 加载训练好的 adapter 权重 ======
ckpt = torch.load("YOUR_TRAINED_ADAPTER.pth", map_location=DEVICE)
adapter.load_state_dict(ckpt, strict=False)
adapter.eval()

new_processors = {}
for layer_name in unet.attn_processors.keys():
    orig_proc = unet.attn_processors[layer_name]
    layer_info = layer_dim_map.get(layer_name, None)
    new_processors[layer_name] = AdapterAttentionProcessor(
        orig_proc, adapter, layer_name, layer_info
    ).to(DEVICE)

unet.set_attn_processor(new_processors)


def encode_prompt(prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    return text_embeddings


def prepare_vehicle_condition(id_emb, box, type_idx):
    id_emb = torch.tensor(id_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    box = torch.tensor(box, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    type_onehot = torch.zeros(len(type2idx))
    type_onehot[type_idx] = 1
    type_onehot = type_onehot.unsqueeze(0).unsqueeze(0).to(DEVICE)

    box_mask = torch.ones(1,1).to(DEVICE)

    with torch.no_grad():
        adapter(id_emb, box, type_onehot, box_mask)


@torch.no_grad()
def generate(prompt, steps=30, guidance_scale=7.5):
    text_cond = encode_prompt(prompt)

    # classifier-free guidance
    uncond = encode_prompt("")
    text_cond = torch.cat([uncond, text_cond], dim=0)

    latents = torch.randn((1, 4, 64, 64)).to(DEVICE)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(steps)

    for t in scheduler.timesteps:
        latent_model_input = torch.cat([latents]*2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_cond
        ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image = vae.decode(latents / vae.config.scaling_factor).sample
    image = (image.clamp(-1,1) + 1) / 2
    return image


# 1️⃣ 准备车辆条件（示例）
vehicle_id_embedding = np.random.randn(768)   # 换成你的TransReID输出
vehicle_box = [0.3, 0.4, 0.7, 0.8]             # 归一化框
vehicle_type_idx = type2idx["SUV"]

prepare_vehicle_condition(vehicle_id_embedding, vehicle_box, vehicle_type_idx)

# 2️⃣ 生成
img = generate("a realistic photo of a SUV on a city street")

# 3️⃣ 保存
from torchvision.utils import save_image
save_image(img, "vehicle_controlled.png")
