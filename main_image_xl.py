import hashlib
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import faiss  # For k-NN graph
import argparse
import time
import math
from tqdm import tqdm, trange
from contextlib import contextmanager
from typing import Sequence, Union, Optional, List, Dict, Any, Tuple
from PIL import Image
import random

# === SD 3.5 Imports ===
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPModel, CLIPProcessor
from diffusers.models.modeling_utils import ModelMixin
from eval_image import evaluate_diversity
from datasets import load_dataset
from utils import *

# ========================= FLOPs profiling helpers (SD3.5 Updated) =========================
try:
    from thop import profile
    _THOP_AVAILABLE = True
except Exception:
    _THOP_AVAILABLE = False

def _to_flops_from_macs(macs: int) -> int:
    return int(2 * macs)

# 헬퍼: 모듈의 device, dtype 얻기
def _module_device_dtype(m: torch.nn.Module):
    try:
        p = next(m.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32

@torch.no_grad()
def profile_macs_text_encoder_sd3(text_encoders: List[torch.nn.Module], tokenizer_max_len: int = 77, batch: int = 1):
    """
    SD3는 최대 3개의 텍스트 인코더를 사용함 (CLIP L/G, T5).
    """
    if not _THOP_AVAILABLE or not text_encoders:
        return 0
    
    total_macs = 0
    for te in text_encoders:
        if te is None: continue
        dev, _ = _module_device_dtype(te)
        # T5는 max_len이 256 또는 512일 수 있으나, SD3 기본은 보통 77(CLIP) / 256 or 512(T5)
        # 여기서는 보수적으로 77 혹은 모델 config 참조가 좋으나, 프로파일링 목적상 77/256 구분
        curr_len = tokenizer_max_len
        if "T5" in te.__class__.__name__:
            curr_len = 256 # SD3 T5 default max length approx
            
        input_ids = torch.ones(batch, curr_len, dtype=torch.long, device=dev)
        try:
            # T5 등의 경우 forward signature가 다를 수 있어 예외처리
            macs, _ = profile(te, inputs=(input_ids,), verbose=False)
            total_macs += macs
        except Exception:
            pass # 일부 복잡한 인코더 구조는 thop 실패 가능성 있음
            
    return int(total_macs)

@torch.no_grad()
def profile_macs_transformer_sd3(transformer, height: int, width: int, batch: int = 1):
    """
    SD3 MMDiT Transformer 프로파일링.
    입력: hidden_states, encoder_hidden_states, pooled_projections, timestep
    """
    if not _THOP_AVAILABLE or transformer is None:
        return 0
    
    dev, dtype = _module_device_dtype(transformer)
    
    # SD3 Latent Channel = 16
    H8, W8 = height // 8, width // 8
    hidden_states = torch.randn(batch, 16, H8, W8, device=dev, dtype=dtype)
    
    # Encoder Hidden States (Context): SD3는 CLIP+T5 결합으로 4096차원 등 큼
    # 대략적인 max seq len 333 (77+256) 가정, dim 4096 (SD3 Large spec)
    encoder_hidden_states = torch.randn(batch, 333, 4096, device=dev, dtype=dtype)
    
    # Pooled Projections: [B, 2048]
    pooled_projections = torch.randn(batch, 2048, device=dev, dtype=dtype)
    
    timestep = torch.tensor([999], device=dev, dtype=dtype)

    # thop profile용 wrapper
    class _Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, h, e, p, t):
            return self.m(hidden_states=h, encoder_hidden_states=e, pooled_projections=p, timestep=t, return_dict=False)[0]

    wrapper = _Wrap(transformer).to(dev).to(dtype)
    
    try:
        macs, _ = profile(wrapper, inputs=(hidden_states, encoder_hidden_states, pooled_projections, timestep), verbose=False)
    except Exception as e:
        print(f"Transformer profiling failed: {e}")
        return 0
        
    return int(macs)

@torch.no_grad()
def profile_macs_vae_decode_sd3(vae, height: int, width: int, batch: int = 1):
    if not _THOP_AVAILABLE or vae is None:
        return 0
    dev, dtype = _module_device_dtype(vae)
    
    # SD3 Latent = 16 channels
    H8, W8 = height // 8, width // 8
    latents = torch.randn(batch, 16, H8, W8, device=dev, dtype=dtype)
    
    class _Wrap(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, z): 
            # SD3 VAE decode returns a DecoderOutput object
            return self.vae.decode(z).sample
            
    wrapper = _Wrap(vae).to(dev).to(dtype)
    try:
        macs, _ = profile(wrapper, inputs=(latents,), verbose=False)
    except Exception:
        return 0
    return int(macs)

@torch.no_grad()
def profile_macs_clip_image(clip_model: CLIPModel, batch: int = 1):
    if not _THOP_AVAILABLE or clip_model is None:
        return 0
    dev, dtype = _module_device_dtype(clip_model)
    x = torch.randn(batch, 3, 224, 224, device=dev, dtype=dtype)
    class _Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.get_image_features(pixel_values=x)
    wrapper = _Wrap(clip_model).to(dev).to(dtype)
    macs, _ = profile(wrapper, inputs=(x,), verbose=False)
    return int(macs)

BACKWARD_FACTOR = 2.0 
def load_prompts_from_coco_hf(
    dataset_id: str = "lmms-lab/COCO-Caption2017",   # 예: "coco_captions" 또는 instruction 변형 레포 id
    year: str = "2017",
    split: str = "val",            # "train" | "validation"
    start_row: int = 0,
    max_rows: int = 20,
    pick: str = "first",                  # "first" | "random" | "all"
    seed: int = 0,
    return_refs: bool = False,            # 참조 이미지/메타까지 받을지
) -> List[str] | List[Dict[str, Any]]:
    """
    스샷의 instruction 스타일과 표준 coco_captions를 모두 지원하는 범용 로더.
    - 반환(기본): 프롬프트(캡션) 문자열 리스트
    - 반환(return_refs=True): [{prompt, image_pil, id, idx, meta...}, ...]
    """
    rng = random.Random(seed)
    # year 인자를 지원하지 않는 레포도 있으니, 실패하면 year 없이 로드
    try:
        ds = load_dataset(dataset_id, split='val')
    except Exception:
        ds = load_dataset(dataset_id, split=split)

    N = len(ds)
    lo = max(0, start_row)
    hi = min(N, start_row + max_rows)

    outputs_prompts: List[str] = []
    outputs_refs: List[Dict[str, Any]] = []

    for i in range(lo, hi):
        rec = ds[i]
        caps = _normalize_caption_list(rec)
        if not caps:
            continue

        # pick 정책
        chosen_caps: List[str]
        if pick == "first":
            chosen_caps = [caps[0]]
        elif pick == "random":
            chosen_caps = [rng.choice(caps)]
        elif pick == "all":
            chosen_caps = caps
        else:
            raise ValueError(f"Unknown pick: {pick}")

        if not return_refs:
            outputs_prompts.extend(chosen_caps)
        else:
            img = _get_image_from_record(rec)
            # id/file_name 같은 메타 필드는 레포마다 다를 수 있으니 안전하게 가져오기
            meta = {
                "id": int(rec.get("id", i)) if isinstance(rec.get("id", i), (int, float)) else i,
                "file_name": rec.get("file_name"),
                "question_id": rec.get("question_id"),
                "coco_url": rec.get("coco_url"),
                "question": rec.get("question"),
                "answers": rec.get("answer"),
                "idx": i,
            }
            for cp in chosen_caps:
                outputs_refs.append({
                    "prompt": cp,
                    "image_pil": img,   # None일 수도 있음 (일부 변형 레포)
                    "meta": meta,
                })

    return outputs_refs if return_refs else outputs_prompts

def estimate_pipeline_flops_once_sd3(
    pipe: "StableDiffusion3Pipeline",
    clip_model: Optional[CLIPModel], # External CLIP for penalty
    height: int,
    width: int,
    steps: int,
    guidance_scale: float = 7.5,
    per_step_decode_224: bool = False,
    per_step_clip_image: bool = False,
) -> Dict[str, int]:
    if not _THOP_AVAILABLE:
        return {"total_flops": 0}

    # SD3 has up to 3 text encoders
    encoders = []
    if hasattr(pipe, "text_encoder") and pipe.text_encoder: encoders.append(pipe.text_encoder)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2: encoders.append(pipe.text_encoder_2)
    if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3: encoders.append(pipe.text_encoder_3)

    text_macs = profile_macs_text_encoder_sd3(encoders, tokenizer_max_len=77, batch=1)

    # CFG batch size
    cfg_batch = 2 if (guidance_scale > 1.0) else 1
    transformer_macs_step = profile_macs_transformer_sd3(pipe.transformer, height=height, width=width, batch=cfg_batch)

    vae_macs_final = profile_macs_vae_decode_sd3(pipe.vae, height=height, width=width, batch=1)

    vae_macs_step_224 = profile_macs_vae_decode_sd3(pipe.vae, height=height, width=width, batch=1) if per_step_decode_224 else 0
    clip_img_macs_step = profile_macs_clip_image(clip_model, batch=1) if per_step_clip_image else 0

    vae_macs_step_total  = vae_macs_step_224 * (1.0 + BACKWARD_FACTOR)
    clip_macs_step_total = clip_img_macs_step * (1.0 + BACKWARD_FACTOR)

    total_macs = text_macs \
        + steps * transformer_macs_step \
        + vae_macs_final \
        + steps * vae_macs_step_total \
        + steps * clip_macs_step_total

    return {
        "total_flops": int(2 * total_macs),
        "text_macs": int(text_macs),
        "transformer_macs_step": int(transformer_macs_step),
        "vae_macs_final": int(vae_macs_final),
        "clip_img_macs_step": int(clip_img_macs_step),
    }

# ========================= Utils & Helpers (Embedded) =========================
def _load_clip(model_id, device, dtype):
    model = CLIPModel.from_pretrained(model_id,use_safetensors=True).to(device, dtype=dtype)
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc

def _clip_img_emb(model, proc, image, device):
    inputs = proc(images=image, return_tensors="pt").to(device)
    return model.get_image_features(**inputs)

def _clip_image_emb_batch(model, proc, images, device):
    inputs = proc(images=images, return_tensors="pt", padding=True).to(device)
    return model.get_image_features(**inputs)

def _norm_last_dim(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

def pil_to_vae_latent(pipe, image, height, width, device):
    # SD3 VAE encode helper
    # Resize
    image = image.resize((width, height), resample=Image.BICUBIC)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) * 2.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).to(pipe.dtype)
    with torch.no_grad():
        # SD3 VAE scaling factor handling
        scaling = pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else 1.5305
        latents = pipe.vae.encode(tensor).latent_dist.sample() * scaling
    return latents

def vae_decode_224(pipe, latents):
    # Decode and resize to 224 for external CLIP
    # SD3 VAE decode
    scaling = pipe.vae.config.scaling_factor if hasattr(pipe.vae.config, "scaling_factor") else 1.5305
    latents = latents / scaling
    latents = latents.to(dtype=pipe.vae.dtype)
    image = pipe.vae.decode(latents).sample
    # Image is [-1, 1], convert to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    # Resize to 224
    image_224 = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    return image_224

# ========================= Scheduling =========================
def schedule_weights(i: int, 
                     mode: str = 'logistic',
                     L0: int = 20,
                     rep_max: float = 1.0,
                     hidden_max: float = 500.0,
                     sharpness: float = 0.1,
                     eps: float = 1e-4):
    
    if mode == 'constant':
        return rep_max, hidden_max

    elif mode == 'linear':
        if i >= L0:
            s = 0.0
        else:
            s = 1.0 - (i / float(L0))
            
    else: # logistic
        s = 1.0 / (1.0 + math.exp(sharpness * (i - L0)))

    w_rep_raw    = rep_max    * s
    w_hidden_raw = hidden_max * (1.0 - s)

    w_rep    = 0.0 if w_rep_raw < rep_max * eps else w_rep_raw
    w_hidden = hidden_max if (hidden_max - w_hidden_raw) < hidden_max * eps else w_hidden_raw

    return w_rep, w_hidden

# ========================= Main Logic (SD 3.5 Compatible) =========================
LAT_GRAD_COST_PER_ELEM = 4
NORM_COST_PER_ELEM     = 2
DOT_COST_PER_ELEM      = 1

def _ensure_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def generate_and_analyze_latents_per_step_Diffusion(
    pipe: "StableDiffusion3Pipeline",
    prompt: str,
    output_filepath: str,
    sim_ver: str, rep_ver: str,
    previous_responses: List[Any] = [],
    prev_probs: Any = None, prev_embs: Any = None,
    fill_order: List[int] = [],
    gen_length: int = 512, steps: int = 50,
    knn_k: int = 5,
    threshold: float = 0.1,
    temperature: float = 0.0,
    k_gamma: float = 0.5,
    L0: int = 20,
    constant: float = 1.0,
    prev_step_traces: Optional[List[Dict[str,Any]]] = None,
    topk_k: int = 64,
    use_noise_penalty: bool = True,
    use_latent_penalty: bool = True,
    noise_max: float = 1.0,
    latent_max: float = 0.25,
    sharpness: float = 0.1,
    avoid_image: Optional[Union[Image.Image, Sequence[Image.Image]]] = None,
    target_image: Optional[Image.Image] = None,
    guidance_scale: float = 7.5,
    height: int = 1024, width: int = 1024, # SD3 Default
    seed: int = 1234,
    device: str = "cuda",
    clip_model_id: str = "openai/clip-vit-base-patch32",
    apply_stride: int = 1,
    mode='logistic'
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, Optional[Dict[str,Any]]]:

    torch.manual_seed(seed)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # CLIP (External for penalty)
    clip_model, clip_proc = _load_clip(clip_model_id, device=device, dtype=torch.float16)

    avoid_imgs = _ensure_list(avoid_image)
    avoid_emb_cache: Optional[torch.Tensor] = None

    def _ensure_avoid_emb_cache():
        nonlocal avoid_emb_cache
        if avoid_emb_cache is not None: return
        if len(avoid_imgs) == 0:
            avoid_emb_cache = None
            return
        with torch.no_grad():
            embs = _clip_image_emb_batch(clip_model, clip_proc, avoid_imgs, device=device)
            embs = torch.nn.functional.normalize(embs.to(torch.float32).to(device), dim=-1)
        avoid_emb_cache = embs

    z_target = pil_to_vae_latent(pipe, target_image, height, width, device) if target_image is not None else None

    clip_sim_avoid_steps: List[float]    = []
    clip_sim_target_steps: List[float]   = []
    latent_cos_avoid_steps: List[float]  = []
    latent_cos_target_steps: List[float] = []
    eps_norm_steps: List[float]          = []
    step_indices: List[int]              = []
    latents_steps: List[torch.Tensor]    = []

    if prev_step_traces is None:
        prev_step_traces = []

    last_noise_pred = torch.zeros(1)
    last_latent     = torch.zeros(1)
    penalty_macs_accum = 0

    def make_callback():
        def _callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            nonlocal last_noise_pred, last_latent, penalty_macs_accum, avoid_emb_cache

            # SD3: Check latents key
            latents = callback_kwargs.get("latents")
            if latents is None: return callback_kwargs

            apply_now = (apply_stride <= 1) or (step % apply_stride == 0)
            
            _ensure_avoid_emb_cache()
            avoid_img_embs = avoid_emb_cache

            target_img_emb = _clip_img_emb(clip_model, clip_proc, target_image, device) if target_image is not None else None
            if target_img_emb is not None:
                target_img_emb = torch.nn.functional.normalize(target_img_emb.to(torch.float32), dim=-1).unsqueeze(0)

            if use_noise_penalty and use_latent_penalty:
                w_noise, w_latent = schedule_weights(step, L0=L0, rep_max=noise_max,
                                                     hidden_max=latent_max, sharpness=sharpness, mode=mode)
            elif use_noise_penalty:
                w_noise, w_latent = noise_max, 0.0
            elif use_latent_penalty:
                w_noise, w_latent = 0.0, latent_max
            else:
                w_noise, w_latent = 0.0, 0.0

            # SD3 Latents: [B, 16, H/8, W/8]
            B = latents.shape[0]
            last_latent = latents.detach().clone()

            H8, W8 = latents.shape[-2], latents.shape[-1]
            C = latents.shape[1] # Should be 16 for SD3
            M = C * H8 * W8

            # ---------- (A) Latent penalty ----------
            if apply_now and (w_latent > 0.0) and use_latent_penalty and (len(prev_step_traces) > 0):
                z_refs = []
                for tr in prev_step_traces:
                    if "latents_steps" in tr and step < len(tr["latents_steps"]):
                        zref = tr["latents_steps"][step]
                        if isinstance(zref, torch.Tensor): zr = zref
                        else: zr = torch.tensor(zref)
                        if zr.dim() == 3: zr = zr.unsqueeze(0) # [1, C, H8, W8]
                        z_refs.append(zr.to(latents.device, dtype=latents.dtype))

                if len(z_refs) > 0:
                    Z = torch.stack(z_refs, dim=0).to(torch.float32) # [R, B, C, H8, W8]
                    z_t = latents.to(torch.float32)

                    x_flat = z_t.view(B, -1)
                    Xn     = x_flat.norm(dim=1, keepdim=True).clamp_min(1e-6)

                    R = Z.shape[0]
                    cos_mat = []
                    for r in range(R):
                        y = Z[r] # [B, C, H8, W8]
                        y_flat = y.view(B, -1)
                        Yn     = y_flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        cos_r  = (x_flat * y_flat).sum(dim=1, keepdim=True) / (Xn * Yn)
                        cos_mat.append(cos_r)
                    cos_mat = torch.cat(cos_mat, dim=1)

                    idx     = torch.argmax(cos_mat, dim=1)
                    max_cos = cos_mat.gather(1, idx.unsqueeze(1)).mean()
                    latent_cos_avoid_steps.append(float(max_cos.item()))

                    # MACs approx
                    macs_norm_x = B * NORM_COST_PER_ELEM * M
                    macs_norm_y = B * R * NORM_COST_PER_ELEM * M
                    macs_dot    = B * R * DOT_COST_PER_ELEM * M
                    macs_grad   = B * LAT_GRAD_COST_PER_ELEM * M
                    penalty_macs_accum += (macs_norm_x + macs_norm_y + macs_dot + macs_grad)

                    # Gradients
                    grad_flat = torch.zeros_like(x_flat)
                    for b in range(B):
                        r = idx[b].item()
                        y = Z[r, b]
                        y_flat = y.view(-1)
                        xf = x_flat[b]
                        xn = Xn[b]
                        yn = y_flat.norm().clamp_min(1e-6)
                        cos = (xf @ y_flat) / (xn * yn)

                        term1 = y_flat / (xn * yn)
                        term2 = (cos / (xn**2)) * xf
                        g     = (term1 - term2)
                        grad_flat[b] = g

                    grad = grad_flat.view_as(z_t)
                    grad = _norm_last_dim(grad)
                    latents = (z_t - w_latent * grad).to(callback_kwargs["latents"].dtype).detach()

            # --- (B) Noise penalty via CLIP ----------
            if apply_now and (w_noise > 0.0) and use_noise_penalty and (len(prev_step_traces) > 0):
                with torch.enable_grad():
                    z32 = latents.detach().to(torch.float32).requires_grad_(True)
                    
                    # Decode and resize for CLIP (SD3 specific)
                    imgs_224 = vae_decode_224(pipe, z32)
                    
                    img_emb = clip_model.get_image_features(pixel_values=imgs_224.to(torch.float16))
                    img_emb = torch.nn.functional.normalize(img_emb.to(torch.float32), dim=-1)

                    D = int(img_emb.shape[-1])
                    if avoid_img_embs is not None:
                        N_avoid = int(avoid_img_embs.shape[0])
                        penalty_macs_accum += B * N_avoid * D + B * D
                    
                    loss = torch.tensor(0.0, device=img_emb.device, dtype=img_emb.dtype)

                    if avoid_img_embs is not None and avoid_img_embs.shape[0] > 0:
                        sims_img = torch.einsum('bd,nd->bn', img_emb, avoid_img_embs)
                        max_sim_img, _ = sims_img.max(dim=1)
                        loss = loss + max_sim_img.mean()
                        clip_sim_avoid_steps.append(float(max_sim_img.mean().item()))

                    if target_img_emb is not None:
                        tgt_sim = torch.einsum('bd,nd->bn', img_emb, target_img_emb).mean()
                        loss = loss - tgt_sim
                        clip_sim_target_steps.append(float(tgt_sim.item()))

                    grad_z, = torch.autograd.grad(loss, z32, retain_graph=False)
                
                grad_z = _norm_last_dim(grad_z)
                latents = (z32 - w_noise * grad_z).detach().to(callback_kwargs["latents"].dtype)

            # Record
            step_indices.append(int(step))
            eps_norm_steps.append(float(latents.norm().item()))
            latents_steps.append(latents.detach().cpu())

            callback_kwargs["latents"] = latents
            return callback_kwargs

        return _callback

    step_cb = make_callback()

    # ====== 파이프라인 실행 (SD3) ======
    # SD3 has different args. We map 'prompt' to primary prompt.
    out = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height, width=width,
        generator=torch.Generator(device).manual_seed(seed),
        callback_on_step_end=step_cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    img_pil = out.images[0]

    record = {
        'clip_sim_avoid_steps':    clip_sim_avoid_steps,
        'clip_sim_target_steps':   clip_sim_target_steps,
        'latent_cos_avoid_steps':  latent_cos_avoid_steps,
        'latent_cos_target_steps': latent_cos_target_steps,
        'eps_norm_steps':          eps_norm_steps,
        'step_indices':            step_indices,
        'latents_steps':           latents_steps,
        'penalty_macs_total':      int(penalty_macs_accum),
        'apply_stride':            int(apply_stride),
    }

    if output_filepath:
        img_pil.save(output_filepath)

    return img_pil, last_noise_pred, last_latent, record

# ===== Dummy prompt loader for execution =====
def load_prompts_from_coco_hf_generic():
    # Placeholder: replace with actual logic or just return sample
    return ["A cyberpunk cat sitting on a neon rooftop", "A serene landscape with mountains"]


# ===========================================================
# 메인 실행부 (Argument Parser & Main Loop 포함)
# ===========================================================

def main():
    # 1. Argument Parsing (빠짐없이 포함)
    parser = argparse.ArgumentParser(description="Generate images on COCO prompts via SD3.5 + Hybrid Avoidance.")
    
    parser.add_argument("--config", type=str, default="image_test/config_image.json", help="Path to config JSON")
    parser.add_argument("--result_suffix", type=str, default="sd3_coco_imgs", help="Suffix for output folder/files")
    
    # 사용자 요청: num_rows, start_row 등 데이터셋 제어 인자 복구
    parser.add_argument("--num_rows", type=int, default=20, help="Number of prompts to process")
    parser.add_argument("--start_row", type=int, default=0, help="Start index for prompts")
    
    parser.add_argument("--early_stop", action="store_true", help="Stop iteration if identical image detected")
    parser.add_argument("--es_cutoff", type=int, default=10, help="Early stop cutoff iteration")
    parser.add_argument("--print_flops", action="store_true", help="Print detailed FLOPs")
    
    args = parser.parse_args()

    # 2. Config Loading
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load config: {e}")
        return

    # ===== Config 파싱 & 변수 설정 =====
    # 모델 설정 (SD3.5 Turbo 기본값 적용)
    model_name     = cfg.get("model_name", "stabilityai/stable-diffusion-3.5-large-turbo")
    steps          = cfg.get("steps", 4)          # Turbo는 4~8 스텝 권장
    guidance_scale = cfg.get("guidance_scale", 1.5) # Turbo는 낮은 CFG 권장
    height         = cfg.get("height", 1024)
    width          = cfg.get("width", 1024)
    seed0          = cfg.get("seed", 42)
    randomize_seed = cfg.get("randomize_seed", False)
    num_iteration  = cfg.get("num_iteration", 5)

    # Penalty 설정
    use_noise_penalty  = cfg.get("use_rep_penalty", True)
    use_latent_penalty = cfg.get("use_hidden_penalty", True)
    noise_max          = cfg.get("rep_max_constant", 0.2)       # Turbo라 낮게 잡는 것 권장
    latent_max         = cfg.get("hidden_max_constant", 0.25)
    L0                 = cfg.get("L0", 2)                       # Step이 4개면 L0도 작아야 함
    sharpness          = cfg.get("sharpness", 0.5)
    apply_stride       = cfg.get("apply_stride", 1)             # Turbo는 무조건 1 권장
    mode               = cfg.get("mode", "logistic")            # 'logistic', 'linear', 'constant'

    # 타겟/회피 이미지/텍스트 설정
    avoid_image_path  = cfg.get("avoid_image_path")
    target_image_path = cfg.get("target_image_path")
    clip_model_id     = cfg.get("clip_model_id", "openai/clip-vit-base-patch32")
    
    # LLM Diversity 설정
    use_llm_diversity = cfg.get("use_llm_diversity", False)

    # 5. 저장 경로 설정
    base = os.path.splitext(os.path.basename(args.config))[0]
    cfg_dir = os.path.dirname(args.config)
    # 결과 폴더 생성
    base = os.path.splitext(args.config)[0]
    cfg_dir = os.path.dirname(base)
    save_base = os.path.join(cfg_dir, args.result_suffix) if args.result_suffix else base
    print(save_base)
    results_json_path = f"{save_base}_result.json"
    print(f"Results will be saved to: {results_json_path}")
    
    # 3. 리소스 준비 (Device, Model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SD3 Pipeline: {model_name} on {device}...")
    
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=False)
    except Exception as e:
        print(f"[Fatal Error] Model load failed: {e}")
        return

    # 이미지 로드 (있을 경우)
    avoid_img  = _load_image_or_none(avoid_image_path)
    target_img = _load_image_or_none(target_image_path)

    # 4. 프롬프트 로드 (COCO 등) - args.num_rows 적용
    # (이 함수는 사용자 코드에 있던 것을 그대로 사용한다고 가정)
    # 실제로는 args.start_row, args.num_rows를 인자로 넘겨야 함
    print(f"Loading prompts (Start: {args.start_row}, Count: {args.num_rows})...")
    
    # ※ 주의: load_prompts_from_coco_hf 함수가 정의되어 있어야 함
    prompt_list = load_prompts_from_coco_hf(
        start_row=args.start_row, 
        max_rows=args.num_rows,
        split=cfg.get("coco_split", "val"),
        year=cfg.get("coco_year", "2017"),
        pick=cfg.get("coco_pick", "first"),
        seed=cfg.get("seed", 0)
    )

    

    all_results = []
    grand_total_flops = 0

    # 6. FLOPs 사전 측정 (Pre-measure)
    print("Estimating FLOPs for SD3 components...")
    clip_model_for_prof, _ = _load_clip(clip_model_id, device, torch.float16)
    
    # (A) No Callback FLOPs
    comp_macs_no_callback = estimate_pipeline_flops_once_sd3(
        pipe=pipe, clip_model=None, height=height, width=width, steps=steps, 
        guidance_scale=guidance_scale, per_step_decode_224=False, per_step_clip_image=False
    )
    
    # (B) With Callback FLOPs (Noise Penalty 활성화 시)
    comp_macs_with_callback = estimate_pipeline_flops_once_sd3(
        pipe=pipe, clip_model=clip_model_for_prof, height=height, width=width, steps=steps, 
        guidance_scale=guidance_scale, per_step_decode_224=True, per_step_clip_image=True
    )
    
    if args.print_flops:
        print(f"FLOPs (Base per img): {comp_macs_no_callback['total_flops']:,}")
        print(f"FLOPs (Penalty per img): {comp_macs_with_callback['total_flops']:,}")

    # ===========================================================
    # 7. 메인 루프: Prompt -> Iteration
    # ===========================================================
    for p_idx, prompt_text in enumerate(tqdm(prompt_list, desc="Prompts")):
        # 원래 데이터셋의 인덱스를 추적하기 위해 start_row 더함
        global_idx = args.start_row + p_idx
        
        per_prompt_results = []
        avoid_images_all = []   # 이번 프롬프트에서 생성된 이미지들을 회피 대상으로 누적
        prev_step_traces = []   # Latent Penalty용 궤적 누적
        seen_hashes = set()
        early_stopped_flag = False
        
        # --- Iteration Loop ---
        for it in trange(num_iteration, desc=f"Iter (P{global_idx})", leave=False):
            t0 = time.time()
            
            # 시드 설정
            if randomize_seed:
                seed_val = random.randint(0, 2**32 - 1)
            else:
                seed_val = seed0 + global_idx * 1000 + it
            
            # 이미지 생성 (핵심 함수 호출)
            img_pil, _, _, record = generate_and_analyze_latents_per_step_Diffusion(
                pipe=pipe,
                prompt=prompt_text,
                output_filepath="", # 파일 저장은 아래에서 직접 함
                sim_ver="max", rep_ver="kl", # 호환용 더미
                steps=steps,
                L0=L0, 
                prev_step_traces=prev_step_traces,
                use_noise_penalty=use_noise_penalty,
                use_latent_penalty=use_latent_penalty,
                noise_max=noise_max,
                latent_max=latent_max,
                sharpness=sharpness,
                avoid_image=avoid_images_all, # 누적된 이미지들 회피
                target_image=target_img,
                guidance_scale=guidance_scale,
                height=height, width=width,
                seed=seed_val,
                device=device,
                clip_model_id=clip_model_id,
                mode=mode,
                apply_stride=apply_stride
            )
            
            elapsed = time.time() - t0
            
            # 다음 스텝을 위해 회피 대상에 추가
            avoid_images_all.append(img_pil)
            prev_step_traces.append(record)

            # --- FLOPs 계산 ---
            # Noise Penalty가 켜져 있고 2번째 반복부터(it>0)는 VAE/CLIP 연산이 들어간 경로 사용
            # (첫 반복은 회피할 대상이 없으므로 보통 base path를 타거나 noise penalty가 0)
            # 엄밀히는 코드 내부 구현에 따라 다르나, 여기서는 보수적으로 계산
            if it == 0 or (not use_noise_penalty):
                flops_estimate = comp_macs_no_callback["total_flops"]
            else:
                flops_estimate = comp_macs_with_callback["total_flops"]
            
            # 순수 벡터 연산 비용 추가
            flops_estimate += _to_flops_from_macs(record.get('penalty_macs_total', 0))
            grand_total_flops += flops_estimate

            # --- 파일 저장 ---
            img_path = f"{save_base}_p{p_idx:04d}_i{it:03d}.png"
            try:
                img_pil.save(img_path)
            except Exception as e:
                print(f"[Warn] Save failed: {e}")

            # --- Early Stop 체크 ---
            if args.early_stop and (it + 1) <= args.es_cutoff:
                h = _sha256_pil(img_pil)
                if h in seen_hashes:
                    print(f"[Early-Stop] Identical image at iter {it}")
                    early_stopped_flag = True
                    # 현재 결과 저장하고 break
                    per_prompt_results.append({
                        "iter": it, "image_path": img_path, "flops": flops_estimate, 
                        "time": elapsed, "early_stopped": True
                    })
                    break
                seen_hashes.add(h)
            
            per_prompt_results.append({
                "iter": it, "image_path": img_path, "flops": flops_estimate, "time": elapsed
            })

        # --- End of Iteration Loop (Diversity 측정) ---
        print(f"   -> Calculating diversity for prompt {global_idx}...")
        img_paths_this_prompt = [r["image_path"] for r in per_prompt_results]
        
        # Diversity 계산 (외부 함수 evaluate_diversity 호출)
        # 에러가 나도 죽지 않도록 try-except 처리
        diversity_metrics = {}
        try:
            diversity_metrics = evaluate_diversity(
                img_paths_this_prompt,
                clip_model_id=clip_model_id,
                device=device,
                use_llm=use_llm_diversity,
                openai_api_key=cfg.get("openai_api_key"),
                llm_model=cfg.get("llm_diversity_model", "gpt-4o-mini"),
                llm_prompt_text=prompt_text,
                llm_max_images=cfg.get("llm_max_images", 8),
            )
        except Exception as e:
            print(f"[Error] Diversity check failed: {e}")

        # 결과 누적
        all_results.append({
            "prompt_index": global_idx,
            "prompt": prompt_text,
            "results": per_prompt_results,
            "diversity": diversity_metrics
        })
        
        # 중간 저장
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "results": all_results}, f, indent=2, ensure_ascii=False)

        if early_stopped_flag:
            print(f"   -> Skipped remaining iterations for prompt {global_idx}")

    # ===========================================================
    # 8. 최종 집계 및 저장
    # ===========================================================
    print("\nCalculating aggregate metrics...")
    try:
        diversity_avg = aggregate_diversity_over_prompts(all_results)
        all_results.append({
            "prompt_index": "ALL",
            "prompt": "__AGGREGATE__",
            "results": [],
            "diversity": diversity_avg,
            "flops_total": grand_total_flops
        })
    except Exception as e:
        print(f"[Warn] Aggregate metrics failed: {e}")

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All done. Results saved to: {results_json_path}")
    print(f"≈ TOTAL FLOPs: {grand_total_flops:,}")


# 유틸 함수들 (Main 실행을 위해 필요한 최소한의 정의, import 안 된 경우 대비)
def _sha256_pil(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()

def _load_image_or_none(p: Optional[str]) -> Optional[Image.Image]:
    if not p or not os.path.exists(p):
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

if __name__ == "__main__":
    main()