import hashlib
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import json
import faiss  # For k-NN graph
import argparse
import time
import math
from tqdm import tqdm, trange
from thop.vision import basic_hooks as _th_basic
import json, math, time
from contextlib import contextmanager

from typing import Sequence, Union, Optional, List, Dict, Any, Tuple
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
from utils import *
from utils import _load_clip, _clip_img_emb, _clip_txt_emb, _norm_last_dim, _clip_image_emb_batch

# ========================= NEW: FLOPs profiling helpers =========================
# We use THOP to estimate MACs and convert to FLOPs (= 2 * MACs).
try:
    from thop import profile, clever_format
    _THOP_AVAILABLE = True
except Exception:
    _THOP_AVAILABLE = False

def _to_flops_from_macs(macs: int) -> int:
    return int(2 * macs)

# 콜백 내부 벡터연산 MACs 가중치 (근사)
LAT_GRAD_COST_PER_ELEM = 4   # latent penalty gradient 조합에서 element-wise MACs 상수(경험적)
NORM_COST_PER_ELEM     = 2   # norm(x)=sqrt(sum(x^2))의 sum of squares(+add) 근사 MACs/elem
DOT_COST_PER_ELEM      = 1   # dot = sum(x*y) 의 MACs/elem

def _safe_add_ops(m, val):
    # m.total_ops는 thop가 초기화해줌(해당 모듈의 device로)
    if isinstance(val, torch.Tensor):
        if val.device != m.total_ops.device:
            val = val.to(m.total_ops.device)
        m.total_ops += val
    else:
        m.total_ops += torch.tensor(val, dtype=m.total_ops.dtype, device=m.total_ops.device)

def _count_linear_cuda(m, x, y):
    # x[0]: (B, in_features)
    in_feats = m.in_features
    out_feats = m.out_features
    batch = x[0].shape[0] if x and hasattr(x[0], "shape") else 1
    # MACs ~= B * (in*out); thop 기본은 여기에 bias 등을 더 고려
    macs = batch * in_feats * out_feats
    _safe_add_ops(m, macs)

def _count_convNd_cuda(m, x, y):
    # x[0]: (B, Cin, H, W)
    x = x[0]
    batch = x.shape[0]
    Cin   = m.in_channels
    Cout  = m.out_channels
    k_ops = 1
    for k in m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,):
        k_ops *= k
    # 출력 spatial 크기
    out = y
    if isinstance(out, (list, tuple)):
        out = out[0]
    Hout = out.shape[-2]
    Wout = out.shape[-1]
    macs = batch * Cout * Hout * Wout * (Cin // m.groups) * k_ops
    _safe_add_ops(m, macs)

# 필요한 애들만 덮어써도 충분합니다(Conv/Linear가 지배적)
_th_basic.count_linear = _count_linear_cuda
_th_basic.count_convNd = _count_convNd_cuda

def _to_floats(macs: int) -> int:
    # By convention: 1 MAC ~ 2 FLOPs
    return int(2 * macs)
# 헬퍼: 모듈의 device, dtype 얻기
def _module_device_dtype(m: torch.nn.Module):
    p = next(m.parameters())
    return p.device, p.dtype

@torch.no_grad()
def profile_macs_text_encoder(text_encoder, tokenizer_max_len: int = 77, batch: int = 1):
    if not _THOP_AVAILABLE or text_encoder is None:
        return 0
    # CLIPTextModel은 input_ids(long)만 맞추면 dtype 충돌 없음
    dev, _ = _module_device_dtype(text_encoder)
    input_ids = torch.ones(batch, tokenizer_max_len, dtype=torch.long, device=dev)
    macs, _ = profile(text_encoder, inputs=(input_ids,), verbose=False)
    return int(macs)

@torch.no_grad()
def profile_macs_unet(unet, height: int, width: int, batch: int = 1, cond_dim: int = 768):
    if not _THOP_AVAILABLE or unet is None:
        return 0
    dev, dtype = _module_device_dtype(unet)  # ← fp16이면 half로 맞춤
    H8, W8 = height // 8, width // 8
    sample   = torch.randn(batch, 4, H8, W8, device=dev, dtype=dtype)
    # timestep은 내부에서 float로 쓰이므로 모델 dtype에 맞춤(half/float 모두 OK)
    timestep = torch.tensor([999], device=dev, dtype=dtype)
    enc      = torch.randn(batch, 77, cond_dim, device=dev, dtype=dtype)
    macs, _ = profile(unet, inputs=(sample, timestep, enc), verbose=False)
    return int(macs)

@torch.no_grad()
def profile_macs_vae_decode(vae, height: int, width: int, batch: int = 1):
    if not _THOP_AVAILABLE or vae is None:
        return 0
    dev, dtype = _module_device_dtype(vae)
    H8, W8 = height // 8, width // 8
    latents = torch.randn(batch, 4, H8, W8, device=dev, dtype=dtype)
    class _Wrap(torch.nn.Module):
        def __init__(self, vae): super().__init__(); self.vae = vae
        def forward(self, z): return self.vae.decode(z).sample
    wrapper = _Wrap(vae).to(dev).to(dtype)
    macs, _ = profile(wrapper, inputs=(latents,), verbose=False)
    return int(macs)

@torch.no_grad()
def profile_macs_clip_image(clip_model: CLIPModel, batch: int = 1):
    if not _THOP_AVAILABLE or clip_model is None:
        return 0
    dev, dtype = _module_device_dtype(clip_model)
    x = torch.randn(batch, 3, 224, 224, device=dev, dtype=dtype)  # ← fp16이면 half로
    class _Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.get_image_features(pixel_values=x)
    wrapper = _Wrap(clip_model).to(dev).to(dtype)
    macs, _ = profile(wrapper, inputs=(x,), verbose=False)
    return int(macs)
# 파일 상단 어딘가에 상수 추가
BACKWARD_FACTOR = 2.0  # 대략 conv/linear backward ≈ 2×forward 라는 실무 경험칙

def estimate_pipeline_flops_once(
    pipe: "StableDiffusionPipeline",
    clip_model: Optional[CLIPModel],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float = 7.5,
    per_step_decode_224: bool = False,
    per_step_clip_image: bool = False,
) -> Dict[str, int]:
    if not _THOP_AVAILABLE:
        return {"total_flops": 0, "text_macs": 0, "unet_macs_step": 0,
                "vae_macs_step_224": 0, "vae_macs_final": 0, "clip_img_macs_step": 0}

    text_enc = getattr(pipe, "text_encoder", None)
    unet = getattr(pipe, "unet", None)
    vae = getattr(pipe, "vae", None)

    text_macs = profile_macs_text_encoder(text_enc, tokenizer_max_len=77, batch=1)

    cfg_batch = 2 if (guidance_scale and guidance_scale > 1.0) else 1
    unet_macs_step = profile_macs_unet(unet, height=height, width=width, batch=cfg_batch)

    vae_macs_final = profile_macs_vae_decode(vae, height=height, width=width, batch=1)

    vae_macs_step_224 = profile_macs_vae_decode(vae, height=height, width=width, batch=1) if per_step_decode_224 else 0
    clip_img_macs_step = profile_macs_clip_image(clip_model, batch=1) if per_step_clip_image else 0

    # per-step VAE/CLIP는 backward까지 포함 (forward + backward_factor * forward)
    vae_macs_step_total  = vae_macs_step_224 * (1.0 + BACKWARD_FACTOR)
    clip_macs_step_total = clip_img_macs_step * (1.0 + BACKWARD_FACTOR)

    total_macs = text_macs \
        + steps * unet_macs_step \
        + vae_macs_final \
        + steps * vae_macs_step_total \
        + steps * clip_macs_step_total

    return {
        "total_flops": int(2 * total_macs),   # FLOPs = 2 × MACs
        "text_macs": int(text_macs),
        "unet_macs_step": int(unet_macs_step),
        "vae_macs_step_224": int(vae_macs_step_224),
        "vae_macs_final": int(vae_macs_final),
        "clip_img_macs_step": int(clip_img_macs_step),
    }
# ======================= END FLOPs profiling helpers ============================

# ===========================================================
# 메인: Diffusion용 "AR-호출감" 함수
# ===========================================================
# 벡터 연산 비용 근사용 상수(원하면 조정 가능)
LAT_GRAD_COST_PER_ELEM = 4   # latent penalty gradient 조합에서 element-wise MACs 상수
NORM_COST_PER_ELEM     = 2   # ||x|| 계산 근사 MACs/elem
DOT_COST_PER_ELEM      = 1   # dot(x,y) MACs/elem

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _to_flops_from_macs(macs: int) -> int:
    return int(2 * macs)

def generate_and_analyze_latents_per_step_Diffusion(
    pipe: "StableDiffusionPipeline",
    prompt: str,
    output_filepath: str,
    sim_ver: str, rep_ver: str,  # 자리만 유지(호환용)
    previous_responses: List[Any] = [],
    prev_probs: Any = None, prev_embs: Any = None,
    fill_order: List[int] = [],
    gen_length: int = 512, steps: int = 50,
    knn_k: int = 5,
    threshold: float = 0.1,
    temperature: float = 0.0,   # 자리만 유지
    k_gamma: float = 0.5,
    L0: int = 20,
    constant: float = 1.0,
    prev_step_traces: Optional[List[Dict[str,Any]]] = None,
    topk_k: int = 64,

    # 하이브리드(Noise/Latent) 페널티 스위치/세기
    use_noise_penalty: bool = True,
    use_latent_penalty: bool = True,
    noise_max: float = 1.0,     # (logit 대응)
    latent_max: float = 0.25,   # (hidden 대응; 보통 더 작게)
    sharpness: float = 0.1,

    # 제어 신호
    avoid_image: Optional[Union[Image.Image, Sequence[Image.Image]]] = None,  # 배열 허용 (이전 run에서 나온 이미지들)
    target_image: Optional[Image.Image] = None,

    # 샘플링 설정
    guidance_scale: float = 7.5,
    height: int = 512, width: int = 512,
    seed: int = 1234,
    device: str = "cuda",
    clip_model_id: str = "openai/clip-vit-base-patch32",

    # === 새로 추가된 옵션 ===
    apply_stride: int = 1,        # k step마다 penalty 적용 (1이면 매 step),
    mode='logistic'
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, Optional[Dict[str,Any]]]:
    """
    Stable Diffusion용: 매 스텝에서 noise/latent penalty 기반 바이어스를 '직접' 적용.
    - avoid_image: 여러 장 지원, CLIP 유사도 max를 회피. (이번 iteration 시작 시 1회만 임베딩 캐시)
    - prev_step_traces: 과거 run의 step별 latent 중 max cosine ref를 골라 회피.
    - 이번 run의 step별 latent를 record['latents_steps']로 저장(다음 run에서 prev_step_traces로 사용).
    - apply_stride: penalty를 매 step이 아니라 k step마다만 적용해 연산량 절감.

    Returns:
      - img_pil:        최종 생성 이미지
      - last_noise_pred:자리값(커스텀 루프가 아니라면 0 텐서 유지)
      - last_latent:    자리값(콜백 내부 last_latent를 간단화)
      - record:         per-step 기록 + penalty_macs_total 포함
    """
    torch.manual_seed(seed)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # CLIP 준비 (utils에 있는 헬퍼 사용 가정)
    clip_model, clip_proc = _load_clip(clip_model_id, device=device, dtype=torch.float16)

    # ====== avoid 이미지 여러 장 지원: 배치 임베딩 1회 캐시 ======
    avoid_imgs = _ensure_list(avoid_image)
    avoid_emb_cache: Optional[torch.Tensor] = None  # [N,D], fp32-normalized, device상주

    def _ensure_avoid_emb_cache():
        nonlocal avoid_emb_cache
        if avoid_emb_cache is not None:
            return
        if len(avoid_imgs) == 0:
            avoid_emb_cache = None
            return
        with torch.no_grad():
            # utils에 있는 배치 임베딩 (이미 정규화된 fp32라 가정; 아니면 아래에서 normalize)
            embs = _clip_image_emb_batch(clip_model, clip_proc, avoid_imgs, device=device)  # [N,D]
            # 안전하게 정규화/형변환
            embs = torch.nn.functional.normalize(embs.to(torch.float32).to(device), dim=-1)
        avoid_emb_cache = embs  # 이후 step마다 재계산 금지

    # target latent (옵션)
    z_target = pil_to_vae_latent(pipe, target_image, height, width, device) if target_image is not None else None

    # 기록용
    clip_sim_avoid_steps: List[float]    = []
    clip_sim_target_steps: List[float]   = []
    latent_cos_avoid_steps: List[float]  = []
    latent_cos_target_steps: List[float] = []
    eps_norm_steps: List[float]          = []
    step_indices: List[int]              = []
    latents_steps: List[torch.Tensor]    = []   # 이번 run의 step별 latent 저장(CPU로 저장)

    if prev_step_traces is None:
        prev_step_traces = []

    last_noise_pred = torch.zeros(1)  # 자리
    last_latent     = torch.zeros(1)  # 자리

    # 콜백 벡터 연산 MACs 누적(근사) — 나중에 record['penalty_macs_total']로 반환
    penalty_macs_accum = 0

    def make_callback():
        def _callback(pipe, step: int, timestep: int, callback_kwargs: dict):
            nonlocal last_noise_pred, last_latent, penalty_macs_accum, avoid_emb_cache

            # apply_stride: k step마다만 penalty 적용
            apply_now = (apply_stride <= 1) or (step % apply_stride == 0)

            # avoid 임베딩 캐시 1회 보장
            _ensure_avoid_emb_cache()
            avoid_img_embs = avoid_emb_cache  # 이후 매 step 재계산 금지

            # target 이미지 임베딩 (단건; 필요 시 1회 계산)
            target_img_emb = _clip_img_emb(clip_model, clip_proc, target_image, device) if target_image is not None else None
            if target_img_emb is not None:
                target_img_emb = torch.nn.functional.normalize(target_img_emb.to(torch.float32), dim=-1).unsqueeze(0)  # [1,D]

            # 스케줄 가중
            if use_noise_penalty and use_latent_penalty:
                w_noise, w_latent = schedule_weights(step, L0=L0, rep_max=noise_max,
                                                     hidden_max=latent_max, sharpness=sharpness,mode=mode)
            elif use_noise_penalty:
                w_noise, w_latent = noise_max, 0.0
            elif use_latent_penalty:
                w_noise, w_latent = 0.0, latent_max
            else:
                w_noise, w_latent = 0.0, 0.0

            # 현재 latents
            latents = callback_kwargs["latents"]   # [B,4,H/8,W/8]
            B = latents.shape[0]
            last_latent = latents.detach().clone()

            # 크기/길이
            H8, W8 = latents.shape[-2], latents.shape[-1]
            M = 4 * H8 * W8  # per-sample latent vector length

            # ---------- (A) Latent penalty: prev_step_traces에서 같은 step의 ref들 중 "max cosine" 회피 ----------
            if apply_now and (w_latent > 0.0) and use_latent_penalty and (len(prev_step_traces) > 0):
                # 과거 ref 수집
                z_refs = []
                for tr in prev_step_traces:
                    if "latents_steps" in tr and step < len(tr["latents_steps"]):
                        zref = tr["latents_steps"][step]
                        if isinstance(zref, torch.Tensor):
                            zr = zref
                        else:
                            zr = torch.tensor(zref)
                        if zr.dim() == 3:   # [4,H/8,W/8]
                            zr = zr.unsqueeze(0)  # [1,4,H/8,W/8]
                        z_refs.append(zr.to(latents.device, dtype=latents.dtype))

                if len(z_refs) > 0:
                    Z = torch.stack(z_refs, dim=0).to(torch.float32)   # [R,B,4,H/8,W/8]
                    z_t = latents.to(torch.float32)

                    x_flat = z_t.view(B, -1)                                   # [B,M]
                    Xn    = x_flat.norm(dim=1, keepdim=True).clamp_min(1e-6)   # [B,1]

                    R = Z.shape[0]
                    cos_mat = []
                    for r in range(R):
                        y      = Z[r]                                          # [B,4,H/8,W/8]
                        y_flat = y.view(B, -1)                                 # [B,M]
                        Yn     = y_flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        cos_r  = (x_flat * y_flat).sum(dim=1, keepdim=True) / (Xn * Yn)  # [B,1]
                        cos_mat.append(cos_r)
                    cos_mat = torch.cat(cos_mat, dim=1)                        # [B,R]

                    idx     = torch.argmax(cos_mat, dim=1)                     # [B]
                    max_cos = cos_mat.gather(1, idx.unsqueeze(1)).mean()
                    latent_cos_avoid_steps.append(float(max_cos.item()))

                    # === MACs 근사 누적 ===
                    macs_norm_x = B * NORM_COST_PER_ELEM * M             # X norm
                    macs_norm_y = B * R * NORM_COST_PER_ELEM * M         # Y norms
                    macs_dot    = B * R * DOT_COST_PER_ELEM * M          # dot(X,Y_r)
                    macs_grad   = B * LAT_GRAD_COST_PER_ELEM * M         # 선택 r의 gradient 조합
                    penalty_macs_accum += (macs_norm_x + macs_norm_y + macs_dot + macs_grad)

                    # === gradient 계산 및 업데이트 ===
                    grad_flat = torch.zeros_like(x_flat)                        # [B,M]
                    for b in range(B):
                        r      = idx[b].item()
                        y      = Z[r, b]                                       # [4,H/8,W/8]
                        y_flat = y.view(-1)                                    # [M]
                        xf     = x_flat[b]                                     # [M]
                        xn     = Xn[b]                                         # [1]
                        yn     = y_flat.norm().clamp_min(1e-6)
                        cos    = (xf @ y_flat) / (xn * yn)

                        term1  = y_flat / (xn * yn)
                        term2  = (cos / (xn**2)) * xf
                        g      = (term1 - term2)                               # [M] = d cos / d x
                        grad_flat[b] = g

                    grad = grad_flat.view_as(z_t)                               # [B,4,H/8,W/8]
                    grad = _norm_last_dim(grad)
                    latents = (z_t - w_latent * grad).to(callback_kwargs["latents"].dtype).detach()

            # --- (B) Noise penalty via CLIP: avoid 집합 max 유사도 ---
            if apply_now and (w_noise > 0.0) and use_noise_penalty and (len(prev_step_traces) > 0):
                with torch.enable_grad():
                    z32 = latents.detach().to(torch.float32).requires_grad_(True)

                    imgs_224 = vae_decode_224(pipe, z32)  # grad 경로 유지 (네트워크 FLOPs는 외부에서 반영)
                    img_emb = clip_model.get_image_features(pixel_values=imgs_224.to(torch.float16))
                    img_emb = torch.nn.functional.normalize(img_emb.to(torch.float32), dim=-1)  # [B,D]

                    # === MACs 근사 누적 (네트워크 외 벡터연산 부분만) ===
                    D = int(img_emb.shape[-1])
                    if avoid_img_embs is not None:
                        N_avoid = int(avoid_img_embs.shape[0])
                        # einsum('bd,nd->bn'): ~ B * N * D MACs
                        penalty_macs_accum += B * N_avoid * D
                        # normalize(img_emb): ~ B * D
                        penalty_macs_accum += B * D
                    if target_img_emb is not None:
                        # dot mean: ~ B * D
                        penalty_macs_accum += B * D

                    loss = torch.tensor(0.0, device=img_emb.device, dtype=img_emb.dtype)

                    if avoid_img_embs is not None and avoid_img_embs.shape[0] > 0:
                        sims_img = torch.einsum('bd,nd->bn', img_emb, avoid_img_embs)  # [B,N]
                        max_sim_img, _ = sims_img.max(dim=1)                           # [B]
                        loss = loss + max_sim_img.mean()
                        clip_sim_avoid_steps.append(float(max_sim_img.mean().item()))

                    if target_img_emb is not None:
                        tgt_sim = torch.einsum('bd,nd->bn', img_emb, target_img_emb).mean()  # scalar
                        loss = loss - tgt_sim
                        clip_sim_target_steps.append(float(tgt_sim.item()))

                    grad_z, = torch.autograd.grad(loss, z32, retain_graph=False)
                grad_z = _norm_last_dim(grad_z)
                latents = (z32 - w_noise * grad_z).detach().to(callback_kwargs["latents"].dtype)

            # 기록/메타
            step_indices.append(int(step))
            eps_norm_steps.append(float(latents.norm().item()))
            latents_steps.append(latents.detach().cpu())

            # 변경 latents 반영
            callback_kwargs["latents"] = latents
            return callback_kwargs

        return _callback

    step_cb = make_callback()

    # ====== 파이프라인 실행 ======
    out = pipe(
        prompt,
        num_inference_steps=min(
            steps,
            pipe.scheduler.config.num_train_timesteps
            if hasattr(pipe.scheduler.config, 'num_train_timesteps') else steps
        ),
        guidance_scale=guidance_scale,
        height=height, width=width,
        generator=torch.Generator(device).manual_seed(seed),
        callback_on_step_end=step_cb,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    img_pil = out.images[0]

    # prev_step_traces 호환 레코드(이번 run 결과)
    record = {
        'probs_topk_idx_steps': None,
        'probs_topk_val_steps': None,
        'seq_emb_steps':        None,
        'clip_sim_avoid_steps':    clip_sim_avoid_steps,
        'clip_sim_target_steps':   clip_sim_target_steps,
        'latent_cos_avoid_steps':  latent_cos_avoid_steps,
        'latent_cos_target_steps': latent_cos_target_steps,
        'eps_norm_steps':          eps_norm_steps,
        'step_indices':            step_indices,
        'fill_order':              None,
        'latents_steps':           latents_steps,         # ★ 다음 run에서 prev_step_traces로 사용
        'penalty_macs_total':      int(penalty_macs_accum),  # ★ 콜백 벡터연산 MACs 누적(근사)
        'apply_stride':            int(apply_stride),     # 참고용
    }

    # 선택 저장
    try:
        if output_filepath:
            img_pil.save(output_filepath)
    except Exception:
        pass

    return img_pil, last_noise_pred, last_latent, record


from eval_image import evaluate_diversity


from datasets import load_dataset

# ===== (필요 전역 유틸이 이미 있다면 그대로 사용/삭제) =====
def _sha256_pil(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()

def _load_image_or_none(p: Optional[str]) -> Optional[Image.Image]:
    if not p or not os.path.exists(p):
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

# ===== COCO(HF) 프롬프트 로더 =====
def load_prompts_from_coco_hf(
    start_row: int = 0,
    max_rows: int = 20,
    split: str = "validation",   # "train" | "validation"  (val2017=validation)
    year: str = "2017",
    pick: str = "first",         # "first" | "random" | "all"
    dedupe: bool = True,
    seed: int = 0,
) -> List[str]:
    import random
    random.seed(seed)
    ds = load_dataset("lmms-lab/COCO-Caption2017",split=split)  # cols: image, id, captions=[{id,text}]
    N = len(ds)
    lo = max(0, start_row)
    hi = min(N, start_row + max_rows)
    prompts: List[str] = []
    for i in range(lo, hi):
        caps = ds[i]["captions"]
        texts = [c.get("text", "").strip() for c in caps if c.get("text", "").strip()]
        if not texts:
            continue
        if pick == "first":
            prompts.append(texts[0])
        elif pick == "random":
            prompts.append(random.choice(texts))
        elif pick == "all":
            prompts.extend(texts)
        else:
            raise ValueError(f"Unknown pick: {pick}")
    if dedupe and pick == "all":
        seen, uniq = set(), []
        for p in prompts:
            if p not in seen:
                uniq.append(p); seen.add(p)
        prompts = uniq
    return prompts

# ===== 여기에 generate_and_analyze_latents_per_step_Diffusion(...) 가 있다고 가정 =====
# from your_module import generate_and_analyze_latents_per_step_Diffusion
import random
def main():
    
    seen_hashes = set()
    parser = argparse.ArgumentParser(description="Generate images on COCO prompts via Diffusion + Hybrid Avoidance.")
    parser.add_argument("--config", type=str, default="image_test/config_image.json")
    parser.add_argument("--result_suffix", type=str, default="coco_imgs")
    parser.add_argument("--num_rows", type=int, default=20)     # ← COCO에서 가져올 프롬프트 개수
    parser.add_argument("--start_row", type=int, default=0)     # ← 시작 인덱스
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--es_cutoff", type=int, default=10)

    # NEW: optional flag to print a FLOPs breakdown per image
    parser.add_argument("--print_flops", action="store_true", help="Print component-wise FLOPs estimates.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ===== Diffusers & 세팅 =====
    model_name     = cfg.get("model_name", "runwayml/stable-diffusion-v1-5")
    steps          = cfg.get("steps", 50)
    guidance_scale = cfg.get("guidance_scale", 7.5)
    height         = cfg.get("height", 512)
    width          = cfg.get("width", 512)
    seed0          = cfg.get("seed", 42)
    num_iteration  = cfg.get("num_iteration", 15)  # 각 프롬프트당 생성 횟수

    # 하이브리드 스케줄/강도 (AR 네이밍과 매핑)
    use_noise_penalty  = cfg.get("use_rep_penalty", True)      # noise = logit 대응
    use_latent_penalty = cfg.get("use_hidden_penalty", True)   # latent = hidden 대응
    noise_max          = cfg.get("rep_max_constant", 0.2)
    latent_max         = cfg.get("hidden_max_constant", 0.25)
    L0                 = cfg.get("L0", 20)
    sharpness          = cfg.get("sharpness", 0.5)

    avoid_text        = cfg.get("avoid_text")
    target_text       = cfg.get("target_text")
    avoid_image_path  = cfg.get("avoid_image_path")
    target_image_path = cfg.get("target_image_path")
    clip_model_id     = cfg.get("clip_model_id", "openai/clip-vit-base-patch32")
    mode = cfg.get('mode','logistic')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)

    avoid_img  = _load_image_or_none(avoid_image_path)
    target_img = _load_image_or_none(target_image_path)

    # ===== COCO 프롬프트 로드 (HF) — num_rows 만큼 =====
    coco_split = cfg.get("coco_split", "val")  # "train"|"validation"
    coco_year  = cfg.get("coco_year", "2017")
    coco_pick  = cfg.get("coco_pick", "first")        # "first"|"random"|"all"
    coco_seed  = cfg.get("seed", 0)
    prompt_list=load_prompts_from_coco_hf_generic(start_row=args.start_row, max_rows=args.num_rows,)
    

    # ===== 저장 경로 =====
    base = os.path.splitext(args.config)[0]
    cfg_dir = os.path.dirname(base)
    save_base = os.path.join(cfg_dir, args.result_suffix) if args.result_suffix else base
    print(save_base)
    results_json_path = f"{save_base}_result.json"

    all_results: List[Dict[str, Any]] = []
    early_stopped_flag = False
    seed0          = cfg.get("seed", 42)
    randomize_seed = cfg.get("randomize_seed", False)

    # ===== FLOPs pre-measure (component MACs)
    clip_model_for_prof, _clip_proc_dummy = _load_clip(clip_model_id, device=device, dtype=torch.float16)
    # For iteration 1, callback CLIP is inactive (prev_step_traces empty). From iteration>=2, it becomes active if noise penalty on.
    per_step_decode_224_active = False  # set True dynamically per-iteration
    per_step_clip_active = False

    # Base component MACs (we will reuse below). We first assume callback off:
    comp_macs_no_callback = estimate_pipeline_flops_once(
        pipe=pipe,
        clip_model=None,
        height=height, width=width,
        steps=steps, guidance_scale=guidance_scale,
        per_step_decode_224=False,
        per_step_clip_image=False,
    )

    # With callback (for it >= 1 and noise penalty on):
    comp_macs_with_callback = estimate_pipeline_flops_once(
        pipe=pipe,
        clip_model=clip_model_for_prof,
        height=height, width=width,
        steps=steps, guidance_scale=guidance_scale,
        per_step_decode_224=True,
        per_step_clip_image=True,
    )

    print("\n[FLOPs ESTIMATE] per image (no-callback path):")
    print(f"  total ~ {comp_macs_no_callback['total_flops']:,} FLOPs")
    print(f"  text_macs: {comp_macs_no_callback['text_macs']:,}, "
            f"unet_macs_step: {comp_macs_no_callback['unet_macs_step']:,}, "
            f"vae_macs_final: {comp_macs_no_callback['vae_macs_final']:,}")
    print("[FLOPs ESTIMATE] per image (with-callback path, i>=2):")
    print(f"  total ~ {comp_macs_with_callback['total_flops']:,} FLOPs")
    print(f"  + per-step vae_macs_step_224: {comp_macs_with_callback['vae_macs_step_224']:,}, "
            f"+ per-step clip_img_macs_step: {comp_macs_with_callback['clip_img_macs_step']:,}")

    # ===== 2중 루프: num_rows(프롬프트 수) × num_iteration(반복 생성) =====
    grand_total_flops = 0

    for p_idx, prompt_text in enumerate(tqdm(prompt_list)):
        per_prompt_results = []
        avoid_images_all = []   # ★ 루프 바깥에서 초기화
        prev_step_traces=[]
        for it in trange(num_iteration):
            t0 = time.time()
            # ★ 시드 결정
            if randomize_seed:
                seed_val = random.randint(0, 2**32 - 1)
            else:
                seed_val = seed0 + p_idx * 1000 + it

            img_pil, last_noise_pred, last_latent, record = generate_and_analyze_latents_per_step_Diffusion(
                pipe=pipe,
                prompt=prompt_text,
                output_filepath="",
                sim_ver="max", rep_ver="kl",
                previous_responses=[], prev_probs=None, prev_embs=None,
                fill_order=[], gen_length=0, steps=steps,
                temperature=0.0, k_gamma=0.5, L0=L0, constant=1.0,
                prev_step_traces=prev_step_traces, topk_k=0,
                use_noise_penalty=use_noise_penalty,
                use_latent_penalty=use_latent_penalty,
                noise_max=noise_max,
                latent_max=latent_max,
                sharpness=sharpness,
                avoid_image=avoid_images_all,
                target_image=target_img,
                guidance_scale=guidance_scale,
                height=height, width=width,
                seed=seed_val,             # ← 여기서 시드 주입
                device=device,
                clip_model_id=clip_model_id,
            )
            elapsed = time.time() - t0
            avoid_images_all.append(img_pil)
            prev_step_traces.append(record)

            # ===== FLOPs per-iteration estimate =====
            # i == 0: callback CLIP path inactive (no prev traces) -> no-callback estimate
            # i >= 1: if noise penalty is enabled and noise_max > 0, callback active -> with-callback estimate
            
            # 기존: with-callback / no-callback total_flops (네트워크 forward/backward 포함)
            if it == 0 or (not use_noise_penalty) or (noise_max is None) or (noise_max <= 0.0):
                flops_estimate = comp_macs_no_callback["total_flops"]
            else:
                flops_estimate = comp_macs_with_callback["total_flops"]

            # ★ 추가: 콜백 벡터연산 MACs → FLOPs로 변환해 더하기
            penalty_extra_flops = _to_flops_from_macs(record.get('penalty_macs_total', 0))
            flops_estimate += penalty_extra_flops
            grand_total_flops += flops_estimate
            # 저장 파일명
            img_path = f"{save_base}_p{p_idx:04d}_i{it:03d}.png"
            try:
                img_pil.save(img_path)
            except Exception as e:
                print(f"[warn] save failed: {img_path}: {e}")

            # 조기 종료(옵션): 동일 이미지(바이트)면 중단
            if args.early_stop and (it + 1) <= args.es_cutoff:
                h = _sha256_pil(img_pil)
                if h in seen_hashes:
                    print(f"[early-stop] prompt={p_idx}, iter={it} reason=identical_image")
                    early_stopped_flag = True
                    per_prompt_results.append({
                        "iter": it, "image_path": img_path, "time": elapsed,
                        "early_stopped": True,
                        "flops": flops_estimate
                    })
                    break
                seen_hashes.add(h)

            per_prompt_results.append({
                "iter": it,
                "image_path": img_path,
                "time": elapsed,
                "flops": flops_estimate
            })
            print(f"[{p_idx:04d}/{it:03d}] saved: {img_path} ({elapsed:.2f}s)  | FLOPs≈{flops_estimate:,}")

        img_paths_this_prompt = [r["image_path"] for r in per_prompt_results]

        diversity_metrics = evaluate_diversity(
            img_paths_this_prompt,
            clip_model_id=cfg.get("clip_model_id", "openai/clip-vit-base-patch32"),
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            use_llm=cfg.get("use_llm_diversity", False),
            openai_api_key=cfg.get("openai_api_key"),
            llm_model=cfg.get("llm_diversity_model", "gpt-4o-mini"),
            llm_prompt_text=prompt_text,
            llm_max_images=cfg.get("llm_max_images", 8),
        )

        all_results.append({
            "prompt_index": p_idx,
            "prompt": prompt_text,
            "results": per_prompt_results,
            "diversity": diversity_metrics,
        })
        print(diversity_metrics)
        
        # 프롬프트 단위로 JSON 업데이트(중간 저장)
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "results": all_results}, f, indent=2, ensure_ascii=False)

        if early_stopped_flag:
            print("[early-stop] stopping remaining prompts in this trial")
            break

    # ===== Aggregate diversity + FLOPs summary =====
    diversity_avg = aggregate_diversity_over_prompts(all_results)
    all_results.append({
        "prompt_index": "ALL",
        "prompt": "__AGGREGATE__",
        "results": [],
        "diversity": diversity_avg,
        "flops_total": grand_total_flops
    })
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All done. JSON saved to {results_json_path}")
    if _THOP_AVAILABLE:
        print(f"≈ TOTAL FLOPs over run: {grand_total_flops:,}")
    else:
        print("Note: thop not installed; FLOPs recorded as 0. Try: pip install thop")

if __name__ == "__main__":
    main()