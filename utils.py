# %%
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

import json, math, time
from contextlib import contextmanager

from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
# from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
# -----------------------
# 유틸: CLIP 임베딩
# -----------------------

from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from PIL import Image
import random
import math
from statistics import mean


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))

def aggregate_diversity_over_prompts(all_results):
    flat = {}          # key -> list[float]
    llm_div_scores = []  # diversity_llm.score
    llm_qual_means = []  # quality_llm.score_mean
    num_prompts = 0
    total_images = 0
    num_images_list = []

    def push(k, v):
        if _is_number(v):
            flat.setdefault(k, []).append(float(v))

    for item in all_results:
        d = (item or {}).get("diversity")
        if not isinstance(d, dict):
            continue
        num_prompts += 1
        ni = d.get("num_images")
        if _is_number(ni):
            total_images += int(ni)
            num_images_list.append(int(ni))
            push("num_images", ni)

        # 1) 1단계 숫자
        for k in ("diversity_clip","clip_mean_cosine","clip_min_cosine",
                  "clip_max_cosine","clip_std_cosine","unique_ratio"):
            if k in d:
                push(k, d[k])

        # 2) 중첩 통계
        for nk in ("size_stats","file_bytes_stats"):
            sub = d.get(nk)
            if isinstance(sub, dict):
                for sk, sv in sub.items():
                    push(f"{nk}.{sk}", sv)

        # 3) LLM diversity score
        dd = d.get("diversity_llm")
        if isinstance(dd, dict) and _is_number(dd.get("score")):
            llm_div_scores.append(float(dd["score"]))

        # 4) LLM quality mean
        qd = d.get("quality_llm")
        if isinstance(qd, dict) and _is_number(qd.get("score_mean")):
            llm_qual_means.append(float(qd["score_mean"]))

    # 평균 계산
    avg = {k: mean(vs) for k, vs in flat.items() if vs}

    out = {
        "num_prompts": num_prompts,
        "total_images": total_images,
        "num_images_mean": mean(num_images_list) if num_images_list else None,

        "num_images": avg.get("num_images"),
        "diversity_clip": avg.get("diversity_clip"),
        "clip_mean_cosine": avg.get("clip_mean_cosine"),
        "clip_min_cosine": avg.get("clip_min_cosine"),
        "clip_max_cosine": avg.get("clip_max_cosine"),
        "clip_std_cosine": avg.get("clip_std_cosine"),
        "unique_ratio": avg.get("unique_ratio"),

        "size_stats": {
            "width_mean":  avg.get("size_stats.width_mean"),
            "height_mean": avg.get("size_stats.height_mean"),
            "width_std":   avg.get("size_stats.width_std"),
            "height_std":  avg.get("size_stats.height_std"),
        },
        "file_bytes_stats": {
            "bytes_mean": avg.get("file_bytes_stats.bytes_mean"),
            "bytes_std":  avg.get("file_bytes_stats.bytes_std"),
        },

        # ── LLM 집계 ──
        "diversity_llm": {
            "score_mean": mean(llm_div_scores) if llm_div_scores else None,
            "coverage": len(llm_div_scores),
        },
        "quality_llm": {
            "score_mean": mean(llm_qual_means) if llm_qual_means else None,
            "coverage": len(llm_qual_means),
        },
    }
    return out


def _normalize_caption_list(rec: Dict[str, Any]) -> List[str]:
    """
    HF COCO 변형 스키마를 자동 감지해서, 캡션(문자열) 리스트를 반환.
    지원:
      - {"captions": [{"text": "..."}], "image": ...}  (HF 'coco_captions')
      - {"answer": ["...", "..."], "question": "...", "image": ...}  (instruction/VQA 스타일)
      - {"caption": "..."} 단일 필드도 방어
    """
    caps: List[str] = []

    # 1) 표준 coco_captions
    if "captions" in rec and isinstance(rec["captions"], list):
        for c in rec["captions"]:
            if isinstance(c, dict):
                t = c.get("text", "")
            else:
                t = str(c)
            t = (t or "").strip()
            if t:
                caps.append(t)
        if caps:
            return caps

    # 2) instruction 변형 (스크린샷과 유사) : answer가 list[str]
    if "answer" in rec and isinstance(rec["answer"], list):
        for t in rec["answer"]:
            t = (t or "").strip()
            if t:
                caps.append(t)
        if caps:
            return caps

    # 3) 단일 caption 방어
    if "caption" in rec and isinstance(rec["caption"], str):
        t = rec["caption"].strip()
        if t:
            caps.append(t)

    return caps

def _get_image_from_record(rec: Dict[str, Any]) -> Optional[Image.Image]:
    """
    datasets가 반환하는 PIL 이미지를 안전하게 꺼내기.
    보통 rec['image']가 바로 PIL.Image.Image.
    """
    img = rec.get("image", None)
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # (드물게 다른 형태면 확장 가능)
    return None

def load_prompts_from_coco_hf_generic(
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
def _load_clip(clip_id: str = "openai/clip-vit-base-patch32", device="cuda", dtype=torch.float16):
    model = CLIPModel.from_pretrained(clip_id, use_safetensors=True).to(device, dtype=dtype).eval()
    proc  = CLIPProcessor.from_pretrained(clip_id)
    return model, proc
# ==== Fallback: batch CLIP image embedding ====
# utils.py에 _clip_image_emb_batch가 없을 때 이 구현을 사용하세요.
# 반환: [N, D] fp32 normalized tensor (device 상주)
def _clip_image_emb_batch(clip_model, clip_processor, pil_images, device="cuda", batch_size=8):
    """
    pil_images: List[PIL.Image.Image] (None 포함 가능)
    clip_model: transformers.CLIPModel (fp16/float 혼용 가능)
    clip_processor: transformers.CLIPProcessor
    """
    # 필터링
    imgs = [im for im in pil_images if im is not None]
    if len(imgs) == 0:
        return None

    # 모델 dtype/디바이스 맞추기
    try:
        p = next(clip_model.parameters())
        model_device = p.device
        model_dtype  = p.dtype
    except StopIteration:
        model_device = torch.device(device)
        model_dtype  = torch.float16

    feats = []
    clip_model.eval()
    with torch.no_grad():
        for s in range(0, len(imgs), batch_size):
            batch = imgs[s:s+batch_size]
            proc = clip_processor(images=batch, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(model_device)
            # fp16 모델이면 half로, 아니면 float로
            if model_dtype == torch.float16:
                pixel_values = pixel_values.half()
            else:
                pixel_values = pixel_values.float()

            # get_image_features: [B, D]
            feats_b = clip_model.get_image_features(pixel_values=pixel_values)
            # 안전 캐스팅 + 정규화(fp32)
            feats_b = torch.nn.functional.normalize(feats_b.to(torch.float32), dim=-1)
            feats.append(feats_b)

    embs = torch.cat(feats, dim=0)  # [N, D]
    return embs
# @torch.no_grad()
def _clip_img_emb(clip_model, clip_proc, pil: Image.Image, device="cuda", dtype=torch.float16):
    inp = clip_proc(text=None, images=[pil], return_tensors="pt").to(device)
    emb = clip_model.get_image_features(**inp)  # [1,D], fp16 OK
    emb = F.normalize(emb.to(torch.float32), dim=-1)  # 안정성 위해 fp32 normalize
    return emb  # [1,D], fp32

# @torch.no_grad()
def _clip_txt_emb(clip_model, clip_proc, text: str, device="cuda"):
    inp = clip_proc(text=[text], images=None, return_tensors="pt", padding=True).to(device)
    emb = clip_model.get_text_features(**inp)  # [1,D]
    emb = F.normalize(emb.to(torch.float32), dim=-1)
    return emb

# -----------------------
# 유틸: VAE latent <-> image
# -----------------------
# @torch.no_grad()
def vae_decode_224(pipe, latents_fp32: torch.Tensor) -> torch.Tensor:
    """
    latents_fp32: [B,4,H/8,W/8], 일반적으로 fp32로 들어오지만
    decode 직전 VAE의 device/dtype로 맞춘다.
    반환은 CLIP 임베딩 계산을 위해 fp32 [B,3,224,224]
    """
    vae = pipe.vae
    vae_device = next(vae.parameters()).device
    vae_dtype  = vae.dtype
    scaling = getattr(getattr(vae, "config", object()), "scaling_factor", 0.18215)

    # 1) VAE 기대 dtype/device로 캐스팅
    x = (latents_fp32.to(device=vae_device, dtype=vae_dtype)) / scaling  # ex) fp16로 맞춤

    # 2) 디코드: 출력은 보통 vae_dtype (fp16/bf16 등)
    imgs = vae.decode(x).sample  # [-1,1]

    # 3) [0,1]로 정규화 후, CLIP 호환을 위해 fp32로 승격
    imgs = (imgs.clamp(-1, 1) + 1) / 2             # [0,1], vae_dtype
    imgs = imgs.to(torch.float32)                   # CLIP 입력 안정성 위해 fp32

    # 4) 224로 리사이즈
    imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
    return imgs  # [B,3,224,224], fp32

# @torch.no_grad()
def pil_to_vae_latent(pipe, pil: Image.Image, height=512, width=512, device="cuda"):
    img = pil.convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                          .view(height, width, 3)
                          .float()
                         ).numpy()).to(torch.float32)  # 안전 변환(선호 로더로 바꿔도 됨)
    t = t.permute(2,0,1)[None] / 255.0            # [1,3,H,W]
    t = t * 2 - 1                                  # [-1,1]
    t = t.to(device)
    posterior = pipe.vae.encode(t).latent_dist
    z = posterior.mean * 0.18215                   # SD v1.5 규약
    return z.half()        
def _append_mean_every_15(path: str, done_event: str, mean_event: str):
    """
    JSONL(path)에서 done_event 라인의 flops_total을 모아,
    개수가 15의 배수일 때 '최근 15개'의 평균을 mean_event로 추가 기록.
    """
    try:
        cnt = 0
        vals = []
        # 한 번 훑어서 개수와 값을 모으되, 메모리 아껴서 필요시 tail-15만 유지
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") == done_event:
                    cnt += 1
                    vals.append(rec.get("flops_total", 0))
                    if len(vals) > 15:
                        vals.pop(0)  # 최근 15개만 유지
        if cnt > 0 and (cnt % 15 == 0) and len(vals) == 15:
            mean_val = sum(vals) / 15.0
            summary = {
                "event": mean_event,          # 예: "AR_mean_15"
                "flops_mean": mean_val,
                "runs_window": 15,
                "done_event_count": cnt,      # 지금까지 누적된 done 개수
                "time": time.time()
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary) + "\n")
    except FileNotFoundError:
        pass
# ---------- FLOPs utilities ----------
# ==== FLOPS global scope helper ====
_FLOPS_GLOBAL = None

from contextlib import contextmanager

@contextmanager
def flops_scope(counter):
    """자식 함수들이 같은 카운터를 공유하도록 하는 컨텍스트."""
    global _FLOPS_GLOBAL
    prev = _FLOPS_GLOBAL
    _FLOPS_GLOBAL = counter
    try:
        yield
    finally:
        _FLOPS_GLOBAL = prev

def _flops_safe():
    """현재 활성화된 글로벌 FLOPs 카운터를 반환 (없으면 None)."""
    global _FLOPS_GLOBAL
    return _FLOPS_GLOBAL

class FlopsCounter:
    def __init__(self):
        self.total = 0
        self.timeline = []  # optional: [(tag, flops, extra)]
    def add(self, n, tag=None, extra=None):
        if n is None: return
        n = int(n)
        self.total += n
        if tag:
            self.timeline.append((tag, n, extra or {}))
    # --- elementwise ops ---
    def elemwise(self, *sizes, tag=None):
        n = 1
        for s in sizes: n *= int(s)
        self.add(n, tag or "elemwise")
        return n
    # --- reductions (sum/max/softmax pre-exp etc.): ~N ---
    def reduce(self, *sizes, tag=None):
        n = 1
        for s in sizes: n *= int(s)
        self.add(n, tag or "reduce")
        return n
    # --- matrix multiply: [M,K]x[K,N] => 2*M*K*N (mul+add) ---
    def mm(self, M, K, N, tag=None):
        self.add(2*int(M)*int(K)*int(N), tag or "mm")
    # --- matrix-vector: [M,K]x[K] => 2*M*K ---
    def mv(self, M, K, tag=None):
        self.add(2*int(M)*int(K), tag or "mv")
    # --- batched matmul (batch,B): [B,M,K]x[B,K,N] => B*2*M*K*N ---
    def bmm(self, B, M, K, N, tag=None):
        self.add(2*int(B)*int(M)*int(K)*int(N), tag or "bmm")
    # --- einsum rough mapper for common patterns ---
    def einsum_blv_vh_to_blh(self, B, L, V, H, tag=None):
        # [b,l,h] = einsum("blh,vh->blv")^T 형태(실제는 blh,vh->blv 계산 후 소프트맥스 등)
        # 여기선 g_sim = einsum("blh,vh->blv") = B*L*V*H *2
        self.add(2*int(B)*int(L)*int(V)*int(H), tag or "einsum_blh_vh_to_blv")
    # --- indexing/scatter/gather: 보수적으로 요소 수만 카운트 ---
    def gather(self, *sizes, tag=None):
        n=1
        for s in sizes: n*=int(s)
        self.add(n, tag or "gather")
    def scatter_add(self, *sizes, tag=None):
        n=1
        for s in sizes: n*=int(s)
        self.add(n, tag or "scatter_add")
    # --- topk: 대략 N log2 K 비교(아주 러프) + 약간의 이동 비용 ---
    def topk(self, N, K, tag=None):
        import math
        self.add(int(N*max(1, math.log2(max(1,K)))) + int(N), tag or "topk")
    # --- softmax: exp(N)+sum(N)+div(N) ~ 3N (보수적으로 4N으로 잡자) ---
    def softmax(self, N, tag=None):
        self.add(4*int(N), tag or "softmax")
    # --- log/exp 등 스칼라-벡터 elementwise ---
    def exp(self, N, tag=None): self.elemwise(N, tag=tag or "exp")
    def log(self, N, tag=None): self.elemwise(N, tag=tag or "log")

def _infer_model_dims(model):
    cfg = getattr(model, "config", None)
    H = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    LAYERS = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    HEADS = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    V = getattr(cfg, "vocab_size", None)
    return H, LAYERS, HEADS, V

def estimate_transformer_forward_flops(counter: FlopsCounter, model, seq_len: int, new_tokens: int, mode: str):
    """
    mode: "ar_with_cache" (증분 1토큰 비용), "full_recompute" (전체 L 재계산),
          "logits_only" (lm_head 투영만, 실제로는 아래서 별도 카운트하므로 보통 불필요)
    근사식:
      - MHA: Q,K,V proj ~ 3*H^2, out proj ~ H^2  (모두 * seq_len)
      - FFN: 2*H*4H + 2*4H*H = 8H^2 (모두 * seq_len)
      - Attention score: QK^T ~ L^2 * H, Softmax*V ~ L^2 * H  (증분/캐시일 땐 L*H)
    레이어당: proj/FFN은 O(L*H^2), 어텐션 곱은 O(L^2*H) (AR cache 시 O(L*H))
    """
    H, L, _, _ = _infer_model_dims(model)
    if not (H and L):
        return  # 정보 없으면 생략

    if mode == "ar_with_cache":
        # 새 토큰 1개 기준
        Ltok = new_tokens  # 보통 1
        # per layer:
        # projections/ffn at current position only (1 token)
        per_layer = ( (3 + 1 + 8) * H * H ) * Ltok  # (QKV+out+FFN) * H^2
        # attention scores/value mix: O(seq_len * H) for the new token
        attn = 2 * seq_len * H  # QK + softmaxV (rough)
        counter.add(L * (per_layer + attn), tag="xformer_ar_cache", extra={"L":L,"seq":seq_len,"H":H})
    elif mode == "full_recompute":
        # 전체 시퀀스 길이 seq_len을 매 step마다 재계산(보수적)
        # per layer:
        proj_ffn = (3 + 1 + 8) * seq_len * H * H
        attn = 2 * (seq_len * seq_len) * H
        counter.add(L * (proj_ffn + attn), tag="xformer_full_recompute", extra={"L":L,"seq":seq_len,"H":H})
    else:
        pass  # 필요시 확장
def _topk_filtering(logits: torch.Tensor, topk_k: int) -> torch.Tensor:
    if topk_k is None or topk_k <= 0 or topk_k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, k=topk_k)
    cutoff = values[-1]
    mask = logits < cutoff
    out = logits.clone()
    out[mask] = float('-inf')
    return out

def _topp_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    top_p = float(top_p)
    if not (0.0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 첫 토큰은 항상 남기고, 누적확률이 top_p를 넘는 구간 마스킹
    cutoff_mask = cumsum > top_p
    cutoff_mask[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[sorted_indices[cutoff_mask]] = True
    out = logits.clone()
    out[mask] = float('-inf')
    return out

def sample_with_temperature(
    logits_1d: torch.Tensor,
    temperature: float,
    sampling_strategy: str = "greedy",  # "greedy" | "top-k" | "nucleus" (="top-p")
    topk_k: int = 64,
    top_p: float = 0.9,
) -> int:
    """
    1D logits에서 전통적 온도 샘플링.
    - temperature==0: greedy
    - temperature>0: (logits/temperature) → 필터 → softmax → multinomial
    """
    if temperature is None:
        temperature = 0.0

    if temperature <= 0.0:
        return int(torch.argmax(logits_1d, dim=-1))

    # 1) 온도 스케일링
    scaled = logits_1d / max(float(temperature), 1e-8)

    # 2) 전략별 필터링
    strat = (sampling_strategy or "greedy").lower()
    if strat == "top-k":
        scaled = _topk_filtering(scaled, topk_k)
    elif strat in ("nucleus", "top-p"):
        scaled = _topp_filtering(scaled, top_p)
    elif strat != "greedy":
        # 알 수 없는 전략이면 안전하게 greedy
        return int(torch.argmax(scaled, dim=-1))

    # 3) 확률화 & 견고성 처리
    probs = torch.softmax(scaled, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits_1d, dim=-1))

    # 4) 샘플
    return int(torch.multinomial(probs, num_samples=1).item())
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = -torch.log(noise)
    # temperature exponent 대신 scale로 사용
    return (logits.exp() / gumbel_noise).to(logits.dtype)


def compute_seq_emb(hidden: torch.Tensor, prompt_len: int) -> torch.Tensor:
    hidden_fp32 = hidden.float()
    valid_mask = torch.ones_like(hidden_fp32[..., :1])
    valid_mask[:, :prompt_len] = 0
    token_cnt = valid_mask.sum(dim=1).clamp(min=1)
    seq_emb   = (hidden_fp32 * valid_mask).sum(dim=1) / token_cnt
    return seq_emb.to(hidden.dtype)  # [1, H]
    # return seq_emb

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


import math
from typing import Literal

def schedule_weights(i: int, 
                     mode: Literal['logistic', 'linear', 'constant'] = 'logistic',
                     L0: int = 20,
                     rep_max: float = 1.0,
                     hidden_max: float = 500.0,
                     sharpness: float = 0.1,
                     eps: float = 1e-4):
    """
    i: 현재 step (0-based)
    mode: 스케줄링 방식 ('logistic', 'linear', 'constant')
    L0: 
        - logistic: 전환의 중심점 (s=0.5가 되는 지점)
        - linear: 전환이 완료되는 지점 (s가 1에서 0으로 떨어지는 기간)
    sharpness: (logistic 전용) 전환 급격함
    rep_max, hidden_max: 각 페널티의 최대 가중치
    eps: 수치적 임계값
    """
    
    # 1. 모드에 따른 감쇠 계수 s(i) 계산 (1.0 -> 0.0)
    if mode == 'constant':
        # 스케줄링 없이 둘 다 최대값 유지 (또는 사용자가 원하는 고정값)
        # 상보적 관계(s, 1-s)를 무시하고 각각 독립적인 상수로 반환
        return rep_max, hidden_max

    elif mode == 'linear':
        # 선형 감소: 0부터 L0까지 1->0으로 감소
        if i >= L0:
            s = 0.0
        else:
            s = 1.0 - (i / float(L0))
            
    else: # mode == 'logistic' (기본값, 기존 로직)
        s = 1.0 / (1.0 + math.exp(sharpness * (i - L0)))

    # 2. 가중치 계산 (상보적 스케줄)
    w_rep_raw    = rep_max    * s
    w_hidden_raw = hidden_max * (1.0 - s)

    # 3. 수치적 임계 처리 (eps)
    # Linear 모드 등에서 0에 근접할 때 확실하게 0으로 떨어뜨리기 위함
    w_rep    = 0.0 if w_rep_raw < rep_max * eps else w_rep_raw
    w_hidden = hidden_max if (hidden_max - w_hidden_raw) < hidden_max * eps else w_hidden_raw

    return w_rep, w_hidden
# 정규화(원래 코드와 동일: center + RMS) 유틸

def _norm_last_dim(t):
    mu  = t.mean(dim=-1, keepdim=True)
    gcz = t - mu
    rms = torch.sqrt((gcz.pow(2).mean(dim=-1, keepdim=True)) + 1e-8)
    return gcz / rms




def resolve_vocab_projection_weight(model):
    # 1) lm_head.weight (가장 표준적)
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight") and model.lm_head.weight is not None:
        return model.lm_head.weight

    # 2) get_output_embeddings().weight
    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight is not None:
            return out_emb.weight
    except Exception:
        pass

    # 3) get_input_embeddings().weight (tied)
    try:
        in_emb = model.get_input_embeddings()
        if in_emb is not None and hasattr(in_emb, "weight") and in_emb.weight is not None:
            return in_emb.weight
    except Exception:
        pass

    # 4) LLaMA/Mistral 계열 공통 경로
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model.model.embed_tokens, "weight"):
        return model.model.embed_tokens.weight

    # 5) GPT-2 계열
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and hasattr(model.transformer.wte, "weight"):
        return model.transformer.wte.weight

    # 6) 다른 디코더(HF Bart류 디코더 전용 등)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens") \
       and hasattr(model.model.decoder.embed_tokens, "weight"):
        return model.model.decoder.embed_tokens.weight

    raise RuntimeError("Could not resolve vocab projection matrix: no lm_head/get_output_embeddings/input_embeddings or known fallbacks.")


__all__ = [
    # --- Public Functions & Classes (일반 함수) ---
    "aggregate_diversity_over_prompts",
    "load_prompts_from_coco_hf_generic",
    "vae_decode_224",
    "pil_to_vae_latent",
    "flops_scope",
    "FlopsCounter",
    "estimate_transformer_forward_flops",
    "sample_with_temperature",
    "add_gumbel_noise",
    "compute_seq_emb",
    "get_num_transfer_tokens",
    "schedule_weights",
    "resolve_vocab_projection_weight",

    # --- Private Functions (언더바 _ 로 시작하는 함수들) ---
    "_is_number",
    "_normalize_caption_list",
    "_get_image_from_record",
    "_load_clip",
    "_clip_image_emb_batch",
    "_clip_img_emb",
    "_clip_txt_emb",
    "_append_mean_every_15",
    "_FLOPS_GLOBAL",   # 전역 변수
    "_flops_safe",
    "_infer_model_dims",
    "_topk_filtering",
    "_topp_filtering",
    "_norm_last_dim",
]