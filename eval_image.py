import os, io, re, json, base64, hashlib
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import base64
from io import BytesIO


# 선택: OpenAI 사용 시만 import
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # 패키지 없을 때 대비

client = OpenAI(api_key="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",)

def _sha256_pil(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()

@torch.no_grad()
def _load_clip(clip_id="openai/clip-vit-base-patch32", device="cuda", dtype=torch.float16):
    model = CLIPModel.from_pretrained(clip_id,use_safetensors=True).to(device, dtype=dtype).eval()
    proc  = CLIPProcessor.from_pretrained(clip_id)
    return model, proc

@torch.no_grad()
def _clip_image_emb_batch(clip_model, clip_proc, pil_list: List[Image.Image], device="cuda") -> torch.Tensor:
    # returns [N, D], fp32 normalized
    inputs = clip_proc(text=None, images=pil_list, return_tensors="pt").to(device)
    emb = clip_model.get_image_features(**inputs)
    emb = F.normalize(emb.to(torch.float32), dim=-1)
    return emb


def _image_to_data_url(img, fmt="PNG"):
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def _score_quality_with_llm(
    images: List[Image.Image],
    prompt_text: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    동일 프롬프트에서 생성된 '각 이미지의 생성 품질'을 0~1로 채점.
    반환:
    {
      "per_image": [{"idx":0,"score":0.xx,"reason":"..."}, ...],
      "score_mean": 0.xx,
      "coverage": N,
      "raw": {...원문 JSON...}  # 파서 실패시 None
    }
    """
    if OpenAI is None:
        return {"per_image": [], "score_mean": None, "coverage": 0, "raw": None, "reason": "openai_package_missing"}

    system_prompt = (
        "You are a strict judge of IMAGE GENERATION QUALITY.\n"
        "You will receive multiple images that were generated from the SAME text prompt.\n"
        "Rate EACH image individually on a 0.0–1.0 scale for: coherence, absence of artifacts, composition, lighting, and overall aesthetics.\n"
        "Do NOT compare images to each other; judge absolute quality per image.\n"
        "Return ONLY JSON of the form:\n"
        "{\n"
        "  \"per_image\": [{\"idx\": <int>, \"score\": <float 0.0~1.0>, \"reason\": \"<short>\"}, ...],\n"
        "  \"score_mean\": <float>\n"
        "}\n"
        "Keep reasons short.\n"
    )

    parts: List[dict] = []
    if prompt_text:
        parts.append({"type": "text", "text": f"Prompt (context only, do not score alignment): {prompt_text}"})
    parts.append({"type": "text", "text": f"Number of images: {len(images)}. Rate each by its index 0..{len(images)-1}."})

    js = _llm_json_score(system_prompt, user_prompt=parts, model_name=model_name, api_key=api_key, images=images)
    if not js or "per_image" not in js:
        return {"per_image": [], "score_mean": None, "coverage": 0, "raw": js, "reason": "parse_fail"}

    # 정규화/검증
    out_list = []
    for item in js.get("per_image", []):
        try:
            idx = int(item.get("idx", len(out_list)))
            sc  = float(item.get("score", 0.0))
            sc  = max(0.0, min(1.0, sc))
            rsn = str(item.get("reason", ""))
            out_list.append({"idx": idx, "score": sc, "reason": rsn})
        except Exception:
            continue

    score_mean = None
    if out_list:
        score_mean = sum(x["score"] for x in out_list) / len(out_list)

    return {
        "per_image": out_list,
        "score_mean": score_mean,
        "coverage": len(out_list),
        "raw": js,
    }

def _llm_json_score(system_prompt: str,
                    user_prompt: str,
                    model_name: str = "gpt-4.1-2025-04-14",
                    api_key=None,
                    images=None):
    """
    user_prompt: 반드시 문자열(단일 프롬프트)
    images: None | List[str | PIL.Image]  (str은 http/https 또는 data URL)
    """
    # 0) 보수적으로 문자열 강제 (혹시 실수 방지)
    if not isinstance(user_prompt, str):
        user_prompt = str(user_prompt)

    # 1) user.content 구성
    if not images:
        # 텍스트만 있을 때는 content를 '단일 문자열'로
        user_content = user_prompt
    else:
        # 이미지가 있으면 '파트 리스트'로
        parts = [{"type": "text", "text": user_prompt}]
        for im in images:
            if hasattr(im, "save"):          # PIL.Image
                data_url = _image_to_data_url(im)
            elif isinstance(im, str):        # http/https 또는 data URL
                data_url = im
            else:
                raise ValueError("Unsupported image type (expected str URL/data-URL or PIL.Image)")
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        user_content = parts

    # 2) OpenAI 호출 (chat.completions는 max_tokens 사용)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=512,
    )

    # 3) JSON 추출
    text = resp.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
def _score_diversity_with_llm(
    images: List[Image.Image],
    prompt_text: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    이미지 리스트를 LLM에게 보내 '서로 얼마나 다양한지' 0~1 점수로 받는다.
    """
    if OpenAI is None:
        return {"score": 0.0, "reason": "openai_package_missing", "raw": None}

    system_prompt = (
        "You are a strict judge of IMAGE DIVERSITY.\n"
        "You will be given multiple images that were generated from the SAME text prompt.\n"
        "Judge how different these images are from each other in content, composition, style, and color palette.\n"
        "Return ONLY a JSON object: {\"score\": <float 0.0~1.0>, \"reason\": \"<short>\"}.\n"
        "0.0 = nearly identical; 1.0 = maximally diverse.\n"
        "Do NOT evaluate prompt-image alignment; only cross-image diversity."
    )

    msg: List[dict] = []
    if prompt_text:
        msg.append({"type": "text", "text": f"Prompt (for context only): {prompt_text}"})



    js = _llm_json_score(system_prompt, msg, model_name=model_name, api_key=api_key, images=images)
    if not js or "score" not in js:
        return {"score": 0.0, "reason": "parse_fail", "raw": js}
    try:
        sc = float(js.get("score", 0.0))
        sc = max(0.0, min(1.0, sc))
    except Exception:
        sc = 0.0
    return {"score": sc, "reason": str(js.get("reason", "")), "raw": js}

def evaluate_diversity(
    image_paths: List[str],
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    # ▼ LLM 다양성 옵션
    use_llm: bool = True,
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    llm_prompt_text: Optional[str] = None,  # 동일 프롬프트 텍스트를 전달(선택)
    llm_max_images: int = 8,                # 토큰/비용 방지를 위해 상한
) -> Dict[str, Any]:
    """
    한 프롬프트에 대해 생성된 이미지 경로들을 받아 다양성 평가.
    반환 dict에 clip 기반 지표 + (옵션) llm 기반 지표를 모두 포함.
    """
    # --- 이미지 로드 ---
    pils, sizes, bytes_list, hashes = [], [], [], []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            pils.append(img)
            sizes.append(img.size)
            bytes_list.append(os.path.getsize(p) if os.path.exists(p) else 0)
            hashes.append(_sha256_pil(img))
        except Exception:
            continue

    N = len(pils)
    if N == 0:
        out = {
            "num_images": 0,
            "diversity_clip": 0.0,
            "clip_mean_cosine": 1.0,
            "unique_ratio": 0.0,
            "note": "no images",
        }
        if use_llm:
            out["diversity_llm"] = {"score": 0.0, "reason": "no_images", "raw": None}
        return out

    # --- CLIP 다양성 ---
    clip_model, clip_proc = _load_clip(
        clip_id=clip_model_id,
        device=device,
        dtype=(torch.float16 if device.startswith("cuda") else torch.float32),
    )
    embs = _clip_image_emb_batch(clip_model, clip_proc, pils, device=device).cpu().numpy()  # [N,D]

    sims = embs @ embs.T  # [N,N], cos sim
    iu = np.triu_indices(N, k=1)
    pair = sims[iu]
    mean_cos = float(pair.mean()) if pair.size > 0 else 1.0
    min_cos  = float(pair.min())  if pair.size > 0 else 1.0
    max_cos  = float(pair.max())  if pair.size > 0 else 1.0
    std_cos  = float(pair.std())  if pair.size > 0 else 0.0
    diversity_clip = 1.0 - mean_cos
    unique_ratio   = len(set(hashes)) / float(N)

    out: Dict[str, Any] = {
        "num_images": N,
        "diversity_clip": diversity_clip,
        "clip_mean_cosine": mean_cos,
        "clip_min_cosine":  min_cos,
        "clip_max_cosine":  max_cos,
        "clip_std_cosine":  std_cos,
        "unique_ratio": unique_ratio,
        "size_stats": {
            "width_mean":  float(np.mean([w for (w,h) in sizes])),
            "height_mean": float(np.mean([h for (w,h) in sizes])),
            "width_std":   float(np.std([w for (w,h) in sizes])),
            "height_std":  float(np.std([h for (w,h) in sizes])),
        } if sizes else {},
        "file_bytes_stats": {
            "bytes_mean": float(np.mean(bytes_list)) if bytes_list else 0.0,
            "bytes_std":  float(np.std(bytes_list))  if bytes_list else 0.0,
        },
    }

    # --- LLM 다양성(옵션) ---
    if use_llm:
        # 과도한 토큰/비용 방지를 위해 최대 llm_max_images 장만 평가
        sample_imgs = pils[:llm_max_images]
        llm_res = _score_diversity_with_llm(
            images=sample_imgs,
            prompt_text=llm_prompt_text,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model_name=llm_model,
        )
        out["diversity_llm"] = llm_res
        try:
            q_js = _score_quality_with_llm(sample_imgs, prompt_text=llm_prompt_text,
                                           api_key=openai_api_key, model_name=llm_model)
            out["quality_llm"] = q_js
        except Exception as e:
            out["quality_llm"] = {"per_image": [], "score_mean": None, "coverage": 0, "raw": None,
                                     "reason": f"exception:{e}"}

    return out
