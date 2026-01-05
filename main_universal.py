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
from utils import *
from utils import _append_mean_every_15, _norm_last_dim 


def generate_and_analyze_logits_per_step_ARver(
    model, tokenizer, prompt, output_filepath, 
    sim_ver, rep_ver,
    previous_responses=[],prev_probs=None,prev_embs=None,
    fill_order=[],
    gen_length=32, steps=32,
    knn_k=5,
    threshold=0.1,
    temperature=0,
    k_gamma=0.5,          # 그대로 둡니다(호환)
    L0=20,
    constant=1,          # 그대로 둡니다(호환/미사용)
    prev_step_traces=None,
    topk_k: int = 64,

    # ★ 추가된 인자들
    use_rep_penalty: bool = True,      # rep만/끄기 제어
    use_hidden_penalty: bool = True,   # hidden(sim)만/끄기 제어
    rep_max: float = 1.0,              # rep_penalty 최대 가중치
    hidden_max: float = 500.0,         # hidden(sim)_penalty 최대 가중치
    sharpness: float = 0.1,            # 전환 급격함

    sampling_strategy: str = "greedy",  # "greedy" | "top-k" | "nucleus"
    top_p: float = 0.9,
    mode='logistic'
):
    """
    AR LLM용: 매 스텝에서 temp_logits에 rep/hidden penalty 기반 바이어스를 '직접' 적용한 뒤,
    다음 토큰을 샘플링하여 x에 이어붙이는 방식. (autograd 미사용)
    반환값 시그니처/형태는 기존과 동일하게 유지.
    """
    # device = prompt.device
    mask_id = tokenizer.mask_token_id or 126336  # 그대로 둡니다(호환)

    _flops = FlopsCounter()
    # _flops_path = output_filepath + ".flops.jsonl"
    def _log_flops(event, extra=None):
        # with open(_flops_path, "a", encoding="utf-8") as _ff:
        #     rec = {"event": event, "flops_total": _flops.total, "time": time.time()}
        #     if extra: rec.update(extra)
        #     _ff.write(json.dumps(rec) + "\n")

        # # ✅ 추가: 15개마다 평균 row
        # if event == "AR_done":
        #     _append_mean_every_15(_flops_path, done_event="AR_done", mean_event="AR_mean_15")
        pass


    # AR 모드에선 '마스크 채우기'가 아니라 '순차 생성'이므로,
    # x는 단순히 prompt로 시작해서 토큰을 한 개씩 append 합니다.
    x = prompt.clone()  # [1, L0]
    prompt_len = x.shape[1]

    # 기록용(기존 키 이름과 호환)
    cur_topk_idx_steps = []
    cur_topk_logit_steps = []
    cur_seqemb_steps   = []
    cur_topk_prob_steps = []

    # prev_step_traces 준비
    if prev_step_traces is None:
        prev_step_traces = []

    # 출력 파일 append 모드(호환; 필요시 사용자 로깅용)
    # f = open(output_filepath, "a", encoding="utf-8")

    # 투영 행렬 (보통 tying): lm_head.weight 또는 get_output_embeddings().weight
    # 대부분 HF Llama/Mistral은 lm_head.weight가 어휘 투영이며 임베딩과 tied.
    W_param = resolve_vocab_projection_weight(model)
    if hasattr(W_param, "data"):
        W = W_param.data.to(torch.bfloat16) # 강제로 bf16으로 변환해서 가져옴
    else:
        W = W_param.to(torch.bfloat16)

    # 스케줄러: 두 페널티 모두 쓸 때만 L0 의미. 아니면 상수 가중.
    def _get_weights(step_idx: int):
        if use_rep_penalty and use_hidden_penalty:
            wr, wh = schedule_weights(step_idx, L0=L0, rep_max=rep_max, hidden_max=hidden_max, sharpness=sharpness,mode=mode)
        elif use_rep_penalty and not use_hidden_penalty:
            wr, wh = rep_max, 0.0
        elif (not use_rep_penalty) and use_hidden_penalty:
            wr, wh = 0.0, hidden_max
        else:
            wr, wh = 0.0, 0.0
        return wr, wh

    raw_logits = None
    last_hidden = None

    # AR 생성 루프: gen_length 또는 steps 중 작은 값만큼 진행 (호환을 위해 min 사용)
    total_steps = min(gen_length, steps)

    for i in trange(total_steps):
        # 현재 시퀀스로 forward (hidden 포함)
        outputs = model(x, output_hidden_states=True, use_cache=True)
        
        # ==== FLOPS: estimate model forward (AR cache, 1 new token) ====
        seq_len_now = x.shape[1]
        estimate_transformer_forward_flops(_flops, model, seq_len=seq_len_now, new_tokens=1, mode="ar_with_cache")

        logits = outputs.logits  # [B, L, V]
        device=logits.device
        W=W.to(device)
        hidden = outputs.hidden_states[-1]  # [B, L, H]
        raw_logits = logits  # 마지막 raw_logits 저장
        last_hidden = hidden

        # 현재 step의 대상 로짓은 "마지막 위치" (AR 생성의 표준)
        # temp_logits는 수정 가능한 사본
        temp_logits = logits.clone()  # [B, L, V]
        dev = temp_logits.device

        # 스케줄 가중
        w_rep, w_hidden = _get_weights(i)

        # L' (프롬프트 이후 길이) 및 현재 타겟 인덱스
        L_total = temp_logits.shape[1]
        j = L_total - 1   # 마지막 위치(이번에 뽑을 토큰)
        Lprime = max(0, L_total - prompt_len)  # 프롬프트 이후 토큰 개수

        # ---------- (1) REP penalty: 분포 내적 회피 ----------
        # prev_step_traces에는 'probs_topk_idx_steps' / 'probs_topk_val_steps' 기록이 (있다면) 담겨 있음.
        # AR에서는 현재 위치 j 한 줄만 대상으로 처리.

        if use_rep_penalty and len(prev_step_traces) > 0 and w_rep > 0.0:
            # 현재 분포 p (full-softmax; 안정성 위해 float32)
            z_row = temp_logits[0, j].to(torch.float32)      # [V]
            p_row = F.softmax(z_row, dim=-1)                 # [V]
            _flops.softmax(z_row.numel(), tag="rep.softmax_row")

            # 후보(과거 step 기록)들 중에서 '현재 스텝 i에 해당하는 top-k 분포'를 모은다.
            # (없으면 스킵)
            rep_Q_list = []
            for tr in prev_step_traces:
                if ('probs_topk_idx_steps' not in tr) or ('probs_topk_val_steps' not in tr):
                    continue

                # 바깥 리스트(길이 1)에서 실제 [L,K] 텐서를 꺼냄
                idx_obj = tr['probs_topk_idx_steps']
                val_obj = tr['probs_topk_val_steps']
                idx_t = idx_obj[0] if isinstance(idx_obj, list) else idx_obj   # [L,K] 또는 [K]
                val_t = val_obj[0] if isinstance(val_obj, list) else val_obj   # [L,K] 또는 [K]

                # 디바이스/ dtype 정리
                idx_t = idx_t.to(dev)
                val_t = val_t.to(dev, dtype=p_row.dtype)

                # 포맷에 따라 i번째 row를 안전하게 꺼내기
                if idx_t.ndim == 2 and val_t.ndim == 2:
                    # [L,K] 형식: i가 길이를 넘으면 마지막 행 사용
                    row = min(i, idx_t.shape[0]-1)
                    idx_row = idx_t[row]           # [K]
                    val_row = val_t[row]           # [K]
                elif idx_t.ndim == 1 and val_t.ndim == 1:
                    # [K] 형식
                    idx_row = idx_t
                    val_row = val_t
                else:
                    continue

                # q_full 복원
                q_full = torch.zeros_like(p_row)                  # [V]
                q_full.index_add_(0, idx_row.long(), val_row)     # [V]
                _flops.scatter_add(idx_row.numel(), tag="rep.index_add_v")
                rep_Q_list.append(q_full)

            if len(rep_Q_list) > 0:
                Q = torch.stack(rep_Q_list, dim=0)                # [N, V]
                _flops.mv(M=Q.shape[0], K=Q.shape[1], tag="rep.mv_Q_p")
                ptq = torch.mv(Q, p_row)                          # [N]
                _flops.elemwise(Q.numel(), tag="rep.elemwise_p_times_Q")
                _flops.mm(M=ptq.numel(), K=p_row.numel(), N=1, tag="rep.outer_ptq_p")  # outer ~ MxN with K mults; approximate
                _flops.elemwise(Q.numel(), tag="rep.elemwise_sub")
                g_rep_each = p_row * Q - torch.outer(ptq, p_row)  # [N, V]
                g_rep = g_rep_each.mean(dim=0)                    # [V]
                _flops.reduce(g_rep_each.numel(), tag="rep.mean")
                # 정규화
                _flops.elemwise(g_rep.numel(), tag="rep.norm")

                g_rep = _norm_last_dim(g_rep)                # 안정화(선택)

                # 회피(내적↓) → 반대방향
                # print(temp_logits)
                temp_logits[0, j] = temp_logits[0, j] - (w_rep * g_rep.to(temp_logits.dtype))
                _flops.elemwise(g_rep.numel(), tag="rep.apply")
                # print(temp_logits)
       ## ---------- (2) HIDDEN penalty: 마지막 토큰 hidden(h_t) 기반 (대표값 1개: max) ----------
        if use_hidden_penalty and len(prev_step_traces) > 0 and w_hidden > 0.0:
            # j는 위에서 L_total - 1 로 이미 계산됨 (마지막 토큰 위치; 여기서 절대 덮어쓰지 말 것)
            h_t = hidden[:, -1, :][0]  # [H]

            # step i에 해당하는 참조 hidden들을 수집 (포맷 방어)
            sim_list = []
            for tr in prev_step_traces:
                if 'seq_emb_steps' not in tr:
                    continue
                obj = tr['seq_emb_steps']

                # case 1) [Tensor [L,H]]을 리스트로 1개 담은 포맷
                if isinstance(obj, list) and len(obj) > 0 and torch.is_tensor(obj[0]):
                    t = obj[0].to(dev, dtype=hidden.dtype)   # [L,H] 또는 [H]
                    if t.ndim == 2:
                        row_idx = min(i, t.shape[0] - 1)
                        sim_list.append(t[row_idx])          # [H]
                    elif t.ndim == 1:
                        sim_list.append(t)                   # [H]
                # case 2) 바로 Tensor
                elif torch.is_tensor(obj):
                    t = obj.to(dev, dtype=hidden.dtype)      # [L,H] 또는 [H]
                    if t.ndim == 2:
                        row_idx = min(i, t.shape[0] - 1)
                        sim_list.append(t[row_idx])          # [H]
                    elif t.ndim == 1:
                        sim_list.append(t)                   # [H]
                # 그 외 포맷은 무시

            if len(sim_list) > 0:
                refs = torch.stack(sim_list, dim=0)          # [N, H]
                sims = torch.matmul(refs, h_t)               # [N] (내적 유사도)
                _flops.mv(M=refs.shape[0], K=refs.shape[1], tag="hidden.mv_refs_ht")
                dL_dh = refs[torch.argmax(sims)]             # ✅ 대표 하나만 선택 (max) → [H]

                # 로짓 공간으로 투영: z = W h_t (임베딩 tying 가정)
                dL_dh_vec = dL_dh.to(W.dtype)                # dtype 정렬
                _flops.mv(M=W.shape[0], K=W.shape[1], tag="hidden.W_mv")  # VxH * H
                g_row = torch.matmul(W, dL_dh_vec)           # [V] = (V,H) @ (H)

                # 정규화(너의 헬퍼로 통일)
                _flops.elemwise(g_row.numel(), tag="hidden.norm")
                g_row = _norm_last_dim(g_row).to(temp_logits.dtype)

                # 회피(내적 ↓) → 반대방향으로 단 1회 업데이트
                temp_logits[0, j] = temp_logits[0, j] - (w_hidden * g_row)
                _flops.elemwise(g_row.numel(), tag="hidden.apply")


                # print(temp_logits)

        # ---------- 최종 조정된 로짓으로 확률/샘플링 ----------
        adjusted_logits = temp_logits
        adjusted_probs  = F.softmax(adjusted_logits[0, j], dim=-1)  # [V]
        _flops.softmax(adjusted_logits.shape[-1], tag="post.softmax_row")

        # 기록(Top-K 및 확률)
        topk_logit_vals, topk_idx = torch.topk(adjusted_logits[0, j], k=min(topk_k, adjusted_logits.shape[-1]), dim=-1)
        _flops.topk(N=adjusted_logits.shape[-1], K=min(topk_k, adjusted_logits.shape[-1]), tag="post.topk")
        # 정확한 확률로 top-k 값 추출
        topk_prob_vals = adjusted_probs.gather(0, topk_idx.to(adjusted_probs.device))
        _flops.gather(topk_idx.numel(), tag="post.gather_topk_probs")

        # --- 기록: top-k, 확률은 동일 ---
        cur_topk_idx_steps.append(topk_idx.detach().to('cpu', dtype=torch.int32).unsqueeze(0))
        cur_topk_logit_steps.append(topk_logit_vals.detach().to('cpu'))
        cur_topk_prob_steps.append(topk_prob_vals.detach().to('cpu').to(torch.float16))

        # --- 기존 키 이름을 유지하되, 이제 마지막 토큰 hidden(h_t)을 넣음 ---
        cur_seqemb_steps.append(hidden.detach()[0, -1, :].to('cpu'))  # [H]

        # 샘플링 (온도/검증용 Gumbel 지원)
        z_row = adjusted_logits[0, j]
        # z_row = add_gumbel_noise(z_row, temperature=temperature)
        # next_id = torch.argmax(z_row, dim=-1).view(1, 1)  # greedy; 필요시 top-p/typical로 바꿔도 OK
        if temperature > 0.0:
            chosen = sample_with_temperature(
                z_row,
                temperature=temperature,
                sampling_strategy=sampling_strategy,  # "greedy" | "top-k" | "nucleus"
                topk_k=topk_k,
                top_p=top_p,
            )
            next_id = torch.tensor([[chosen]], device=z_row.device, dtype=torch.long)
        else:
            # temperature==0 -> greedy
            next_id = torch.argmax(z_row, dim=-1, keepdim=True).view(1, 1)

        # x에 append
        x = torch.cat([x, next_id.to(x.device)], dim=1)
        _log_flops("AR_step", {"step": int(i), "seq_len": int(x.shape[1])})

    # 루프 종료: raw_logits/hidden는 마지막 forward 결과를 반환
    # f.close()

    # prev_step_traces 호환 레코드 구성
    record = None
    record = None
    if use_hidden_penalty or use_rep_penalty:
        record = {
            'probs_topk_idx_steps': [torch.cat(cur_topk_idx_steps, dim=0)],  # [gen_len, K]
            'probs_topk_val_steps': [torch.stack(cur_topk_prob_steps, dim=0)],  # [gen_len, K]
            'seq_emb_steps':        [torch.stack(cur_seqemb_steps, dim=0)],  # [gen_len, H]  ← h_t들이 들어있음
            'fill_order':           None,
        }

    # 기존 반환 형태 유지: (x, softmax(raw_logits), hidden, record)
    # raw_logits는 마지막 forward의 로짓 (전체 시퀀스 길이 기준)
    _log_flops("AR_done", {"steps": int(total_steps)})

    return x, F.softmax(raw_logits, dim=-1), last_hidden, record



def generate_and_analyze_logits_per_step(
    model, tokenizer, prompt, output_filepath, 
    sim_ver, rep_ver,
    previous_responses=[],prev_probs=None,prev_embs=None,
    fill_order=[],
    gen_length=32, steps=32,
    knn_k=5,
    threshold=0.1,
    temperature=0,
    k_gamma=0.5,          # 그대로 둡니다(호환)
    L0=20,
    constant=1,          # 그대로 둡니다(호환/미사용)
    prev_step_traces=None,
    topk_k: int = 64,

    # ★ 추가된 인자들
    use_rep_penalty: bool = True,      # rep만/끄기 제어
    use_hidden_penalty: bool = True,   # hidden(sim)만/끄기 제어
    rep_max: float = 1.0,              # rep_penalty 최대 가중치
    hidden_max: float = 500.0,         # hidden(sim)_penalty 최대 가중치
    sharpness: float = 0.1,            # 전환 급격함
    sampling_strategy: str = "greedy",  # "greedy" | "top-k" | "nucleus"
    top_p: float = 0.9,
    mode='logistic'
):
    # steps = max(steps, gen_length)
    
    _flops = FlopsCounter()
    _flops_path = output_filepath + ".flops.jsonl"
    def _log_flops(event, extra=None):
        # with open(_flops_path, "a", encoding="utf-8") as _ff:
        #     rec = {"event": event, "flops_total": _flops.total, "time": time.time()}
        #     if extra: rec.update(extra)
        #     _ff.write(json.dumps(rec) + "\n")

        # # ✅ 추가
        # if event == "NAR_done":
        #     _append_mean_every_15(_flops_path, done_event="NAR_done", mean_event="NAR_mean_15")
        pass
    
    mask_id = tokenizer.mask_token_id or 126336
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_len = prompt.shape[1]
    initial_mask_region = (x[:, prompt_len:] == mask_id)                # [1, gen_length]
    num_transfer_tokens = get_num_transfer_tokens(initial_mask_region, steps)  # [1, steps]

    raw_output = model(x)
    estimate_transformer_forward_flops(_flops, model, seq_len=x.shape[1], new_tokens=x.shape[1], mode="full_recompute")
    logit_bias = torch.zeros_like(raw_output.logits)
    # W = model.get_output_embeddings().weight.to(dev, non_blocking=True)
    raw_logits=None

    cur_topk_idx_steps = []
    cur_topk_logit_steps = []
    cur_seqemb_steps   = []
    cur_topk_prob_steps = [] 

    chosen_token_ids = []   # ★ 선택된 토큰들을 모아둘 리스트
    chosen_token_strs = []  # ★ 사람이 읽기 쉽게 decode한 문자열 
    
    fixed_fill_order = None   # list[int] (전역 pos들)
    fill_ptr = 0              # 이번 호출 내에서 진행 포인터

    # with open(output_filepath, "a", encoding="utf-8") as f:
    for i in trange(steps):
        if not (x == mask_id).any():
            print("All masks have been filled. Breaking.")
            break

        output      = model(x, output_hidden_states=True)
        estimate_transformer_forward_flops(_flops, model, seq_len=x.shape[1], new_tokens=x.shape[1], mode="full_recompute")

        raw_logits  = output.logits
        hidden      = output.hidden_states[-1]
        dev = raw_logits.device
        W = model.get_output_embeddings().weight.to(dev, non_blocking=True)

        # ── (A) 스케줄 가중치 계산
        if use_rep_penalty and use_hidden_penalty:
        # L0는 둘 다 쓸 때만 의미
            w_rep, w_hidden = schedule_weights(
                i=i, L0=L0, rep_max=rep_max, hidden_max=hidden_max, sharpness=sharpness, mode=mode,
            )
        elif use_rep_penalty and not use_hidden_penalty:
            w_rep = rep_max         # rep만 상수 스케줄
            w_hidden = 0.0
        elif (not use_rep_penalty) and use_hidden_penalty:
            w_rep = 0.0
            w_hidden = hidden_max   # hidden만 상수 스케줄
        else:
            # none 모드: 둘 다 끔
            w_rep = 0.0
            w_hidden = 0.0

        # print(w_rep)
        # print(w_hidden)
        # new!
        temp_logits = raw_logits.clone()
        
        L_total = temp_logits.shape[1]
        j0 = prompt_len
        Lprime = L_total - j0

        # === 후보 풀 만들기 (현재 step i의 과거 참조들) =========================
        rep_pool = []   # 요소: (idx_slice[L',K], q_slice[L',K])
        sim_pool = []   # 요소: e_ref[H]V = temp_logits.shape[-1]

        # 하이퍼 (필요시 cfg에 넣어도 됨)
        eta_rep = 1.0
        eta_sim = 1.0
        eta_rep_decay = 0.9
        eta_sim_decay = 0.9
        if prev_step_traces is not None and len(prev_step_traces) > 0:
            for tr in prev_step_traces:
                # REP 참조(같은 step i의 top-k 분포)
                if ('probs_topk_idx_steps' in tr and 'probs_topk_val_steps' in tr
                    and i < len(tr['probs_topk_idx_steps']) and i < len(tr['probs_topk_val_steps'])):
                    idx_mat = torch.tensor(tr['probs_topk_idx_steps'][i], device=dev, dtype=torch.long)          # [L, K]
                    val_mat = torch.tensor(tr['probs_topk_val_steps'][i], device=dev, dtype=temp_logits.dtype)   # [L, K]
                    idx_slice = idx_mat[j0:]    # [L', K]
                    q_slice   = val_mat[j0:]    # [L', K]
                    # q_slice   = q_slice / q_slice.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    rep_pool.append((idx_slice, q_slice))
                # SIM 참조(같은 step i의 seq_emb)
                if 'seq_emb_steps' in tr and i < len(tr['seq_emb_steps']):
                    sim_pool.append(torch.tensor(tr['seq_emb_steps'][i], device=dev, dtype=hidden.dtype))
        

    

        if use_rep_penalty and prev_step_traces is not None and len(prev_step_traces) > 0:
            # 후보 풀 만들기 (현재 step i의 top-k 분포들)
            

            # 후보를 하나씩 소진하며 매번 p 재계산 → CE 최소 참조부터 적용
            
            while len(rep_pool) > 0:
                # 1) 현재 분포 p_full
                p_full = F.softmax(temp_logits[0, j0:], dim=-1)   # [L', V]
                _flops.softmax(p_full.numel(), tag="rep.full.softmax")  # L'*V
                Lprime, V = p_full.shape

                # 2) 내적(similarity) s = sum_j p_j q_j 가 가장 큰 참조 선택
                best_idx, best_s = None, None
                # (원한다면, CE(q||p) 대신 s를 쓰므로 아래 루프만 바뀜)
                for ridx, (idx_slice, q_slice) in enumerate(rep_pool):
                    # p의 same support 값
                    pC = p_full.gather(1, idx_slice)          # [L', K]
                    _flops.gather(idx_slice.numel(), tag="rep.gather_support")
                    _flops.elemwise(pC.numel(), tag="rep.pos_mul")
                    _flops.reduce(pC.shape[0], tag="rep.pos_sum")
                    # q_slice는 합<1 원시 확률 (정규화 안 함)
                    s_pos = (pC * q_slice).sum(dim=1, keepdim=True)  # [L',1]
                    s = s_pos.mean()                                # scalar
                    if (best_s is None) or (s > best_s):
                        best_s, best_idx = s, ridx

                # 3) 선택된 참조로 그레디언트 계산: grad(+s) = p ⊙ q - (p·q) p
                # _log_p_minus_q(temp_logits, j0, rep_pool, tag=f"after_one_update_step_{i}", cheap_only=True)
                idx_slice, q_slice = rep_pool.pop(best_idx)
                _flops.scatter_add(Lprime*idx_slice.shape[1], tag="rep.scatter_add_qfull")

                # q_full 복원 (top-k 위치에만 질량, 나머지 0)
                q_full = torch.zeros(Lprime, V, device=p_full.device, dtype=p_full.dtype)  # [L',V]
                q_full.scatter_add_(1, idx_slice, q_slice)

                # 위치별 내적 <p, q> (열 합)
                _flops.elemwise(p_full.numel(), tag="rep.mul_full_q")
                _flops.reduce(Lprime, tag="rep.sum_rows")
                s_pos = (p_full * q_full).sum(dim=1, keepdim=True)  # [L',1]

                # grad(+s)
                _flops.elemwise(p_full.numel(), tag="rep.grad_build")

                grad_pos = p_full * q_full - s_pos * p_full         # [L',V]

                # (선택) 위치 평균 스케일 (네 기존 스타일 유지)
                # grad_pos = grad_pos / float(Lprime)
                _flops.elemwise(grad_pos.numel(), tag="rep.norm")
                grad_pos = _norm_last_dim(grad_pos)

                # 4) 로짓 업데이트: s를 줄이려면 반대방향
                g_rep = torch.zeros_like(temp_logits)   # [1, L, V]
                g_rep[0, j0:] = grad_pos
                _flops.elemwise(grad_pos.numel(), tag="rep.apply")
                # penalty 부호 음수(회피): temp_logits -= w_rep * grad(+s)
                temp_logits = temp_logits - (w_rep * g_rep)

        
        # ----------------------------------------
        # (2) SIM: 과거 embedding들을 순차적으로 모두 적용
        # ----------------------------------------
        if use_hidden_penalty and prev_step_traces is not None and len(prev_step_traces) > 0:
            # 현재 seq_emb (원래 버전: 프롬프트 이후 전 구간 평균; 마스크 포함)
            seq_emb_cur = compute_seq_emb(hidden, prompt_len)[0]  # [H]
            _flops.elemwise((hidden.shape[1]-prompt_len)*hidden.shape[2], tag="sim.meanpool_est")

            # 과거 참조 모으기
            sim_list = []
            for tr in prev_step_traces:
                if 'seq_emb_steps' in tr and i < len(tr['seq_emb_steps']):
                    sim_list.append(torch.tensor(tr['seq_emb_steps'][i],
                                                device=dev, dtype=hidden.dtype))

            if len(sim_list) > 0:
                refs = torch.stack(sim_list, dim=0)     # [N, H]
                sims = torch.matmul(refs, seq_emb_cur)  # [N]
                _flops.mv(M=refs.shape[0], K=refs.shape[1], tag="sim.mv_refs_cur")
                
                # 원래 옵션(sim_ver)에 따라 dL/de 선택
                if sim_ver == "avg":
                    _flops.reduce(refs.numel(), tag="sim.avg")
                    dL_de = refs.mean(dim=0)            # [H]
                elif sim_ver == "max":
                    dL_de = refs[torch.argmax(sims)]    # [H]
                else:
                    dL_de = refs[-1]                    # [H]

                # mean-pool 야코비안(프롬프트 이후 동일 분배)
                B, L, Hdim = hidden.shape
                T = max(0, L - prompt_len)
                g_hidden = torch.zeros_like(hidden)     # [1, L, H]
                if T > 0:
                    _flops.einsum_blv_vh_to_blh(B=1, L=L, V=W.shape[0], H=W.shape[1], tag="sim.einsum")
                    g_hidden[:, prompt_len:, :] = dL_de.view(1, 1, Hdim) / float(T)

                # logits 방향 투영 → 정규화 → temp_logits 갱신(유사도 ↓ 위해 −)
                g_sim = torch.einsum("blh,vh->blv", g_hidden, W)  # [1, L, V]
                _flops.elemwise(B*L*W.shape[0], tag="sim.norm_apply_est")
                g_norm = _norm_last_dim(g_sim)
                temp_logits = temp_logits - (w_hidden * g_norm)

        # ── 순차 적용 결과를 최종 adjusted 로짓으로 사용 ─────────────────────────
        adjusted_logits = temp_logits
        adjusted_probs  = F.softmax(adjusted_logits, dim=-1)
        _flops.softmax(adjusted_logits.numel(), tag="post.full.softmax")



        adjusted_logits = add_gumbel_noise(adjusted_logits, temperature=temperature)
        _flops.reduce(adjusted_logits.numel(), tag="post.argmax_est")
        # x_filled = adjusted_logits[0, select_index].argmax(dim=-1)          # [k_now]
        # x[0, select_index] = x_filled

        x0   = adjusted_logits.argmax(dim=-1)
        x0_p = torch.gather(adjusted_probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        mask_index = (x == mask_id)
        mask_index[:, :prompt_len] = False

        
        k_planned = int(num_transfer_tokens[0, i].item())     # i는 바깥 steps 루프 인덱스
        k_now     = min(k_planned, int(mask_index.sum().item()))
        if k_now > 0:
            # 비마스크/프롬프트/이미 채워진 곳은 제외(-inf)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

            # top-k 확신 위치 선택 (전역 인덱스)
            _, select_index = torch.topk(confidence[0], k=k_now)  # [k_now]

            # 선택 위치 토큰 채우기
            x[0, select_index] = x0[0, select_index]


        
        with torch.no_grad():
            logits_step = adjusted_logits.detach()[0]  
            topk_logit_vals, topk_idx = torch.topk(logits_step, k=topk_k, dim=-1)  # [L, K], [L, K]
            _flops.topk(N=logits_step.shape[-1]*logits_step.shape[0], K=topk_k, tag="post.topk_full")

            # 전체 softmax에서 확률 뽑기 (정확)
            probs_full = F.softmax(logits_step.to(torch.float32), dim=-1)           # [L, V]

            _flops.softmax(logits_step.numel(), tag="post.softmax_logitstep")

            topk_prob_vals = probs_full.gather(1, topk_idx)                         # [L, K]
            _flops.gather(topk_idx.numel(), tag="post.gather_full")

            cur_topk_idx_steps.append(topk_idx)
            cur_topk_logit_steps.append(topk_logit_vals)    # (원하면 계속 저장)
            cur_topk_prob_steps.append(topk_prob_vals)       
            # seq emb도 기존대로
            cur_seqemb_steps.append(compute_seq_emb(hidden.detach(), prompt_len)[0])

        _log_flops("NAR_step", {"step": int(i), "seq_len": int(x.shape[1])})
    # print("\n=== Generation finished ===")
    # print("Chosen token ids:", chosen_token_ids)
    # print("Chosen token strs:", chosen_token_strs)

    record=None
    if use_hidden_penalty or use_rep_penalty:
        record = {
            'probs_topk_idx_steps': [t.detach().to('cpu', dtype=torch.int32) for t in cur_topk_idx_steps],
            # 'probs_topk_val_steps': [F.softmax(v.detach().to(torch.float32), dim=-1).to(torch.float16).cpu()
            #                          for v in cur_topk_logit_steps],
            "probs_topk_val_steps": [p.detach().to(torch.float16).cpu() for p in cur_topk_prob_steps], # ★ 전체 softmax에서 뽑은 확률값 저장
            'seq_emb_steps':        [s.detach().to('cpu', dtype=torch.float32) for s in cur_seqemb_steps],
            'fill_order':           fixed_fill_order,   # ★ 고정 실행계획 저장
        }
        # prev_step_traces.append(record)
    _log_flops("NAR_done", {"steps": int(steps)})
    return x, F.softmax(raw_logits,dim=-1), hidden, record


from eval import evaluate_diversity
from transformers import BitsAndBytesConfig
def load_prompts_from_dataset(prompt_type, start_row=0, max_rows=20):
    if "reedsyPrompts" in prompt_type:
        path = "/home/zzangmane/2025_EMNLP_NegativeDecoding/reedsy_20plus_prompts.json"
    elif "writingPrompts" in prompt_type:
        path = "/home/zzangmane/2025_EMNLP_NegativeDecoding/writingprompts_20plus_prompts.json"
    else:
        raise ValueError
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [e["prompt"] for e in data[start_row:start_row+max_rows]]
def _is_llada(name: str) -> bool:
    # 필요시 키워드 더 추가 가능
    return ("llada" in name) or ("gsai-ml/llada" in name)
def main():
    parser = argparse.ArgumentParser(description="Generate text with LLaDA guidance.")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--num_rows", type=int, default=20)
    parser.add_argument("--start_row", type=int, default=0)
    # ★ 추가: 결과 파일 이름 구분용
    parser.add_argument("--result_suffix", type=str, default="test",
                        help="결과 파일명에 붙일 접미사(예: sweep_both_trial12)")
    parser.add_argument("--early_stop", action="store_true",
                    help="초반 es-cutoff 이내에 동일 응답이 나오면 즉시 조기 종료")
    parser.add_argument("--es_cutoff", type=int, default=10,
                    help="이 횟수 이전(iter index 기준)에는 중복 발생 시 즉시 종료. 이후에는 무시")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    print(args.config)
    print("suffix :")
    print(args.result_suffix)

    sim_ver = cfg.get('sim_penalty_version', 'max')
    rep_ver = cfg.get('rep_penalty_version', 'kl')
    degen_threshold=cfg.get('degen_threshold',0.1)
    temperature = cfg.get('temperature', 0)

    # ★ 스케줄/사용 플래그 관련 설정을 cfg에서 읽기
    use_rep_penalty    = cfg.get('use_rep_penalty', True)
    use_hidden_penalty = cfg.get('use_hidden_penalty', True)
    rep_max            = cfg.get('rep_max_constant', 1.0)
    hidden_max         = cfg.get('hidden_max_constant', 500.0)
    L0                 = cfg.get('L0', 20)
    sharpness          = cfg.get('sharpness', 1.0)
    sampling_strategy  = cfg.get('sampling','greedy')
    mode = cfg.get('mode','logistic')
    print(mode)
    top_p  = cfg.get('top_p',0.9)
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name_str = cfg['model_name'].lower()
    gen_step_fn = generate_and_analyze_logits_per_step if _is_llada(model_name_str) else generate_and_analyze_logits_per_step_ARver
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # 연산은 bf16으로
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
    )
    print(f"[routing] Using {'LLaDA (diffusion-style) fn' if _is_llada(model_name_str) else 'AR decoding fn'} for model: {cfg['model_name']}")
    if _is_llada(model_name_str):
        model = AutoModel.from_pretrained(
            cfg['model_name'], trust_remote_code=True,
            device_map="auto", torch_dtype=torch.bfloat16
        ).eval()
    else:
        # ★ AR용: 반드시 CausalLM 클래스로!
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_name'], trust_remote_code=True,device_map="auto",
            quantization_config=bnb_config,
        )
        model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)

    all_results = []
    prompt_list = ([cfg['prompt']] if not cfg.get('dataset') else
        load_prompts_from_dataset(cfg['dataset'], args.start_row, args.num_rows))
    
    print(f"w_rep : {rep_max}")
    print(f"h_rep : {hidden_max}")
    print(f"L0 : {L0}")
    print(f"sharpness : {sharpness}")
    print(f"use_rep : {use_rep_penalty}")
    print(f"use_hidden : {use_hidden_penalty}")
    print(f"num_rows : {len(prompt_list)}")
    print(f"temperature : {temperature}")
    
    for idx, prompt_text in enumerate(prompt_list):
        prev_probs, prev_embs = None, None
        previous_resps=[]
        previous_hidden=[]
        previous_probs=[]

        previous_traces=[]
        previous_step_traces=[]
        
        formatted = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful and creative assistant that always responds in English."},
             {"role":"user","content":"Please write a story from the following prompt." + f"\nPrompt:\n{prompt_text}"}],
            add_generation_prompt=True, tokenize=False
        )
        input_ids = torch.tensor(tokenizer(formatted)['input_ids']).unsqueeze(0).to(device)

        answers=[]
        times = []
        base = os.path.splitext(args.config)[0]
        early_stopped_flag = False
        
        for it in trange(cfg.get('num_iteration', 15)):
            start = time.time()
            seq, prev_prob, prev_hidden, record = gen_step_fn(
                model, tokenizer, input_ids, cfg.get('output_filename', base + '_log.jsonl'),
                sim_ver, rep_ver,
                previous_responses=previous_resps,prev_probs=prev_probs,prev_embs=prev_embs,
                gen_length=cfg.get('generation_length',32),
                steps=cfg.get('steps',32),
                temperature=temperature,
                prev_step_traces=previous_step_traces,
                topk_k=cfg.get('topk_k', 20),

                # ★ 새 스케줄/플래그 전달
                use_rep_penalty=use_rep_penalty,
                use_hidden_penalty=use_hidden_penalty,
                rep_max=rep_max,
                hidden_max=hidden_max,
                L0=L0,
                sharpness=sharpness,
                sampling_strategy=sampling_strategy,
                top_p=top_p,
                mode=mode,
            )
            elapsed = time.time() - start
            new_ids = seq[0, input_ids.shape[1]:]
            new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
            print(f"result : {new_text}")
            print()
            answers.append(new_text)
            times.append(elapsed)
            if record is not None:
                previous_step_traces.append(record)
            
            if args.early_stop and (it + 1) <= args.es_cutoff:
                # 직전까지 생성된 답변 중 동일 텍스트가 있었는지
                # (정확 일치 기준; 더 느슨한 기준이 필요하면 여기서 전처리/정규화를 추가)
                if new_text in answers[:-1]:
                    print(f"[early-stop] iter={it}, cutoff={args.es_cutoff}, "
                        f"reason=duplicate_before_cutoff")
                    # 필요하면 더 자세한 정보 남기기 가능
                    early_stopped_flag = True
                    break

            valid_mask = torch.ones_like(prev_hidden[..., :1])
            valid_mask[:, :input_ids.shape[1]] = 0
            token_cnt = valid_mask.sum(dim=1).clamp(min=1)
            seq_emb   = (prev_hidden * valid_mask).sum(dim=1) / token_cnt

            prev_prob= prev_prob[0][input_ids.shape[1]:]  

            previous_resps.append(new_text)
            previous_probs.append(prev_prob)
            previous_hidden.append(seq_emb)
            prev_probs=torch.stack(previous_probs, dim=0)
            prev_embs=torch.stack(previous_hidden, dim=0)
            # input()  # 필요 시만 사용

        metrics = evaluate_diversity(answers)
        deg_score = metrics.get("degeneration_llm_score")
        if deg_score is not None and deg_score > degen_threshold:
            print(f"[early-stop] reason=degeneration_score {deg_score:.3f} >= 0.9")
            early_stopped_flag = True

        result = {"prompt_index": idx, "prompt": prompt_text,
                  "answers": answers, "times": times,
                  "metrics": metrics, "early_stopped": early_stopped_flag}
        all_results.append(result)


        # ★ 결과 파일명: config와 같은 폴더 + 선택적 suffix
        base = os.path.splitext(args.config)[0]
        cfg_dir = os.path.dirname(base)

        if args.result_suffix is None:
            save_base = base
        else:
            save_base = os.path.join(cfg_dir, args.result_suffix)

        save_path = f"{save_base}_result.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({"config": cfg, "results": all_results}, f,
                      indent=2, ensure_ascii=False)
        if early_stopped_flag:
            print("[early-stop] stopping remaining prompts in this trial")
            break


    print(f"✅ All prompts processed. Results saved to {save_path}")

if __name__ == "__main__":
    main()