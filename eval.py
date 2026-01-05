from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# --- ADD: LLM JSON scorer (자체 모델/토크나이저 사용) ---
import re, json, math
from openai import OpenAI

client = OpenAI(
    api_key="",
)

def _llm_json_score(system_prompt: str, user_prompt: str, model_name="gpt-4.1-2025-04-14") -> dict | None:
    """
    system_prompt, user_prompt를 넣으면 LLM이 JSON만 반환하도록 유도하고 dict로 파싱해 돌려줌.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=512,
    )
    text = response.choices[0].message.content.strip()
    # JSON만 추출
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _score_degeneration_with_llm(answers, rubric=None):
    if rubric is None:
        rubric = (
            "You are a strict judge of text degeneration. "
            "Degeneration includes garbled tokens, random symbols, language mixing, "
            "nonsense, broken Unicode, repetitive babble, or clear loss of coherence.\n"
            "Don't judge the repetitiveness across answers, assess the individual quality and average them.\n"
            "Rate on 0.0~1.0: 0.0 clean/coherent, 1.0 severely degenerated.\n"
            "Consider ALL provided answers jointly and set the score to reflect the average observed degeneration.\n"
            'Return pure JSON: {"score": <float>, "reason": "<short>"}'
        )

    all_answers = "\n".join(f"[{i}] {a}" for i, a in enumerate(answers))
    user_prompt = (
        f"All answers (N={len(answers)}):\n{all_answers}\n\n"
        "Evaluate degeneration strictly."
    )

    js = _llm_json_score(rubric, user_prompt)
    if not js or "score" not in js:
        print("gpt parsing error occur-")
        print(js)
        return {"score": 0.0, "reason": "parse_fail_default_0.5", "raw": js}

    score = float(js.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return {"score": score, "reason": str(js.get("reason", "")), "raw": js}


sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_diversity(answer_list):
    bleu_scores = []
    rouge_l_scores = []
    meteor_scores = []
    cosine_sims = []
    cider_scores = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # 1. Pairwise 계산
    for i in range(len(answer_list)):
        for j in range(i + 1, len(answer_list)):
            ref, hyp = answer_list[i], answer_list[j]

            # BLEU
            bleu = sentence_bleu(
                [word_tokenize(ref)],
                word_tokenize(hyp),
                smoothing_function=SmoothingFunction().method1
            )
            bleu_scores.append(bleu)

            # ROUGE
            rouge_l = scorer.score(ref, hyp)["rougeL"].fmeasure
            rouge_l_scores.append(rouge_l)

            # METEOR
            meteor = meteor_score([word_tokenize(ref)], word_tokenize(hyp))
            meteor_scores.append(meteor)

            # Cosine similarity
            emb1 = sentence_model.encode(ref, convert_to_tensor=True)
            emb2 = sentence_model.encode(hyp, convert_to_tensor=True)
            cos_sim = cosine_similarity(
                emb1.unsqueeze(0).cpu().numpy(),
                emb2.unsqueeze(0).cpu().numpy()
            )[0][0]
            cosine_sims.append(cos_sim)

    # 2. BERTScore
    # P, R, F1 = bert_score(
    #     answer_list,
    #     [answer_list[0]] * len(answer_list),  # arbitrary ref (could also be mean across others)
    #     lang="en",
    #     verbose=False,
    #     rescale_with_baseline=True
    # )
    # avg_bertscore = float(F1.mean())

    # 3. CIDEr (based on COCO-style evaluation)
    # Prepare dummy COCO-style structure
    gts, res = {}, {}
    for idx, ans in enumerate(answer_list):
        gts[idx] = [{"caption": ref} for ref in answer_list if ref != ans]  # references
        res[idx] = [{"caption": ans}]  # hypothesis

    # Apply COCO tokenizer
    try:
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        cider = Cider()
        cider_scores_list, _ = cider.compute_score(gts, res)
        avg_cider = float(np.mean(cider_scores_list))
    except:
        avg_cider = -1

    degen = _score_degeneration_with_llm(answer_list)
    degeneration_llm_score = degen["score"]
    degeneration_llm_reason = degen["reason"]

    total_result = {
        "avg_bleu": float(np.mean(bleu_scores)),
        "avg_rougeL": float(np.mean(rouge_l_scores)),
        "avg_meteor": float(np.mean(meteor_scores)),
        "avg_cosine_similarity": float(np.mean(cosine_sims)),
        # "avg_bertscore_F1": avg_bertscore,
        "avg_cider": avg_cider,
        "degeneration_llm_score": degeneration_llm_score,
        "degeneration_llm_reason": degeneration_llm_reason,
    }
    print(total_result)
    return total_result
