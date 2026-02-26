from __future__ import annotations

"""Qwen RP용 GRPO 학습 스크립트.

설계 목표:
- 코드 가독성 우선
- 보상 함수 단순화
- 디버깅 용이성 확보

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run models/qwen3_core/grpo_trainer.py \
  --model_name models/qwen3_core/model_assets/qwen3_4b_rp \
  --train_data /mnt/d/rp_data/grpo/grpo_train.jsonl \
  --output_dir /tmp/grpo_dryrun \
  --max_completion_length 220 \
  --temperature 0.9 \
  --top_p 0.95 \
  --top_k 50 \
  --repetition_penalty 1.05 \
  --dry_run_compare \
  --dry_run_samples 8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run models/qwen3_core/grpo_trainer.py \
  --model_name models/qwen3_core/model_assets/qwen3_4b_rp \
  --train_data /mnt/d/rp_data/grpo/grpo_train.jsonl \
  --eval_data /mnt/d/rp_data/grpo/grpo_eval.jsonl \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_grpo \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-6 \
  --max_prompt_length 1024 \
  --max_completion_length 220 \
  --num_generations 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 40 \
  --repetition_penalty 1.1 \
  --gradient_checkpointing \
  --bf16 \
  --use_lora \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant \
  --bnb_4bit_compute_dtype bfloat16 \
  --reward_embedding_model data/embedding/BGE-m3-ko \
  --reward_embedding_batch_size 16 \
  --reward_embedding_max_length 256 \
  --debug_log_completions \
  --debug_log_every_calls 5 \
  --debug_log_num_samples 2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run models/qwen3_core/grpo_trainer.py \
  --model_name models/qwen3_core/model_assets/saya_rp_4b_sft \
  --train_data /mnt/d/rp_data/grpo/grpo2_train.jsonl \
  --eval_data /mnt/d/rp_data/grpo/grpo2_eval.jsonl \
  --output_dir models/qwen3_core/model_assets/saya_rp_4b_grpo \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-6 \
  --max_prompt_length 1024 \
  --max_completion_length 220 \
  --num_generations 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 40 \
  --repetition_penalty 1.1 \
  --gradient_checkpointing \
  --bf16 \
  --use_lora \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant \
  --bnb_4bit_compute_dtype bfloat16 \
  --reward_embedding_model data/embedding/BGE-m3-ko \
  --reward_embedding_batch_size 16 \
  --reward_embedding_max_length 256 \
  --debug_log_completions \
  --debug_log_every_calls 5 \
  --debug_log_num_samples 2
"""

import argparse
import re
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


PLAYER_NAMES = ("하야토", "카즈키", "소마")
ROLE_MARKERS = ("SYSTEM:", "USER:", "ASSISTANT:", "role:", "<|im_start|", "<|assistant|")

_EMBED_TOKENIZER: Any = None
_EMBED_MODEL: Any = None
_EMBED_DEVICE: str = "cpu"
_EMBED_BATCH_SIZE: int = 16
_EMBED_MAX_LENGTH: int = 256

_DEBUG_LOG_COMPLETIONS: bool = False
_DEBUG_LOG_EVERY_CALLS: int = 20
_DEBUG_LOG_NUM_SAMPLES: int = 2
_REWARD_CALL_COUNT: int = 0


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------
def _as_text(x: Any) -> str:
    """보상 함수 입력을 문자열로 정규화한다."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if isinstance(x.get("content"), str):
            return x["content"]
        return str(x)
    if isinstance(x, list):
        parts: List[str] = []
        for item in x:
            t = _as_text(item)
            if t:
                parts.append(t)
        return "\n".join(parts)
    return str(x)


def _prompt_to_role_text(prompt: Any) -> str:
    """프롬프트를 ROLE 접두어를 가진 평탄 텍스트로 변환한다."""
    if isinstance(prompt, list):
        out: List[str] = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            role = _as_text(m.get("role")).strip().upper()
            content = _as_text(m.get("content")).strip()
            if role and content:
                out.append(f"{role}: {content}")
        return "\n".join(out).strip()
    return _as_text(prompt)


def _extract_last_user_from_prompt(prompt: str) -> str:
    """평탄한 prompt에서 마지막 USER 블록을 추출한다."""
    if not prompt:
        return ""
    matches = re.findall(
        r"USER:\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)",
        prompt,
        flags=re.DOTALL,
    )
    return matches[-1].strip() if matches else ""


def _parse_prompt_protagonist_name(prompt: str) -> str:
    """prompt에서 주인공 이름을 추정한다."""
    if not prompt:
        return ""
    m = re.search(r"당신은(?:\s+이제)?\s+([가-힣A-Za-z0-9_]+)(?:다|입니다)\.?", prompt)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:주인공|assistant)\s*이름\s*[:：]\s*([가-힣A-Za-z0-9_]+)", prompt)
    if m:
        return m.group(1).strip()
    m = re.search(r"\n\s*이름\s*[:：]\s*([가-힣A-Za-z0-9_]+)", prompt)
    return m.group(1).strip() if m else ""


def _parse_prompt_speech_style(prompt: str) -> str:
    """prompt에서 목표 말투를 추정한다."""
    if not prompt:
        return ""
    if re.search(r"기본\s*말투\s*[:：]\s*존댓말|존댓말", prompt):
        return "formal"
    if re.search(r"기본\s*말투\s*[:：]\s*반말|반말", prompt):
        return "informal"
    return ""


def _normalize_completion_for_scoring(raw: str) -> Tuple[str, str, bool]:
    """completion을 채점용 2블록으로 정규화한다.

    Returns:
        narration: 서술 블록
        quote: 대사 블록(큰따옴표 포함)
        valid: 형식 유효 여부
    """
    txt = (raw or "").strip()
    if not txt:
        return "", "", False

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "", "", False

    narration = lines[0]
    quote = ""
    for ln in lines[1:]:
        if re.fullmatch(r'"[^"\n]{2,300}"', ln):
            quote = ln
            break

    if not quote:
        return "", "", False

    # 엄격한 형식: 유효 라인이 정확히 2줄이어야 한다.
    valid = len(lines) == 2
    return narration, quote, valid


def _has_quote_tail_violation(raw: str) -> bool:
    """둘째 줄 대사 닫힘 이후에 잔여 텍스트가 있으면 True를 반환한다."""
    txt = (raw or "").strip()
    if not txt:
        return False
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    second = lines[1]
    m = re.search(r'"[^"\n]{2,300}"', second)
    if not m:
        return False
    trailing = second[m.end() :].strip()
    return len(trailing) > 0


def _formal_count_in_quote(quote: str) -> int:
    """대사 블록에서 문장 끝 종결어미 기준 존댓말 출현 수를 계산한다."""
    sentence_chunks = [
        seg.strip()
        for seg in re.split(r"[.!?…\n]+", quote)
        if seg and seg.strip()
    ]
    endings: List[str] = []
    for seg in sentence_chunks:
        cleaned = re.sub(r'["\'“”‘’,\s]+$', "", seg)
        m = re.search(r"([가-힣]+)$", cleaned)
        if m:
            endings.append(m.group(1))

    formal_suffixes = (
        "습니다", "습니까", "세요", "해요", "이에요", "예요",
        "네요", "군요", "죠", "까요", "어요", "아요", "여요",
        "게요", "드립니다", "랍니다", "입니다",
    )
    return sum(1 for e in endings if e.endswith(formal_suffixes))


def _ngram_overlap_ratio(a: str, b: str, n: int = 3) -> float:
    """두 문자열의 단어 n-gram 중복 비율을 계산한다."""
    if not a or not b:
        return 0.0
    ta = re.findall(r"[가-힣A-Za-z0-9]+", a.lower())
    tb = re.findall(r"[가-힣A-Za-z0-9]+", b.lower())
    if len(ta) < n or len(tb) < n:
        return 0.0

    a_ngrams = {" ".join(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    b_ngrams = {" ".join(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    if not a_ngrams:
        return 0.0
    return len(a_ngrams & b_ngrams) / float(len(a_ngrams))


def _rough_token_len(text: str) -> int:
    """간이 토큰 길이를 계산한다."""
    if not text:
        return 0
    pieces = re.findall(r"[가-힣A-Za-z0-9]+|[^\s]", text)
    return len(pieces)


def _init_reward_embedder(model_name: str, device: str, batch_size: int, max_length: int) -> None:
    """임베딩 보상 인코더를 초기화한다."""
    global _EMBED_TOKENIZER, _EMBED_MODEL, _EMBED_DEVICE, _EMBED_BATCH_SIZE, _EMBED_MAX_LENGTH
    _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _EMBED_MODEL = AutoModel.from_pretrained(model_name)
    _EMBED_MODEL.eval()
    _EMBED_MODEL.to(device)
    _EMBED_DEVICE = device
    _EMBED_BATCH_SIZE = max(1, int(batch_size))
    _EMBED_MAX_LENGTH = max(32, int(max_length))


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """attention mask를 적용한 mean pooling을 계산한다."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _embed_texts(texts: List[str]) -> Optional[torch.Tensor]:
    """텍스트 리스트를 L2 정규화 임베딩 텐서로 변환한다."""
    if _EMBED_TOKENIZER is None or _EMBED_MODEL is None:
        return None
    if not texts:
        return None

    vectors: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            chunk = texts[i : i + _EMBED_BATCH_SIZE]
            encoded = _EMBED_TOKENIZER(
                chunk,
                padding=True,
                truncation=True,
                max_length=_EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(_EMBED_DEVICE) for k, v in encoded.items()}
            out = _EMBED_MODEL(**encoded)
            pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())

    return torch.cat(vectors, dim=0)


def _embedding_similarity_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
    """문자열 쌍 리스트의 임베딩 유사도를 배치로 계산한다."""
    if not pairs:
        return []
    for a, b in pairs:
        if not a or not b:
            raise RuntimeError("Embedding similarity pairs must not contain empty strings.")

    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    vecs = _embed_texts(left + right)
    if vecs is None:
        raise RuntimeError("Reward embedder is not initialized.")

    n = len(pairs)
    left_vecs = vecs[:n]
    right_vecs = vecs[n:]
    cos = torch.clamp((left_vecs * right_vecs).sum(dim=1), min=-1.0, max=1.0)
    sims = (cos + 1.0) / 2.0
    return [float(v.item()) for v in sims]


def _debug_log_completion_samples(completions: List[Any]) -> None:
    """디버그 모드에서 completion 샘플의 원문/정규화 결과를 출력한다."""
    global _REWARD_CALL_COUNT
    if not _DEBUG_LOG_COMPLETIONS:
        return

    _REWARD_CALL_COUNT += 1
    if _REWARD_CALL_COUNT % max(1, _DEBUG_LOG_EVERY_CALLS) != 0:
        return

    print(f"[GRPO_DEBUG] reward_call={_REWARD_CALL_COUNT} batch_size={len(completions)}")
    for i, comp in enumerate(completions[: max(1, _DEBUG_LOG_NUM_SAMPLES)]):
        raw = _as_text(comp)
        n1, q1, valid = _normalize_completion_for_scoring(raw)
        quote_count = raw.count('\"')
        print(f"[GRPO_DEBUG] sample={i} raw_chars={len(raw)} raw_quotes={quote_count}")
        print(f"[GRPO_DEBUG] sample={i} raw_head={raw[:160].replace(chr(10), '<NL>')}")
        print(f"[GRPO_DEBUG] sample={i} raw_tail={raw[-160:].replace(chr(10), '<NL>')}")
        print(f"[GRPO_DEBUG] sample={i} norm_valid={valid}")
        print(f"[GRPO_DEBUG] sample={i} norm_narr={n1}")
        print(f"[GRPO_DEBUG] sample={i} norm_quote={q1}")


# ------------------------------------------------------------
# 보상 함수
# ------------------------------------------------------------
def reward_format(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """형식 보상. 부분 점수 방식으로 서술/대사/2블록 준수를 평가한다."""
    _debug_log_completion_samples(completions)
    scores: List[float] = []
    for comp in completions:
        raw = _as_text(comp)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        _, quote, valid = _normalize_completion_for_scoring(raw)
        score = 0.0

        # 서술 1줄 존재
        if lines:
            first = lines[0]
            if first and not first.startswith('"') and "USER:" not in first.upper() and "ASSISTANT:" not in first.upper():
                score += 0.30

        # 큰따옴표 대사 1쌍 존재
        if quote:
            score += 0.40

        # 정확히 2블록 + role marker 없음
        has_role_marker = any(marker in raw.upper() for marker in ROLE_MARKERS)
        if valid and not has_role_marker:
            score += 0.30

        # 둘째 줄 닫힘 뒤 꼬리 생성은 강한 감점
        if _has_quote_tail_violation(raw):
            score -= 0.60

        scores.append(max(0.0, min(1.0, score)))
    return scores


def reward_role_split(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """역할 혼동 패턴에 패널티를 적용한다."""
    scores: List[float] = []
    player_re = "|".join(PLAYER_NAMES)

    for comp in completions:
        raw = _as_text(comp)
        upper = raw.upper()
        penalty = 0.0

        for marker in ROLE_MARKERS:
            if marker in upper:
                penalty += 0.45

        # 플레이어 이름 자체 언급은 허용한다.
        # 대신 플레이어를 화자로 세우는 전형 패턴만 패널티로 본다.
        if re.search(rf"(?:^|\n)\s*(?:{player_re})\s*[:：]", raw):
            penalty += 0.55
        if re.search(rf"(?:^|\n)\s*(?:{player_re}).{{0,16}}(?:말했|말한다|묻는다|대답했|속삭였|소리쳤)", raw):
            penalty += 0.45

        scores.append(max(0.0, 1.0 - penalty))

    return scores


def reward_length(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """출력 길이와 종료 안정성을 함께 평가한다."""
    scores: List[float] = []
    for comp in completions:
        raw = _as_text(comp)
        upper = raw.upper()
        n = _rough_token_len(raw)

        if 40 <= n <= 140:
            score = 1.0
        elif 20 <= n < 40:
            score = 0.70
        elif 140 < n <= 200:
            score = 0.70
        elif 200 < n <= 240:
            score = 0.35
        else:
            score = 0.10

        if "USER:" in upper or "ASSISTANT:" in upper or "SYSTEM:" in upper:
            score -= 0.40

        scores.append(max(0.0, min(1.0, score)))
    return scores


def reward_grounded_to_user(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """직전 user 발화와의 의미 정합성을 계산하고 과복창을 약감점한다."""
    scores: List[float] = [0.0 for _ in completions]
    valid_meta: List[Tuple[int, str, str, bool]] = []

    for idx, (prompt, comp) in enumerate(zip(prompts, completions)):
        prompt_txt = _prompt_to_role_text(prompt)
        user_txt = _extract_last_user_from_prompt(prompt_txt)
        raw = _as_text(comp)
        narration, quote, valid = _normalize_completion_for_scoring(raw)
        if not user_txt or not valid:
            continue
        out_txt = f"{narration} {quote}"
        valid_meta.append((idx, user_txt, out_txt, "?" in user_txt))

    if not valid_meta:
        return scores

    sims = _embedding_similarity_pairs([(u, o) for _, u, o, _ in valid_meta])

    for (idx, user_txt, out_txt, is_question), sim in zip(valid_meta, sims):
        if is_question and sim < 0.10:
            scores[idx] = 0.0
            continue

        if sim >= 0.50:
            score = 1.0
        elif sim >= 0.35:
            score = 0.75
        elif sim >= 0.20:
            score = 0.45
        elif sim >= 0.10:
            score = 0.20
        else:
            score = 0.05

        overlap = _ngram_overlap_ratio(user_txt, out_txt, n=3)
        if overlap > 0.70:
            score -= 0.10
        elif overlap > 0.55:
            score -= 0.05

        scores[idx] = max(0.0, score)

    return scores


def reward_character_consistency(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """주인공 이름/말투 일관성 점수를 계산한다."""
    scores: List[float] = []

    for prompt, comp in zip(prompts, completions):
        prompt_txt = _prompt_to_role_text(prompt)
        raw = _as_text(comp)
        narration, quote, valid = _normalize_completion_for_scoring(raw)
        if not valid:
            scores.append(0.0)
            continue

        protagonist = _parse_prompt_protagonist_name(prompt_txt)
        style = _parse_prompt_speech_style(prompt_txt)

        name_ok = (not protagonist) or (protagonist in narration)
        formal_count = _formal_count_in_quote(quote)

        style_ok = True
        if style == "formal":
            style_ok = formal_count >= 1
        elif style == "informal":
            style_ok = formal_count == 0

        score = 1.0
        if not name_ok:
            score -= 0.35
        if not style_ok:
            score -= 0.45

        scores.append(max(0.0, min(1.0, score)))

    return scores


def reward_reference_alignment(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """reference와의 의미 정렬 점수를 계산한다."""
    refs = kwargs.get("reference")
    if refs is None:
        refs = kwargs.get("references")

    scores: List[float] = [0.5 for _ in completions]
    valid_meta: List[Tuple[int, str, str]] = []

    for idx, comp in enumerate(completions):
        raw = _as_text(comp)
        narration, quote, valid = _normalize_completion_for_scoring(raw)
        if not valid:
            scores[idx] = 0.0
            continue
        out_txt = f"{narration} {quote}"

        ref_txt = ""
        if isinstance(refs, list) and idx < len(refs):
            ref_txt = _as_text(refs[idx]).strip()
        elif isinstance(refs, str):
            ref_txt = refs.strip()

        if not ref_txt:
            continue
        valid_meta.append((idx, out_txt, ref_txt))

    if not valid_meta:
        return scores

    sims = _embedding_similarity_pairs([(o, r) for _, o, r in valid_meta])

    for (idx, _, _), sim in zip(valid_meta, sims):
        if sim >= 0.55:
            scores[idx] = 1.0
        elif sim >= 0.40:
            scores[idx] = 0.8
        elif sim >= 0.25:
            scores[idx] = 0.55
        elif sim >= 0.12:
            scores[idx] = 0.30
        else:
            scores[idx] = 0.10

    return scores


# ------------------------------------------------------------
# 데이터/모델 빌더
# ------------------------------------------------------------
def _messages_to_prompt(messages: Any) -> str:
    """chat 메시지 리스트를 평탄한 prompt 문자열로 변환한다."""
    if not isinstance(messages, list):
        return ""

    out: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().upper()
        content = str(m.get("content", "")).strip()
        if role and content:
            out.append(f"{role}: {content}")

    return "\n".join(out).strip()


def _ensure_prompt_column(ds: Dataset) -> Dataset:
    """데이터셋에 prompt 컬럼이 없으면 prompt_messages에서 생성한다."""
    names = set(ds.column_names)
    if "prompt" in names:
        return ds
    if "prompt_messages" in names:
        return ds.map(
            lambda x: {"prompt": _messages_to_prompt(x.get("prompt_messages"))},
            desc="build prompt from prompt_messages",
        )
    raise ValueError("Dataset must contain 'prompt' or 'prompt_messages'.")


def _drop_empty_prompt(ds: Dataset) -> Dataset:
    """빈 prompt 행을 제거한다."""
    def _ok_prompt(x: Any) -> bool:
        p = x.get("prompt")
        if isinstance(p, str):
            return len(p.strip()) > 0
        if isinstance(p, list):
            return len(p) > 0
        return False

    return ds.filter(
        _ok_prompt,
        desc="drop empty prompt rows",
    )


def load_grpo_dataset(path: str) -> Dataset:
    """JSONL을 로드하고 GRPO 입력 형식으로 정규화한다."""
    ds = load_dataset("json", data_files=path, split="train")
    ds = _ensure_prompt_column(ds)
    if "prompt_messages" in set(ds.column_names):
        ds = ds.map(
            lambda x: {"prompt": x["prompt_messages"]} if isinstance(x.get("prompt_messages"), list) else {"prompt": x.get("prompt")},
            desc="prefer prompt_messages as prompt",
        )
    ds = _drop_empty_prompt(ds)
    return ds


def _prepare_generation_inputs_from_row(row: Dict[str, Any]) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    """데이터셋 행에서 평탄 프롬프트와 chat 메시지 프롬프트를 추출한다."""
    flat_prompt = _prompt_to_role_text(row.get("prompt")).strip()

    prompt_messages: Optional[List[Dict[str, str]]] = None
    pm = row.get("prompt_messages")
    if not isinstance(pm, list) and isinstance(row.get("prompt"), list):
        pm = row.get("prompt")
    if isinstance(pm, list):
        cleaned: List[Dict[str, str]] = []
        for m in pm:
            if not isinstance(m, dict):
                continue
            role = _as_text(m.get("role")).strip()
            content = _as_text(m.get("content")).strip()
            if role and content:
                cleaned.append({"role": role, "content": content})
        if cleaned:
            prompt_messages = cleaned

    return flat_prompt, prompt_messages


def _generate_with_flat_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """평탄 문자열 프롬프트를 바로 넣어 생성한다."""
    if not prompt:
        return ""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "renormalize_logits": True,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _generate_with_chat_template(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """chat_template 기반 프롬프트로 생성한다."""
    if not messages:
        return ""

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "renormalize_logits": True,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_dry_generation_compare(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    sample_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> None:
    """훈련 없이 생성 경로 차이를 출력해 진단한다."""
    n = min(len(dataset), max(1, sample_count))
    print(f"[DRY_RUN] start compare samples={n}")

    for i in range(n):
        row = dataset[i]
        flat_prompt, prompt_messages = _prepare_generation_inputs_from_row(row)
        ref = _as_text(row.get("reference")).strip()
        user_tail = _extract_last_user_from_prompt(flat_prompt)

        print("\n" + "=" * 80)
        print(f"[DRY_RUN] sample={i}")
        print(f"[DRY_RUN] user_tail={user_tail[:180]}")
        print(f"[DRY_RUN] reference={ref[:180]}")

        flat_out = _generate_with_flat_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=flat_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        n1, q1, v1 = _normalize_completion_for_scoring(flat_out)
        print(f"[DRY_RUN][flat] chars={len(flat_out)} quotes={flat_out.count(chr(34))} valid={v1}")
        print(f"[DRY_RUN][flat] narr={n1}")
        print(f"[DRY_RUN][flat] quote={q1}")
        print(f"[DRY_RUN][flat] head={flat_out[:220].replace(chr(10), '<NL>')}")

        if prompt_messages:
            chat_out = _generate_with_chat_template(
                model=model,
                tokenizer=tokenizer,
                messages=prompt_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            n2, q2, v2 = _normalize_completion_for_scoring(chat_out)
            print(f"[DRY_RUN][chat] chars={len(chat_out)} quotes={chat_out.count(chr(34))} valid={v2}")
            print(f"[DRY_RUN][chat] narr={n2}")
            print(f"[DRY_RUN][chat] quote={q2}")
            print(f"[DRY_RUN][chat] head={chat_out[:220].replace(chr(10), '<NL>')}")
        else:
            print("[DRY_RUN][chat] prompt_messages not found in this row")

    print("[DRY_RUN] done")


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Qwen 계열 모듈에 맞는 LoRA 설정을 생성한다."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def resolve_dtype(name: str) -> torch.dtype:
    """문자열 dtype을 torch dtype으로 변환한다."""
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="GRPO trainer for merged Qwen RP model.")

    parser.add_argument("--model_name", type=str, default="models/qwen3_core/model_assets/qwen3_4b_rp")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=220)
    parser.add_argument("--num_generations", type=int, default=4)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

    parser.add_argument("--reward_embedding_model", type=str, default="data/embedding/BGE-m3-ko")
    parser.add_argument("--reward_embedding_batch_size", type=int, default=16)
    parser.add_argument("--reward_embedding_max_length", type=int, default=256)

    parser.add_argument("--debug_log_completions", action="store_true")
    parser.add_argument("--debug_log_every_calls", type=int, default=20)
    parser.add_argument("--debug_log_num_samples", type=int, default=2)
    parser.add_argument("--dry_run_compare", action="store_true")
    parser.add_argument("--dry_run_samples", type=int, default=8)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )

    return parser.parse_args()


def main() -> None:
    """GRPO 학습을 실행한다."""
    args = parse_args()

    global _DEBUG_LOG_COMPLETIONS, _DEBUG_LOG_EVERY_CALLS, _DEBUG_LOG_NUM_SAMPLES
    _DEBUG_LOG_COMPLETIONS = bool(args.debug_log_completions)
    _DEBUG_LOG_EVERY_CALLS = max(1, int(args.debug_log_every_calls))
    _DEBUG_LOG_NUM_SAMPLES = max(1, int(args.debug_log_num_samples))

    if args.load_in_4bit and not args.use_lora:
        raise ValueError("QLoRA mode requires --use_lora.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        fix_mistral_regex=True,
    )
    # 디코더 전용 모델의 배치 생성 안정성을 위해 좌측 패딩을 사용한다.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_grpo_dataset(args.train_data)
    eval_ds = load_grpo_dataset(args.eval_data) if args.eval_data else None

    grpo_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if eval_ds is not None else "no",
        "eval_steps": args.eval_steps if eval_ds is not None else None,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
        "reward_weights": [0.28, 0.22, 0.12, 0.12, 0.18, 0.08],
    }
    supported = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    dropped = sorted(k for k in grpo_kwargs if k not in supported)
    if dropped:
        print(f"[warn] Unsupported GRPOConfig kwargs in current trl: {dropped}")
    grpo_args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in supported})

    peft_cfg = build_lora_config(args) if args.use_lora else None
    model_arg: Any = args.model_name

    if args.load_in_4bit:
        compute_dtype = resolve_dtype(args.bnb_4bit_compute_dtype)
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=compute_dtype,
            quantization_config=quant_cfg,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
        model_arg = model

    if args.dry_run_compare:
        dry_model = model_arg
        if isinstance(dry_model, str):
            dry_model = AutoModelForCausalLM.from_pretrained(
                dry_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if args.bf16 else None,
                attn_implementation="sdpa",
            )
        dry_model.eval()
        run_dry_generation_compare(
            model=dry_model,
            tokenizer=tokenizer,
            dataset=train_ds,
            sample_count=args.dry_run_samples,
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        return

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    _init_reward_embedder(
        model_name=args.reward_embedding_model,
        device=embed_device,
        batch_size=args.reward_embedding_batch_size,
        max_length=args.reward_embedding_max_length,
    )

    reward_funcs = [
        reward_format,
        reward_role_split,
        reward_grounded_to_user,
        reward_character_consistency,
        reward_reference_alignment,
        reward_length,
    ]

    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
