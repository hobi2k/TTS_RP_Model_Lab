"""v6 Qwen 멀티턴 백엔드 호출 모듈.

이 모듈은 로컬 Qwen 모델의 생성 호출과 JSON 추출을 담당한다.
생성 파라미터 정책을 단일 위치로 모아 `gen`/`validators` 중복을 제거한다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch


RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)


def generate_text(
    *,
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    no_repeat_ngram_size: int = 0,
) -> str:
    """로컬 Qwen 모델로 채팅 텍스트를 생성한다.

    Args:
        model: 로컬 추론에 사용할 CausalLM 모델 객체.
        tokenizer: 채팅 템플릿 적용 및 토크나이즈에 사용할 토크나이저.
        messages: OpenAI chat 형식 메시지 목록.
        max_new_tokens: 최대 생성 토큰 수.
        temperature: 샘플링 온도.
        top_p: 누적 확률 샘플링 상한.
        repetition_penalty: 반복 억제 강도.
        do_sample: 샘플링 사용 여부.
        no_repeat_ngram_size: 동일 n-gram 반복 금지 크기. 0이면 비활성.

    Returns:
        생성된 assistant 문자열.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    # 생성 단계에서 동일 구문 반복을 줄여 기계적 에코를 완화한다.
    if int(no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_json(
    *,
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 1.0,
    do_sample: bool = True,
    log_raw: bool = False,
) -> Optional[Dict[str, Any]]:
    """텍스트 생성 결과에서 첫 번째 JSON 객체를 파싱해 반환한다."""
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.0,
        do_sample=do_sample,
    )
    if log_raw:
        print(f"[EVAL_RAW] {raw}", flush=True)

    matches = list(RE_JSON.finditer(raw))
    if not matches:
        return None

    for m in matches:
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            blob2 = re.sub(r",\s*}", "}", blob)
            blob2 = re.sub(r",\s*]", "]", blob2)
            try:
                return json.loads(blob2)
            except Exception:
                continue
    return None
