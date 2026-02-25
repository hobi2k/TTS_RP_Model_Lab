#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""v6 GPT 멀티턴 백엔드 호출 모듈.

이 모듈은 OpenAI Responses API 호출과 JSON 추출을 담당한다.
생성 파라미터 정책은 한 곳에서만 관리해 `gen`/`validators` 중복을 제거한다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)


def _collect_text_fields(obj: Any, out: List[str]) -> None:
    """응답 객체/딕셔너리를 재귀 순회하며 텍스트 필드를 수집한다."""
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
        return
    if isinstance(obj, list):
        for x in obj:
            _collect_text_fields(x, out)
        return
    if isinstance(obj, dict):
        for k in ("output_text", "text", "content"):
            if k in obj:
                _collect_text_fields(obj.get(k), out)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _collect_text_fields(v, out)
        return
    # pydantic model/object fallback
    for attr in ("output_text", "text", "content"):
        if hasattr(obj, attr):
            _collect_text_fields(getattr(obj, attr), out)


def _extract_response_text(res: Any) -> str:
    """Responses 응답 객체에서 텍스트를 안전하게 추출한다."""
    txt = getattr(res, "output_text", None)
    if txt:
        return txt.strip()

    chunks: List[str] = []
    for item in getattr(res, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                chunks.append(t)
    if chunks:
        return "\n".join(chunks).strip()

    # SDK 타입 변화 대응: model_dump 기반 재귀 수집
    try:
        dumped = res.model_dump()
    except Exception:
        dumped = None
    more: List[str] = []
    _collect_text_fields(dumped, more)
    return "\n".join(more).strip()


def _supports_sampling_controls(model_name: Optional[str]) -> bool:
    """모델이 `temperature/top_p` 샘플링 제어를 지원하는지 반환한다."""
    m = (model_name or "").strip().lower()
    if not m:
        return False
    if m.startswith("gpt-5"):
        return False
    return True


def generate_text(
    *,
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
) -> str:
    """OpenAI API로 텍스트를 생성해 문자열로 반환한다."""
    _ = repetition_penalty
    if not do_sample:
        temperature = 0.0
        top_p = 1.0

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "input": messages,
    }
    is_gpt5 = (model_name or "").strip().lower().startswith("gpt-5")
    if is_gpt5:
        # reasoning 토큰이 출력 토큰을 잠식해 empty가 나는 케이스를 줄인다.
        kwargs["reasoning"] = {"effort": "minimal"}
        kwargs["text"] = {"verbosity": "low"}
    elif _supports_sampling_controls(model_name):
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)

    token_budget = int(max_new_tokens)
    if is_gpt5:
        # gpt-5는 reasoning을 내부적으로 사용하므로 출력 예산을 보수적으로 상향한다.
        token_budget = max(token_budget + 128, 320)

    try:
        res = client.responses.create(**kwargs, max_output_tokens=token_budget)
    except TypeError:
        res = client.responses.create(**kwargs, max_tokens=token_budget)

    text = _extract_response_text(res)
    if text:
        return text

    # 1차 결과가 empty면 reasoning 없이 한 번 더 시도한다.
    if is_gpt5:
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("reasoning", None)
        retry_kwargs["text"] = {"verbosity": "low"}
        retry_budget = max(token_budget, 512)
        try:
            res2 = client.responses.create(**retry_kwargs, max_output_tokens=retry_budget)
        except TypeError:
            res2 = client.responses.create(**retry_kwargs, max_tokens=retry_budget)
        return _extract_response_text(res2)
    return text


def generate_json(
    *,
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 1.0,
    do_sample: bool = True,
    log_raw: bool = False,
) -> Optional[Dict[str, Any]]:
    """텍스트 생성 결과에서 첫 번째 JSON 객체를 파싱해 반환한다."""
    raw = generate_text(
        client=client,
        model_name=model_name,
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
