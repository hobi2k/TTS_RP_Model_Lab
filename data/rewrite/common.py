from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each JSONL row must be an object at {path}:{lineno}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_messages_sample(sample: dict[str, Any]) -> bool:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
    return True


def build_assistant_rewrite_instruction() -> str:
    return (
        "너는 한국어 RP 데이터셋 리라이터다.\n"
        "assistant 답변만 리라이트한다.\n"
        "최우선 기준은 system(시나리오북) 규칙 준수다.\n"
        "assistant의 말투/성격/형식은 반드시 시나리오북(system)에 맞춰라.\n"
        "3인칭 서술 + 1인칭 큰따옴표 대사 형식을 반드시 유지한다.\n"
        "서술 파트는 반드시 3인칭 평어체를 유지한다.\n"
        "대사 파트의 말투는 반드시 시나리오북에 정의된 주인공 말투를 따른다.\n"
        "의미, 장면, 관계는 유지하되 문장 자연스러움과 몰입감만 개선하라.\n"
        "- 시나리오북에 출력 형식이 있으면 반드시 따른다.\n"
        "- 플레이어(user) 대사/행동을 작성하지 않는다.\n"
        "- 메타 설명, 주석, 해설, 마크다운 금지.\n"
        "반드시 리라이트된 assistant 본문만 출력하라."
    )


def build_user_rewrite_instruction() -> str:
    return (
        "너는 한국어 RP 데이터셋 리라이터다.\n"
        "user 입력만 리라이트한다.\n"
        "핵심 의미/의도/정보는 유지하되 한국어 자연스러움만 개선하라.\n"
        "원문이 3인칭 서술 + 1인칭 큰따옴표 대사 형식이면 반드시 유지하라.\n"
        "서술 파트는 반드시 평어체를 유지하라.\n"
        "대사 파트의 말투(존댓말/반말)는 원문 대사를 유지하라.\n"
        "- 메타 설명, 주석, 해설, 마크다운 금지.\n"
        "반드시 리라이트된 user 본문만 출력하라."
    )


def split_narration_dialogue(text: str) -> tuple[str, str] | None:
    if not text or not text.strip():
        return None
    src = text.strip()
    m = re.search(r'"([^"\n]{1,1200})"', src)
    if not m:
        return None
    narration = src[: m.start()].strip()
    dialogue = m.group(1).strip()
    if not narration or not dialogue:
        return None
    return narration, dialogue


def has_rp_narration_dialogue_format(text: str) -> bool:
    """서술 1블록 + 큰따옴표 대사 1블록 최소 형식 검사."""
    return split_narration_dialogue(text) is not None


def normalize_rp_linebreak_spacing(text: str) -> str:
    """
    서술+대사 형식에서 개행 직전 공백을 제거한다.
    예) '서술이다.  \\n"대사"' -> '서술이다.\\n"대사"'
    """
    if not text:
        return ""
    return re.sub(r"[ \t]+\n", "\n", text)


def _is_polite_ending_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if not s:
        return False
    return bool(
        re.search(
            r"(습니다|습니까|어요|아요|해요|예요|이에요|네요|군요|세요|죠)[.!?…\"\']?$",
            s,
        )
    )


def is_plain_style_text(text: str) -> bool:
    """평어체(해요체/하십시오체 아님) 간이 판정."""
    if not text or not text.strip():
        return False
    chunks = re.split(r"[\n\r]+|(?<=[.!?…])\s+", text.strip())
    chunks = [c.strip() for c in chunks if c.strip()]
    if not chunks:
        return False
    return not any(_is_polite_ending_sentence(c) for c in chunks)


def detect_speech_level(text: str) -> str:
    """존댓말/반말 간이 판정."""
    if not text or not text.strip():
        return "banmal"
    polite_hits = len(
        re.findall(
            r"(습니다|습니까|어요|아요|해요|예요|이에요|네요|군요|세요|죠)(?=[.!?…\"\']?$|\s)",
            text,
        )
    )
    return "jondaetmal" if polite_hits > 0 else "banmal"


def _same_speech_level(a: str, b: str) -> bool:
    return detect_speech_level(a) == detect_speech_level(b)


def enforce_assistant_rp_format(original: str, rewritten: str) -> str:
    """
    assistant 리라이트 형식/말투 검증:
    - 원문이 서술+대사면 리라이트도 유지
    - 서술은 평어체 유지
    위반 시 원문 반환
    """
    rewritten = normalize_rp_linebreak_spacing(rewritten.strip())
    original = original.strip()
    original_split = split_narration_dialogue(original)
    rewritten_split = split_narration_dialogue(rewritten)
    if original_split is not None:
        if rewritten_split is None:
            return original
        rewritten_narration, _ = rewritten_split
        if not is_plain_style_text(rewritten_narration):
            return original
    return normalize_rp_linebreak_spacing(rewritten)


def enforce_user_rewrite_format(original: str, rewritten: str) -> str:
    """
    user 리라이트 형식/말투 검증:
    - 원문이 서술+대사면 리라이트도 동일 형식 유지
    - 서술은 평어체
    - 대사 말투(존댓말/반말)는 원문 대사와 일치
    위반 시 원문 반환
    """
    rewritten = normalize_rp_linebreak_spacing(rewritten.strip())
    original = original.strip()
    original_split = split_narration_dialogue(original)
    rewritten_split = split_narration_dialogue(rewritten)

    if original_split is not None:
        if rewritten_split is None:
            return original
        _, original_dialogue = original_split
        rewritten_narration, _ = rewritten_split
        if not is_plain_style_text(rewritten_narration):
            return original
        _, rewritten_dialogue = rewritten_split
        if not _same_speech_level(original_dialogue, rewritten_dialogue):
            return original
        return normalize_rp_linebreak_spacing(rewritten)

    if not _same_speech_level(original, rewritten):
        return original
    return normalize_rp_linebreak_spacing(rewritten)
