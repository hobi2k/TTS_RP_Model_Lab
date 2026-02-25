"""
singleturn messages JSONL을 현재 RP 포맷 기준으로 정규화한다.

규칙:
- {{user}} -> [하야토, 소마, 카즈키] 중 하나로 샘플 단위 치환
- 모든 '*' 문자 제거
- assistant 대사 라벨 제거 (예: '네뷸라: ...')
- assistant 대사는 큰따옴표로 감싸 보정

예시:
uv run data/v7_singleturn_normalize.py \
  --in_jsonl /mnt/d/rp_data/backup/final_ko/rp_generated_cleaned.jsonl \
  --out_jsonl /mnt/d/rp_data/v7/rp_single_norm.jsonl \
  --seed 42

uv run data/singleturn/singleturn_message_converter.py \
  --in_path /mnt/d/rp_data/singleturn/rp_generated_local.jsonl \
  --out_path /mnt/d/rp_data/singleturn/rp_generated_local_cleaned.jsonl
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List


PLAYER_NAMES = ["하야토", "소마", "카즈키"]


RE_SPEAKER_LABEL = re.compile(r'^\s*"?(?P<name>[^"\n:]{1,20})\s*:\s*')


def strip_stars(text: str) -> str:
    """문자열의 '*'를 전부 제거한다."""
    return text.replace("*", "")


def quote_if_needed(text: str) -> str:
    """대사가 따옴표로 감싸지지 않은 경우 큰따옴표를 붙인다."""
    t = text.strip()
    if not t:
        return '""'
    if t.startswith('"') and t.endswith('"'):
        return t
    return f'"{t}"'


def normalize_assistant_content(content: str) -> str:
    """assistant content를 '서술 1줄 + 대사 1줄' 형태로 보정한다.

    - 화자 라벨 제거
    - 대사 줄 큰따옴표 보정
    - '*' 제거
    """
    raw = strip_stars(content or "").strip()
    if not raw:
        return raw

    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if not lines:
        return raw

    # 첫 줄만 있는 경우: 화자 라벨 제거 후 대사로 간주
    if len(lines) == 1:
        one = RE_SPEAKER_LABEL.sub("", lines[0]).strip()
        return quote_if_needed(one)

    # 여러 줄인 경우 마지막 줄을 대사로 보정
    narr = lines[0]
    dia = lines[-1]
    dia = RE_SPEAKER_LABEL.sub("", dia).strip()
    dia = quote_if_needed(dia)
    return f"{narr}\n{dia}"


def normalize_record(rec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """한 레코드를 정규화한다."""
    msgs = rec.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("messages 필드가 없거나 list가 아님")

    player = rng.choice(PLAYER_NAMES)
    out_msgs: List[Dict[str, str]] = []

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # 공통 치환
        content = content.replace("{{user}}", player)
        content = strip_stars(content)

        if role == "assistant":
            content = normalize_assistant_content(content)

        out_msgs.append({"role": role, "content": content})

    return {"messages": out_msgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="입력 JSONL")
    ap.add_argument("--out_jsonl", required=True, help="출력 JSONL")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total = 0
    written = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                rec = json.loads(s)
                out = normalize_record(rec, rng)
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                raise RuntimeError(f"{line_no}번째 줄 처리 실패: {e}")

    print("DONE")
    print(f"- total rows: {total}")
    print(f"- written rows: {written}")
    print(f"- out: {out_path}")


if __name__ == "__main__":
    main()
