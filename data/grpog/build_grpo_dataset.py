"""
uv run data/grpog/build_grpo_dataset.py \
  --inputs /mnt/d/rp_data/v7/rp_datum_unite_cleaned.jsonl \
  --out_train /mnt/d/rp_data/grpo/grpo_train.jsonl \
  --out_eval /mnt/d/rp_data/grpo/grpo_eval.jsonl \
  --eval_ratio 0.05 \
  --seed 42 \
  --max_context_messages 32 \
  --min_prompt_chars 8 \
  --min_reference_chars 4 \
  --tokenizer_path models/qwen3_core/model_assets/qwen3_4b_rp \
  --max_prompt_tokens 1024 \
  --max_reference_tokens 220
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


ROLE_SET = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    """
    GRPO 데이터셋 생성용 CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 입력/출력 경로와 필터 옵션이 담긴 객체.
    """
    parser = argparse.ArgumentParser(
        description="Build GRPO prompt/reference datasets from RP jsonl sources."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input jsonl files. Supports chat-style messages and legacy singleturn rows.",
    )
    parser.add_argument(
        "--out_train",
        required=True,
        help="Output train jsonl path.",
    )
    parser.add_argument(
        "--out_eval",
        required=True,
        help="Output eval jsonl path.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Evaluation split ratio in [0,1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_context_messages",
        type=int,
        default=12,
        help="Keep only the last N prompt messages.",
    )
    parser.add_argument(
        "--min_prompt_chars",
        type=int,
        default=8,
        help="Drop samples with very short prompt text.",
    )
    parser.add_argument(
        "--min_reference_chars",
        type=int,
        default=4,
        help="Drop samples with very short assistant reference.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Tokenizer path for token-length filtering.",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=0,
        help="Drop rows if prompt token length exceeds this. 0 disables token filtering.",
    )
    parser.add_argument(
        "--max_reference_tokens",
        type=int,
        default=0,
        help="Drop rows if reference token length exceeds this. 0 disables token filtering.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    JSONL 파일을 한 줄씩 읽어 dict row를 순회한다.

    유효한 JSON 객체(dict)만 내보내며, 소스 추적을 위해
    `_source_path`, `_source_line` 메타 정보를 주입한다.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                row["_source_path"] = str(path)
                row["_source_line"] = line_no
                yield row


def normalize_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    다양한 입력 row를 표준 messages 형식으로 정규화한다.

    지원 형식:
    1) chat 형식: {"messages":[{"role":"...","content":"..."}]}
    2) 레거시 형식: {"system":"...","user":"...","assistant":"..."}
    """
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        out: List[Dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            role = role.strip().lower()
            content = content.strip()
            if role not in ROLE_SET or not content:
                continue
            out.append({"role": role, "content": content})
        if out:
            return out

    # 레거시 싱글턴 형식 지원
    system = row.get("system")
    user = row.get("user")
    assistant = row.get("assistant")
    if all(isinstance(x, str) for x in (system, user, assistant)):
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]

    return None


def build_prompt_text(prompt_messages: List[Dict[str, str]]) -> str:
    """
    prompt_messages를 단일 텍스트 prompt로 평탄화한다.

    예:
    SYSTEM: ...
    USER: ...
    """
    lines: List[str] = []
    for m in prompt_messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines).strip()


def extract_samples_from_messages(
    messages: List[Dict[str, str]],
    max_context_messages: int,
    min_prompt_chars: int,
    min_reference_chars: int,
    tokenizer: Optional[Any],
    max_prompt_tokens: int,
    max_reference_tokens: int,
    source_path: str,
    source_line: int,
    item_offset: int,
) -> List[Dict[str, Any]]:
    """
    단일 대화(messages)에서 assistant 턴 기준 GRPO 샘플을 추출한다.

    추출 규칙:
    - 각 assistant 턴을 하나의 학습 타깃(reference)으로 사용
    - prompt는 해당 assistant 직전까지의 문맥 사용
    - prompt 마지막 역할은 반드시 user여야 함
    - 문자 길이/토큰 길이 필터를 통과한 샘플만 유지
    """
    samples: List[Dict[str, Any]] = []
    sample_local_idx = 0

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        prompt_messages = messages[:i]
        if not prompt_messages:
            continue
        if not any(m["role"] == "user" for m in prompt_messages):
            continue
        # GRPO prompt는 마지막 턴이 user일 때 응답 최적화가 안정적이다.
        if prompt_messages[-1]["role"] != "user":
            continue

        reference = msg["content"].strip()
        if len(reference) < min_reference_chars:
            continue

        if max_context_messages > 0 and len(prompt_messages) > max_context_messages:
            prompt_messages = prompt_messages[-max_context_messages:]

        prompt_text = build_prompt_text(prompt_messages)
        if len(prompt_text) < min_prompt_chars:
            continue
        if tokenizer is not None:
            # 토큰 길이 상한을 넘기면 잘라내지 않고 drop한다.
            if max_prompt_tokens > 0:
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                if len(prompt_ids) > max_prompt_tokens:
                    continue
            if max_reference_tokens > 0:
                ref_ids = tokenizer.encode(reference, add_special_tokens=False)
                if len(ref_ids) > max_reference_tokens:
                    continue

        sample_id = f"{Path(source_path).name}:{source_line}:{item_offset}:{sample_local_idx}"
        samples.append(
            {
                "id": sample_id,
                "prompt": prompt_text,
                "prompt_messages": prompt_messages,
                "reference": reference,
                "meta": {
                    "source_path": source_path,
                    "source_line": source_line,
                    "assistant_turn_index": i,
                },
            }
        )
        sample_local_idx += 1

    return samples


def build_dataset(
    input_paths: List[Path],
    max_context_messages: int,
    min_prompt_chars: int,
    min_reference_chars: int,
    tokenizer: Optional[Any],
    max_prompt_tokens: int,
    max_reference_tokens: int,
) -> List[Dict[str, Any]]:
    """
    여러 입력 파일에서 GRPO 샘플을 모아 단일 리스트로 만든다.

    파일 단위 -> row 단위 -> assistant 턴 단위로 샘플을 누적한다.
    """
    all_samples: List[Dict[str, Any]] = []
    for path in input_paths:
        item_offset = 0
        for row in iter_jsonl(path):
            messages = normalize_messages(row)
            if not messages:
                item_offset += 1
                continue
            samples = extract_samples_from_messages(
                messages=messages,
                max_context_messages=max_context_messages,
                min_prompt_chars=min_prompt_chars,
                min_reference_chars=min_reference_chars,
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                max_reference_tokens=max_reference_tokens,
                source_path=row["_source_path"],
                source_line=row["_source_line"],
                item_offset=item_offset,
            )
            all_samples.extend(samples)
            item_offset += 1
    return all_samples


def split_dataset(
    samples: List[Dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    전체 샘플을 train/eval로 랜덤 분할한다.

    Args:
        samples: 전체 샘플 목록
        eval_ratio: eval 비율
        seed: 셔플 시드
    """
    rnd = random.Random(seed)
    rnd.shuffle(samples)

    if eval_ratio <= 0.0:
        return samples, []

    eval_size = int(len(samples) * eval_ratio)
    if eval_size <= 0:
        eval_size = 1 if len(samples) > 1 else 0
    eval_set = samples[:eval_size]
    train_set = samples[eval_size:]
    return train_set, eval_set


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """행 목록을 JSONL 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """
    GRPO 데이터셋 생성 진입점.

    처리 흐름:
    1) 인자 파싱 및 입력 경로 검증
    2) 필요 시 tokenizer 로드(토큰 길이 필터용)
    3) 샘플 생성/필터링
    4) train/eval 분할
    5) JSONL 저장 및 통계 출력
    """
    args = parse_args()
    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be in [0, 1).")

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    tokenizer = None
    if args.max_prompt_tokens > 0 or args.max_reference_tokens > 0:
        # 토큰 길이 필터를 사용할 때만 tokenizer를 로드한다.
        if not args.tokenizer_path:
            raise ValueError(
                "--tokenizer_path is required when using --max_prompt_tokens or --max_reference_tokens."
            )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    samples = build_dataset(
        input_paths=input_paths,
        max_context_messages=args.max_context_messages,
        min_prompt_chars=args.min_prompt_chars,
        min_reference_chars=args.min_reference_chars,
        tokenizer=tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_reference_tokens=args.max_reference_tokens,
    )
    if not samples:
        raise RuntimeError("No valid samples extracted. Check input format/filters.")

    train_set, eval_set = split_dataset(samples, args.eval_ratio, args.seed)

    out_train = Path(args.out_train)
    out_eval = Path(args.out_eval)
    write_jsonl(out_train, train_set)
    write_jsonl(out_eval, eval_set)

    print(f"[GRPOG] total={len(samples)} train={len(train_set)} eval={len(eval_set)}")
    print(f"[GRPOG] train_out={out_train}")
    print(f"[GRPOG] eval_out={out_eval}")


if __name__ == "__main__":
    main()
