"""
uv run data/singleturn/singleturn_message_converter.py \
  --in_path /mnt/d/rp_data/singleturn/rp_generated_local.jsonl \
  --out_path /mnt/d/rp_data/singleturn/rp_generated_local_cleaned.jsonl

uv run data/singleturn/singleturn_message_converter.py \
  --in_path /mnt/d/rp_data/singleturn/rp_generated.jsonl \
  --out_path /mnt/d/rp_data/singleturn/rp_generated_cleaned.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def convert_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    system / user / assistant_raw → messages[] singleton SFT format
    """
    system = rec.get("system")
    user = rec.get("user")
    assistant = rec.get("assistant_raw")

    if not system or not user or not assistant:
        raise ValueError("필수 필드(system/user/assistant_raw) 누락")

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                new_rec = convert_record(rec)
                fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                converted += 1
            except Exception as e:
                raise RuntimeError(f"{line_no}번째 줄 처리 실패: {e}")

    print(f"✅ 변환 완료: {converted} samples")
    print(f"📄 출력 파일: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", required=True, help="입력 jsonl 파일")
    parser.add_argument("--out_path", required=True, help="출력 jsonl 파일")
    args = parser.parse_args()

    main(args.in_path, args.out_path)
