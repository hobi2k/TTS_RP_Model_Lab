"""
jsonl_to_translation_sft.py

목적:
- RP용 원본 JSONL 데이터셋을
- 번역 전용 Qwen/Qwen3-1.7B-Base SFT 학습용 JSONL로 변환한다.

핵심 설계 원칙:
1. system / user / assistant_raw 완전 제거
2. narration / dialogue를 서로 다른 학습 샘플로 분리
3. 1 input (ko) → 1 output (jp) 구조 유지
4. 캐릭터 / 감정 / 연출 정보는 절대 번역 모델에 주입하지 않음
"""

import json
import os
from pathlib import Path


# 설정부
BASE_DIR = Path("/mnt/d/rp_data/singleturn")
INPUT_JSONL_PATH = Path(BASE_DIR / "rp_generated.jsonl")
OUTPUT_JSONL_PATH = Path(BASE_DIR / "ko-ja_translation_sft.jsonl")

INSTRUCTION_TEXT = "다음 한국어 문장을 자연스러운 일본어로 번역하시오."


# 유틸 함수
def is_valid_pair(ko: str, jp: str) -> bool:
    """
    한국어/일본어 번역 쌍이 학습에 적합한지 검사한다.

    조건:
    - None 아님
    - 빈 문자열 아님
    - 좌우 공백 제거 후 길이 > 0
    """
    if ko is None or jp is None:
        return False

    ko = ko.strip()
    jp = jp.strip()

    if not ko or not jp:
        return False

    return True


def make_sft_sample(ko: str, jp: str) -> dict:
    """
    Qwen SFT 학습용 샘플을 생성한다.
    """
    return {
        "instruction": INSTRUCTION_TEXT,
        "input": ko.strip(),
        "output": jp.strip(),
    }


# 메인 변환 로직
def convert_jsonl():
    output_samples = []

    with INPUT_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_idx}: {e}")
                continue

            # scene 정보가 없는 경우는 스킵
            scene = data.get("scene")
            if scene is None:
                continue

            # narration 처리
            # narration = scene.get("narration")
            # if narration:
            #     ko = narration.get("ko")
            #     jp = narration.get("jp")

            #     if is_valid_pair(ko, jp):
            #         output_samples.append(
            #             make_sft_sample(ko, jp)
            #         )

            # dialogue 처리
            dialogue = scene.get("dialogue")
            if dialogue:
                ko = dialogue.get("ko")
                jp = dialogue.get("jp")

                if is_valid_pair(ko, jp):
                    output_samples.append(
                        make_sft_sample(ko, jp)
                    )

    # 결과 저장
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as f:
        for sample in output_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(output_samples)} samples written to {OUTPUT_JSONL_PATH}")


# 엔트리 포인트
if __name__ == "__main__":
    convert_jsonl()
