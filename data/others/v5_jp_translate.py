"""
jsonl_to_translation_sft.py

목적:
- RP 원본 JSONL에서
- assistant의 "대사"만 추출
- EEVE로 일본어 번역
- 번역 전용 SFT 학습용 JSONL 생성

핵심 설계 원칙:
1. system / user / 서술 완전 제거
2. assistant의 큰따옴표(" ") 대사만 사용
3. 1 input (ko) -> 1 output (jp)
4. RP 문맥 / 캐릭터 / 감정 정보 미주입
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Optional
from openai import OpenAI


# 설정부
BASE_DIR = Path("/mnt/d/rp_data")
INPUT_JSONL_PATH = BASE_DIR / "rp_generated.jsonl"
OUTPUT_JSONL_PATH = BASE_DIR / "ko-ja_translation_sft.jsonl"
PROGRESS_PATH = OUTPUT_JSONL_PATH.with_suffix(".progress.json")

INSTRUCTION_TEXT = "다음 한국어 대사를 자연스러운 일본어로 번역하시오."

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)
MODEL_NAME = "yanolja/EEVE-Korean-Instruct-7B-v2.0"

SAVE_EVERY = 50  # 몇 개 샘플마다 progress 저장


# 유틸 함수

RE_QUOTE = re.compile(r'"([^"]+)"')

def extract_quoted_dialogues(text: str) -> list[str]:
    if not text:
        return []
    dialogues = RE_QUOTE.findall(text)
    return [d.strip() for d in dialogues if d and d.strip()]

def normalize_ko(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def ko_key(ko: str) -> str:
    # 중복 제거/저장용 해시 키
    return hashlib.md5(normalize_ko(ko).encode("utf-8")).hexdigest()

def translate_ko_to_ja(ko: str) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INSTRUCTION_TEXT},
                {"role": "user", "content": ko},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out if out else None
    except Exception as e:
        print(f"[WARN] Translation failed: {e}")
        return None

def is_valid_pair(ko: str, jp: Optional[str]) -> bool:
    if not ko or not jp:
        return False
    return bool(ko.strip()) and bool(jp.strip())

def make_sft_sample(ko: str, jp: str) -> dict:
    return {"instruction": INSTRUCTION_TEXT, "input": ko, "output": jp}


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "line_idx": 1,      # INPUT_JSONL의 1-based line index
        "msg_idx": 0,       # messages 내 assistant msg index(assistant만 카운트)
        "dlg_idx": 0,       # 해당 assistant 메시지의 dialogue index
        "accepted": 0,
        "seen": 0,
    }

def save_progress(p: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")

def load_existing_keys() -> set[str]:
    """
    이미 OUTPUT_JSONL에 저장된 input(ko) 중복 방지.
    파일이 크면 시간이 걸릴 수 있지만, 재개/중복 방지에 확실함.
    """
    keys = set()
    if not OUTPUT_JSONL_PATH.exists():
        return keys

    with OUTPUT_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ko = obj.get("input", "")
                if ko:
                    keys.add(ko_key(ko))
            except Exception:
                continue
    return keys


# 메인 변환 로직 (resume)

def convert_jsonl_resume():
    prog = load_progress()
    start_line = int(prog.get("line_idx", 1))
    start_msg = int(prog.get("msg_idx", 0))
    start_dlg = int(prog.get("dlg_idx", 0))

    accepted = int(prog.get("accepted", 0))
    seen = int(prog.get("seen", 0))

    existing = load_existing_keys()
    print(f"[INFO] resume from line={start_line}, msg={start_msg}, dlg={start_dlg}")
    print(f"[INFO] existing samples={len(existing)}")

    # append 모드: 중간에 죽어도 결과 유지
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUTPUT_JSONL_PATH.open("a", encoding="utf-8")

    try:
        with INPUT_JSONL_PATH.open("r", encoding="utf-8") as in_f:
            for line_idx, line in enumerate(in_f, 1):
                if line_idx < start_line:
                    continue

                line = line.strip()
                if not line:
                    # progress는 라인 단위로도 업데이트
                    prog.update({"line_idx": line_idx + 1, "msg_idx": 0, "dlg_idx": 0})
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON decode error at line {line_idx}: {e}")
                    prog.update({"line_idx": line_idx + 1, "msg_idx": 0, "dlg_idx": 0})
                    continue

                messages = data.get("messages", [])

                # assistant 메시지를 "assistant만 카운트"해서 msg_idx 진행
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

                for a_i, msg in enumerate(assistant_msgs):
                    if line_idx == start_line and a_i < start_msg:
                        continue

                    content = msg.get("content", "")
                    dialogues = extract_quoted_dialogues(content)

                    for d_i, ko in enumerate(dialogues):
                        if line_idx == start_line and a_i == start_msg and d_i < start_dlg:
                            continue

                        seen += 1
                        ko = normalize_ko(ko)
                        key = ko_key(ko)

                        # 이미 번역된 ko면 skip
                        if key in existing:
                            prog.update({"line_idx": line_idx, "msg_idx": a_i, "dlg_idx": d_i + 1,
                                         "accepted": accepted, "seen": seen})
                            continue

                        jp = translate_ko_to_ja(ko)
                        if not is_valid_pair(ko, jp):
                            prog.update({"line_idx": line_idx, "msg_idx": a_i, "dlg_idx": d_i + 1,
                                         "accepted": accepted, "seen": seen})
                            continue

                        sample = make_sft_sample(ko, jp)
                        out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        out_f.flush()

                        existing.add(key)
                        accepted += 1

                        prog.update({"line_idx": line_idx, "msg_idx": a_i, "dlg_idx": d_i + 1,
                                     "accepted": accepted, "seen": seen})

                        if accepted % SAVE_EVERY == 0:
                            save_progress(prog)
                            print(f"[INFO] accepted={accepted}, seen={seen}, last=({line_idx},{a_i},{d_i})")

                # 한 줄(한 scenario) 끝나면 다음 line로 진행, 인덱스 리셋
                prog.update({"line_idx": line_idx + 1, "msg_idx": 0, "dlg_idx": 0,
                             "accepted": accepted, "seen": seen})
                save_progress(prog)

    finally:
        out_f.close()
        save_progress(prog)

    print(f"[DONE] seen_dialogues={seen}, accepted_samples={accepted}")
    print(f"[PATH] {OUTPUT_JSONL_PATH}")
    print(f"[PROGRESS] {PROGRESS_PATH}")


if __name__ == "__main__":
    convert_jsonl_resume()
