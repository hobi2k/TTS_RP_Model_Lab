#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rp_dataset_preprocess_rewritten.py

RP 멀티턴 데이터셋 전처리 (REWRITTEN 결과물 전용) (JSONL -> JSONL)

포함 기능 (요구사항 + 추가 메타 제거)
- (system) 2번째 system(멀티턴 규칙) 제거, 1번째 시나리오북 system 유지
- (assistant) rewrite 메타 블록/라벨/자기지시/선택지/scene direction/fsm 등 제거 (문장/블록 단위)
- (assistant) /결정/, /반응/결정] 같은 깨진 슬래시 메타 라벨 라인 제거
- (assistant) **리라의 대사** 같은 굵은 라벨 라인 제거
- (assistant) 소연(주인공) -> 소연 (괄호 설명만 제거)
- (assistant) '주인공' 표지 토큰을 성별 대명사(그/그녀)로 치환(조사 보존)
- (all) 끊긴 문장(라인 끝이 . ? ! " … “ ” 로 끝나지 않음) 라인 삭제

사용 예시:
uv run data/others/v5_data_preprocessing.py \
  --in_jsonl /mnt/d/rp_data/singleturn/rp_singleturn_united.jsonl \
  --out_jsonl /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --model_name models/qwen3_core/model_assets/qwen3-8b \
  --max_length 4096

  
uv run data/others/v5_data_preprocessing.py \
  --in_jsonl /mnt/d/rp_data/v7/rp_datum_unite.jsonl \
  --out_jsonl /mnt/d/rp_data/v7/rp_datum_unite_cleaned.jsonl \
  --model_name models/qwen3_core/model_assets/qwen3-8b \
  --max_length 4096  
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional

from transformers import AutoTokenizer


# 0) system(멀티턴 규칙) 탐지
RE_MULTITURN_SYSTEM = re.compile(
    r"(멀티턴\s*대화\s*기본\s*형식|\[멀티턴|직접\s*발화는\s*큰따옴표|별표\(\*.*\*\)|이모지\s*및\s*장식\s*문자|\[성인\s*모드\])",
    flags=re.IGNORECASE,
)

def remove_second_system_multiturn_rule(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    - 첫 번째 system(시나리오북)은 유지
    - 이후 system 중 멀티턴 규칙(system injected)으로 보이는 건 제거
    """
    out: List[Dict[str, str]] = []
    system_seen = 0

    for m in messages:
        if m.get("role") != "system":
            out.append(m)
            continue

        system_seen += 1
        if system_seen == 1:
            out.append(m)
            continue

        content = m.get("content", "")
        if RE_MULTITURN_SYSTEM.search(content):
            continue
        out.append(m)

    return out


# 1) 성별 추정 (system 기반)
def infer_protagonist_gender_from_system(system_text: str) -> Optional[str]:
    """
    return: 'female' | 'male' | None
    """
    t = system_text or ""

    # '주인공 정의' / '역할 선언' 주변만 대략 스캔
    def slice_around(keyword: str, window: int = 1200) -> str:
        idx = t.find(keyword)
        if idx < 0:
            return ""
        return t[max(0, idx - 200): min(len(t), idx + window)]

    region = slice_around("주인공") + "\n" + slice_around("역할 선언") + "\n" + slice_around("성별")
    if re.search(r"(여주인공|성별\s*[:：]\s*여성|여성|여자|소녀|여전사|여사제)", region):
        return "female"
    if re.search(r"(남주인공|성별\s*[:：]\s*남성|남성|남자|소년|남전사|왕자)", region):
        return "male"

    # 약한 힌트
    if re.search(r"\b그녀\b", region):
        return "female"
    if re.search(r"\b그\b", region):
        return "male"
    return None


# 2-1) 이름 정규화 (오탈자/표기 변형 보정)
RE_SEP_LINE = re.compile(r"^\s*[-=]{3,}\s*$")


def infer_names_from_system(system_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    return: (protagonist_name, player_name)
    """
    t = system_text or ""

    protagonist_name = None
    player_name = None

    # 섹션 단위로 우선 탐지
    m_pro = re.search(r"3\.\s*주인공\s*정의(?P<body>.*?)(?:\n\s*4\.\s*플레이어\s*정의|\Z)", t, flags=re.DOTALL)
    if m_pro:
        m = re.search(r"이름\s*[:：]\s*([가-힣A-Za-z]{2,20})", m_pro.group("body"))
        if m:
            protagonist_name = m.group(1).strip()

    m_player = re.search(r"4\.\s*플레이어\s*정의(?P<body>.*?)(?:\n\s*5\.\s*관계|\Z)", t, flags=re.DOTALL)
    if m_player:
        m = re.search(r"이름\s*[:：]\s*([가-힣A-Za-z]{2,20})", m_player.group("body"))
        if m:
            player_name = m.group(1).strip()

    # 폴백: 이름 라인 앞 120자에 키워드가 있는 경우만 채택
    if protagonist_name is None or player_name is None:
        for m in re.finditer(r"이름\s*[:：]\s*([가-힣A-Za-z]{2,20})", t):
            name = m.group(1).strip()
            s = max(0, m.start() - 120)
            ctx = t[s:m.start()]
            if protagonist_name is None and re.search(r"(주인공|당신|사야|코하루|마이)", ctx):
                protagonist_name = name
            elif player_name is None and re.search(r"(플레이어|카즈키|하야토|소마)", ctx):
                player_name = name

    return protagonist_name, player_name


def _replace_whole_word_ko(text: str, wrong: str, right: str) -> str:
    if not wrong or not right or wrong == right:
        return text
    # 한글/영문 경계 기반 단어 치환
    pattern = re.compile(rf"(?<![가-힣A-Za-z]){re.escape(wrong)}(?![가-힣A-Za-z])")
    return pattern.sub(right, text)


def normalize_character_names(
    text: str,
    protagonist_name: Optional[str],
    player_name: Optional[str],
) -> str:
    if not text:
        return ""

    out = text

    # 구분선 제거
    out = "\n".join(
        ln for ln in out.split("\n")
        if not RE_SEP_LINE.match(ln.strip())
    )

    # 문맥 안전한 범위만 치환한다.
    # 일반 명사와 충돌 가능성이 있는 근접 오탈자(예: 사유)는 하드코딩 보정하지 않는다.
    if protagonist_name:
        out = _replace_whole_word_ko(out, "주인공", protagonist_name)
    if player_name:
        out = _replace_whole_word_ko(out, "플레이어", player_name)

    return out


# 3) assistant 전용: rewrite 메타 제거 (문장/블록 단위)

# (a) 깨진 슬래시 메타 토큰이 포함된 라인 제거: /결정/ /반응/결정] 등
RE_BROKEN_META_SLASH_TOKEN = re.compile(r"/[^/\n]{1,40}/")

# (b) **라벨** 단독 라인 제거: **리라의 대사**
RE_BOLD_META_LINE = re.compile(r"^\s*\*\*[^*]{1,80}\*\*\s*$")

# (c) 이름(주인공) 같은 괄호 설명 제거
RE_NAME_PAREN_ROLE = re.compile(r"(\b[가-힣]{2,10})\s*\((주인공|조연|플레이어)\)")

RE_NAME_PAREN_NAME = re.compile(r"(\b[가-힣]{2,10})\s*\(이름\)")

# (f) Scene Direction / Choice / FSM 등 기존 메타 제거
RE_SCENE_DIRECTION_LINE = re.compile(
    r"^\s*(\[Scene\s*Direction\]|Scene\s*Direction|Player'?s?\s+response\s+is\s+forced|must\s+choose|플레이어는\s+다음\s+중\s+하나를\s+선택).*$",
    flags=re.IGNORECASE,
)

RE_FSM_LINE = re.compile(
    r"^\s*(FSM|STATE|현재\s*상태|전환\s*됨|상태\s*[:=]).*$",
    flags=re.IGNORECASE,
)

RE_CHOICE_ENUM_LINE = re.compile(
    r"^\s*(?:[\(\[]?[A-C①②③④⑤]\)?[\.\)]|\d+[\.\)])\s+.*$"
)

# (g) 자기지시/자기평가 문장 제거
RE_SELF_INSTRUCTION_LINE = re.compile(
    r"(이제\s+.*(시작|진행)|아래는\s+.*(결과|출력)|다음은\s+.*(결과|출력)|본\s+출력은|위\s+기준을\s+충족)",
    flags=re.IGNORECASE,
)

# (h) '주인공' 토큰을 그/그녀로 치환(조사 보존)
RE_PROTAGONIST_TOKEN = re.compile(
    r"\b주인공(?P<josa>은|는|이|가|을|를|의|에게|에서|으로|와|과|랑|한테|께|도|만|까지|부터|조차|마저|처럼|보다)?\b"
)

def preprocess_assistant_rewritten_text(
    text: str,
    protagonist_gender: Optional[str],
    protagonist_name: Optional[str],
    player_name: Optional[str],
) -> str:
    if not text:
        return ""

    cleaned_lines: List[str] = []

    for line in text.split("\n"):
        raw = line.strip()
        if not raw:
            continue

        # 1) 깨진 슬래시 메타 토큰이 있으면 라인 제거
        if RE_BROKEN_META_SLASH_TOKEN.search(raw):
            continue

        # 2) **라벨** 단독 라인 제거
        if RE_BOLD_META_LINE.match(raw):
            continue

        # 3) Scene Direction / FSM / 선택지 라인 제거
        if RE_SCENE_DIRECTION_LINE.match(raw):
            continue
        if RE_FSM_LINE.match(raw):
            continue
        if RE_CHOICE_ENUM_LINE.match(raw):
            continue

        # 4) 자기지시/자기평가 제거
        if RE_SELF_INSTRUCTION_LINE.search(raw):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 5) 이름(주인공/조연/플레이어) -> 이름
    text = RE_NAME_PAREN_ROLE.sub(r"\1", text)

    # 5-2) 이름(이름) -> 이름
    text = RE_NAME_PAREN_NAME.sub(r"\1", text)

    # 6) 주인공 토큰 -> 그/그녀 (조사 보존)
    pron = "그녀" if protagonist_gender == "female" else "그"
    text = RE_PROTAGONIST_TOKEN.sub(lambda m: pron + (m.group("josa") or ""), text)

    # 7) 플레이어 표기 통일
    text = normalize_character_names(text, protagonist_name, player_name)

    # 8) 공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# 4) 끊긴 문장 제거 (라인 단위)
RE_LINE_END_OK = re.compile(r'.*([.!?…""“”])\s*$')

def drop_broken_sentences_by_line(text: str) -> str:
    if not text:
        return ""
    kept: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if not RE_LINE_END_OK.match(s):
            continue
        kept.append(s)
    return "\n".join(kept).strip()


# 5) 길이 제한(토큰) 처리
def count_chat_tokens(messages: List[Dict[str, str]], tokenizer) -> int:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(ids)


def trim_oldest_user_assistant_pairs_to_max_tokens(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int,
) -> Optional[List[Dict[str, str]]]:
    """
    - max_length를 넘기면 가장 오래된 (user, assistant) 2턴씩 제거
    - system 메시지는 유지
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    dialogue = [m for m in messages if m.get("role") in ("user", "assistant")]

    if len(dialogue) < 2:
        return None

    out = [*system_msgs, *dialogue]
    try:
        tok_len = count_chat_tokens(out, tokenizer)
    except Exception:
        return None

    while tok_len > max_length:
        if len(dialogue) < 2:
            return None
        dialogue = dialogue[2:]
        if len(dialogue) < 2:
            return None

        out = [*system_msgs, *dialogue]
        try:
            tok_len = count_chat_tokens(out, tokenizer)
        except Exception:
            return None

    return out


# 6) 샘플 단위 처리
def preprocess_one_sample(sample: Dict, tokenizer, max_length: int) -> Optional[Dict]:
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    # (A) system 2번째(멀티턴 규칙) 제거
    messages = remove_second_system_multiturn_rule(messages)

    # (B) 첫 system 텍스트 확보
    system_text = ""
    for m in messages:
        if m.get("role") == "system":
            system_text = m.get("content", "") or ""
            break

    protagonist_gender = infer_protagonist_gender_from_system(system_text)  # None 가능
    protagonist_name, player_name = infer_names_from_system(system_text)

    out_msgs: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            # system은 유지하되, 플레이어 표기 통일은 해도 무방(권장)
            c = normalize_character_names(content, protagonist_name, player_name)
            out_msgs.append({"role": "system", "content": c})
            continue

        if role == "user":
            # user는 성적 단어/표현은 건드리지 말라는 요구가 있으니
            # 메타 제거를 과격하게 하지 않고, 플레이어 표기 통일 + 끊긴 문장만
            c = normalize_character_names(content, protagonist_name, player_name)
            c = re.sub(r"\n{3,}", "\n\n", c).strip()
            c = drop_broken_sentences_by_line(c)
            if c:
                out_msgs.append({"role": "user", "content": c})
            continue

        if role == "assistant":
            # rewritten assistant는 메타가 많이 섞이므로 강하게 제거
            c = preprocess_assistant_rewritten_text(
                content,
                protagonist_gender,
                protagonist_name,
                player_name,
            )
            c = drop_broken_sentences_by_line(c)
            c = re.sub(r"\n{3,}", "\n\n", c).strip()
            if c:
                out_msgs.append({"role": "assistant", "content": c})
            continue

        # 기타 role은 보수적으로 유지
        c = normalize_character_names(content, protagonist_name, player_name)
        c = drop_broken_sentences_by_line(c)
        if c:
            out_msgs.append({"role": role, "content": c})

    # 최소 구조 검증
    if not out_msgs or out_msgs[0].get("role") != "system":
        return None

    # 9) 대화 턴 정렬: user-assistant 페어 강제
    # - 첫 발화가 assistant면 제거
    # - 연속 동일 role은 마지막 것만 유지
    # - 마지막이 user면 제거 (요청사항)
    system_msgs = [m for m in out_msgs if m.get("role") == "system"]
    dialogue_msgs = [m for m in out_msgs if m.get("role") in ("user", "assistant")]

    normalized_dialogue: List[Dict[str, str]] = []
    for m in dialogue_msgs:
        role = m.get("role")

        # 첫 턴 assistant는 버린다.
        if not normalized_dialogue and role == "assistant":
            continue

        # 같은 role 연속 시 마지막 메시지로 교체
        if normalized_dialogue and normalized_dialogue[-1]["role"] == role:
            normalized_dialogue[-1] = m
            continue

        normalized_dialogue.append(m)

    # 마지막 턴이 user면 제거하여 user-assistant 페어 맞춤
    if normalized_dialogue and normalized_dialogue[-1]["role"] == "user":
        normalized_dialogue.pop()

    if len(normalized_dialogue) < 2:
        return None
    if normalized_dialogue[0]["role"] != "user":
        return None
    if normalized_dialogue[-1]["role"] != "assistant":
        return None

    # 완전 교차(UAUA...) 검증
    for i, m in enumerate(normalized_dialogue):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None

    out_msgs = [*system_msgs, *normalized_dialogue]

    if len([m for m in out_msgs if m["role"] in ("user", "assistant")]) < 2:
        return None

    trimmed = trim_oldest_user_assistant_pairs_to_max_tokens(
        out_msgs,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if trimmed is None:
        return None

    return {"messages": trimmed}


# 7) IO
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="input jsonl path")
    parser.add_argument("--out_jsonl", required=True, help="output jsonl path")
    parser.add_argument(
        "--model_name",
        default="models/qwen3_core/model_assets/qwen3-8b",
        help="tokenizer path for chat-template token counting",
    )
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=True,
    )

    kept = 0
    dropped = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                dropped += 1
                continue

            cleaned = preprocess_one_sample(
                obj,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
            if cleaned is None:
                dropped += 1
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"DONE. kept={kept}, dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
