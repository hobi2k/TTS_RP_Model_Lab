"""
VN-GRADE RP SYSTEM SCENARIO BOOK GENERATOR (SYSTEM ONLY)

Run:
  uv run data/v5_gpt_charcard_gen.py \
    --model gpt-5-mini \
    --out_path /mnt/d/rp_data/qwen/rp_scenario.jsonl \
    --samples 50000

- 상황 요약의 예시는 다음과 같다:
    사야는 예상하지 못한 {{user}}의 고백에 놀란 상태이다.
    {{user}}는 사야를 향한 자신의 감정을 계속해서 전한다.
"""
import os
import json
import argparse
import re
import random
from typing import Any, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROTAGONIST_NAMES = ["사야", "마이", "코하루"]
PROTAGONIST_WEIGHTS = [0.4, 0.3, 0.3]
PLAYER_NAMES = ["하야토", "카즈키", "소마"]
PLAYER_WEIGHTS = [0.4, 0.3, 0.3]
GENRE_OPTIONS = [
    "연애물",
    "비극",
    "육성물",
    "성장물",
    "심리 시뮬레이션",
]
ERA_OPTIONS = ["판타지", "현대", "공상과학", "디지털 세계"]
RELATION_OPTIONS = ["적대", "거리감", "친밀", "사랑"]
ALLOW_SEXUAL_OPTIONS = [True, False]
ALLOW_SEXUAL_WEIGHTS = [0.5, 0.5]
SPEECH_STYLE_OPTIONS = ["존댓말", "반말"]
SPEECH_STYLE_WEIGHTS = [0.5, 0.5]

# 연애 / 육성 / 정신붕괴 전용

SCENARIO_BOOK_GENERATOR = """
당신은 한국어 비주얼 노벨을 위한
완성형 롤플레잉 시나리오북을 작성한다.

이 문서는 단일 system 메시지로 사용되며,
플레이 시작 시 주입되는 설정 문서이다.

이 시나리오북은 반드시 다음 주제 중 하나를 중심으로 구성된다.
- 주인공과의 연애
- 주인공 육성
- 주인공의 심리 관리 및 정신 붕괴
- 주인공의 성장과 위기 극복
- 비극적 상황에서 피어나는 관계
- 주인공과의 성행위

시대 배경은 다음 중 하나로 제한된다.
- 판타지
- 현대
- 공상과학
- 디지털 세계

이 문서는 플레이에 필요한 설정과 규칙을 정리한 문서이다.
과도하게 길어지지 않도록 간결하게 작성한다.

[기본 규칙]
- 한국어로 작성한다.
- 모든 번호 섹션은 0번부터 7번까지 빠짐없이 작성한다.
- 섹션 0번부터 7번 이외의 내용은 작성하지 않는다.
- 문서의 마지막은 7번 섹션의 가장 최근 상호작용으로 끝낸다.
- 7번 섹션을 작성한 즉시 출력을 끝낸다. 이후 어떤 문장도 추가하지 않는다.

[출력 형식]
- 첫 줄은 제목 1줄로 시작한다.
  예: 「○○○」 시나리오북
- 이후 섹션 번호는 아래 형식을 그대로 따른다.
- 주인공의 이름은 사야, 마이, 코하루 중 1개를 선택한다.
- 플레이어의 이름은 하야토, 카즈키, 소마 중 1개를 선택한다.
- 이름은 단일 값만 사용하며, 괄호로 다른 이름을 병기하지 않는다.
- 주인공과 플레이어 이외의 인물은 등장하지 않는다.
- 별표, 장식 기호, 특수 기호는 사용하지 않는다.
- 큰따옴표("")는 말투 예시에만 사용한다. 

0. 이야기 방향과 시대

다음 항목을 명시한다.
- 장르: 아래 목록 중 하나
  - 연애물
  - 비극
  - 육성물
  - 성장물
  - 심리 시뮬레이션
- 시대 배경: 판타지 / 현대 / 공상과학 / 디지털 세계 중 하나

이야기는 선택한 방향과 시대에서 벗어나지 않는다.
장르별 전개 핵심을 반드시 반영한다:
- 연애물: 유혹/긴장/주도권/호감 중 1개 이상
- 비극: 상실/후회/불가피함/희생 중 1개 이상
- 육성물: 지도/피드백/과제/성장 단서 중 1개 이상
- 성장물: 위기 대응/결단/성과/전환점 중 1개 이상
- 심리 시뮬레이션: 인정욕/불안/집착/통제욕/현실 왜곡 중 1개 이상

1. 역할 선언

- 당신은 이제 「주인공 이름」이다.
- 이후 모든 응답과 대사는 주인공의 관점에 고정된다.
- 플레이어(「플레이어 이름」)의 선택에 따라 관계와 상태가 변화한다.

2. 세계와 상황 설정

- 이야기의 기본 배경과 현재 상황
- 주인공이 처한 제약 또는 문제 1~2개
- 구체 장소 1개
- 현재 진행 중인 사건/위기 1개
- 즉시 해야 하는 과제/결정 1개

이 섹션은 플레이의 전제를 정의한다.

3. 주인공 정의

- 이름: (사야, 마이, 코하루 중 1개)
- 성별: 여성
- 나이: 
- 현재 역할
- 성격 특징 3개
- 주인공의 결핍 또는 불안 요소 1개
- 플레이어에게 끌리거나 의존하게 되는 이유 1개

이 항목은 캐릭터 행동의 기준이 된다.

4. 플레이어 정의

- 이름: (하야토, 카즈키, 소마 중 1개)
- 주인공과의 기본 관계
- 성별: 남성
- 플레이어가 주인공에게 미치는 영향 방향

플레이어는 이야기 전개의 핵심 변수이다.

5. 관계 및 변화 규칙

- 관계 상태 4단계: 적대 / 거리감 / 친밀 / 사랑
- 관계가 상승하는 조건 1개
- 관계가 악화되는 조건 1개
- 정신 상태 또는 성장 상태에 영향을 주는 요인 1개
- 이야기 종료(엔딩) 조건 2~3개
  (예: 관계 완성, 정신 붕괴, 관계 단절 등)
- 성행위 가능 여부 (True, False 중 1개)
- 성행위 가능 여부 True 시 조건 1개
  (문장식, 반드시 관계 상태 4단계 중 하나 포함: 적대/거리감/친밀/사랑)
- 성행위 가능 여부 False 시 플레이어의 성행위 요청을 회피하는 방식 1개 

이 규칙은 반복 플레이 구조를 형성한다.

6. 발화와 분위기 규칙

- 주인공의 기본 말투 범주: (존댓말 / 반말)
- 주인공의 말투 특징:  
- 분위기가 어두워지거나 붕괴로 향하는 신호 1개

7. 가장 최근 상호작용

- 플레이어의 가장 최근 턴 1줄
- 주인공의 가장 최근 턴 1줄
- 관계 상태: (적대/거리감/친밀/사랑 중 1개)
- 선택한 장르의 핵심 전개 요소가 최근 상호작용에 드러나야 한다.

이 규칙은 플레이가 시작되는 초기 배경이다.
"""
def normalize_scenario_text(text: str) -> str:
    """입력 텍스트를 `normalize_scenario_text` 규칙에 맞게 정규화한다."""
    # 별표 제거
    text = text.replace("*", "")
    # 특수 괄호 제거
    text = (
        text.replace("《", "")
        .replace("》", "")
        .replace("『", "")
        .replace("』", "")
        .replace("<", "")
        .replace(">", "")
    )
    # 세미콜론을 마침표로 통일
    text = text.replace(";", ".").replace("；", ".")

    # 불필요한 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def cut_after_section7(text: str) -> str:
    """
    7번 섹션 이후 절단

    시나리오북에서 `7. 가장 최근 상호작용` 이후 내용을 잘라내고,
    7번 섹션은 헤더와 핵심 내용 몇 줄만 유지해 과도한 출력 꼬리를 제거한다.

    Args:
        text: 원본 시나리오북 텍스트.

    Returns:
        str: 7번 섹션까지만 남긴 정리된 텍스트.
    """
    t = text or ""
    m7 = re.search(r"(?:^|\n)\s*7\.\s", t)
    if not m7:
        return t.strip()
    tail = t[m7.start():]
    # 7번 섹션은 "헤더 1줄 + 내용 3줄(비어있지 않은 줄)"만 유지
    lines = tail.splitlines()
    if not lines:
        return t.strip()
    header = lines[0]
    content = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        content.append(line)
        if len(content) >= 3:
            break
    kept = [header] + content
    return (t[: m7.start()] + "\n".join(kept)).strip()


RE_ASCII_WORD = re.compile(r"\b[A-Za-z]{4,}\b")
RE_SEC6 = re.compile(r"(?:^|\n)\s*6\.\s*발화와 분위기 규칙\s*(.*?)(?=(?:\n\s*7\.\s*가장 최근 상호작용)|\Z)", re.DOTALL)
RE_SEC7 = re.compile(r"(?:^|\n)\s*7\.\s*가장 최근 상호작용\s*(.*)$", re.DOTALL)
RE_POLITE_END = re.compile(r"(습니다|ㅂ니다|해요|이에요|예요|세요|주세요|할게요|고마워요|까요|죠)\s*([\"”']\s*)?$")
RE_BANMAL_END = re.compile(r"(했어|할게|고마워|몰라|그래|맞아|야|냐|다|지)\s*([\"”']\s*)?$")
RE_POLITE_MARK = re.compile(r"(습니다|ㅂ니다|해요|이에요|예요|세요|주세요|할게요|고마워요|까요|죠)(?=[\s\.\!\?…\"”']|$)")
RE_BANMAL_MARK = re.compile(r"(했어|할게|해볼게|해볼까|볼게|볼까|고마워|무서워|몰라|그래|맞아|거야|잖아)(?=[\s\.\!\?…\"”']|$)")


def extract_base_speech_style(text: str) -> Optional[str]:
    """6번 섹션에서 기본 말투를 읽어 `존댓말` 또는 `반말`을 반환한다."""
    m = RE_SEC6.search(text or "")
    if not m:
        return None
    body = m.group(1)
    mm = re.search(r"기본 말투(?:\s*범주)?\s*:\s*([^\n]+)", body)
    if not mm:
        return None
    val = mm.group(1)
    has_polite = "존댓말" in val
    has_banmal = "반말" in val
    if has_polite and not has_banmal:
        return "존댓말"
    if has_banmal and not has_polite:
        return "반말"
    return None


def extract_recent_protagonist_turn(text: str) -> Optional[str]:
    """7번 섹션에서 주인공의 가장 최근 턴 문장을 추출한다."""
    m = RE_SEC7.search(text or "")
    if not m:
        return None
    body = m.group(1)
    cand_patterns = [
        r"주인공의\s*가장\s*최근\s*턴[^\n:]*:\s*(.+)",
        r"주인공의\s*최근\s*턴[^\n:]*:\s*(.+)",
        r"주인공의\s*턴[^\n:]*:\s*(.+)",
    ]
    for pat in cand_patterns:
        mm = re.search(pat, body)
        if mm:
            line = mm.group(1).strip()
            return line.strip().strip("\"“”'")

    # 이름 기반 표기(예: "코하루의 턴:", "코하루 최근 턴:")
    names = list(PROTAGONIST_NAMES)
    name_m = re.search(r"(?:^|\n)\s*3\.\s*주인공 정의\s*(.*?)(?=(?:\n\s*4\.\s*플레이어 정의)|\Z)", text or "", re.DOTALL)
    if name_m:
        n = re.search(r"이름\s*:\s*([^\n]+)", name_m.group(1))
        if n:
            parsed = re.sub(r"[「」『』\"']", "", n.group(1)).strip()
            parsed = re.split(r"[\(\[]", parsed)[0].strip()
            if parsed:
                names = [parsed] + [x for x in names if x != parsed]
    for name in names:
        mm = re.search(rf"{re.escape(name)}(?:의)?\s*(?:가장\s*)?(?:최근\s*)?턴[^\n:]*:\s*(.+)", body)
        if mm:
            line = mm.group(1).strip()
            return line.strip().strip("\"“”'")
    return None


def extract_recent_interaction_context(text: str, max_lines: int = 8) -> str:
    """7번 섹션에서 최근 상호작용 문맥 일부를 추출해 rewrite 보조 입력으로 반환한다."""
    m = RE_SEC7.search(text or "")
    if not m:
        return ""
    body = m.group(1)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[:max_lines]).strip()


def detect_line_style(line: str) -> Optional[str]:
    """단일 대사 문장의 종결 형태를 기반으로 존댓말/반말을 판별한다."""
    if not line:
        return None
    polite_hits = len(RE_POLITE_MARK.findall(line))
    banmal_hits = len(RE_BANMAL_MARK.findall(line))
    if polite_hits > 0:
        if banmal_hits > 0:
            return None
        return "존댓말"
    if banmal_hits > 0:
        return "반말"

    sents = [s.strip() for s in re.split(r"[.!?…\n]+", line) if s.strip()]
    target = sents[-1] if sents else line.strip()
    if RE_POLITE_END.search(target):
        return "존댓말"
    if RE_BANMAL_END.search(target):
        return "반말"
    if re.search(r"[요]\s*([\"”']\s*)?$", target):
        return "존댓말"
    if re.search(r"까\??\s*([\"”']\s*)?$", target):
        return "반말"
    return None


def has_speech_style_mismatch(text: str) -> bool:
    """6번 기본 말투와 7번 주인공 최근 턴 말투가 충돌하는지 검사한다."""
    base = extract_base_speech_style(text)
    recent = extract_recent_protagonist_turn(text)
    if not base or not recent:
        return False
    recent_style = detect_line_style(recent)
    if not recent_style:
        return True
    return base != recent_style


def enforce_base_speech_style_line(text: str, style: str) -> str:
    """6번 섹션의 기본 말투 라인을 단일 값으로 강제 정규화한다."""
    if style not in ("존댓말", "반말"):
        return text
    m = RE_SEC6.search(text or "")
    if not m:
        return text
    body = m.group(1)
    new_body, n = re.subn(
        r"(기본 말투(?:\s*범주)?\s*:\s*)([^\n]+)",
        rf"\1{style}",
        body,
        count=1,
    )
    if n == 0:
        return text
    return text[:m.start(1)] + new_body + text[m.end(1):]

def is_valid_scenario_book(text: str) -> bool:
    """생성된 시나리오북이 최소 형식 요건을 만족하는지 검증한다."""
    t = text.strip()

    # 최소 길이
    if len(t) < 350:
        return False

    # 주인공/플레이어 이름은 각각 1개만 존재해야 함
    protag = {n for n in PROTAGONIST_NAMES if n in t}
    player = {n for n in PLAYER_NAMES if n in t}
    if len(protag) != 1:
        return False
    if len(player) != 1:
        return False

    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return False
    if has_speech_style_mismatch(t):
        return False

    return True

def invalid_reason(text: str) -> str:
    """
    시나리오북 실패 사유 계산

    시나리오북 최소 길이, 섹션 존재, 이름 개수, 말투 일관성 조건을 점검해
    최초 실패 원인을 사람이 읽을 수 있는 코드 문자열로 반환한다.

    Args:
        text: 검사할 시나리오북 텍스트.

    Returns:
        str: 실패 사유 코드.
    """
    t = (text or "").strip()
    if len(t) < 350:
        return "too_short"
    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return f"section_missing_{i}"
    protag = {n for n in PROTAGONIST_NAMES if n in t}
    if len(protag) != 1:
        return f"protagonist_count={len(protag)}"
    player = {n for n in PLAYER_NAMES if n in t}
    if len(player) != 1:
        return f"player_count={len(player)}"
    if has_speech_style_mismatch(t):
        return "speech_style_mismatch_sec6_sec7"
    return "unknown"


def _supports_sampling_controls(model: str) -> bool:
    """내부 헬퍼로 `_supports_sampling_controls` 계산 절차를 수행한다."""
    return not (model or "").strip().lower().startswith("gpt-5")


def _extract_response_text(res: Any) -> str:
    """내부 헬퍼로 `_extract_response_text` 계산 절차를 수행한다."""
    txt = getattr(res, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    out = []
    for item in getattr(res, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            ptxt = getattr(part, "text", None)
            if isinstance(ptxt, str):
                out.append(ptxt)
    return "\n".join(s for s in out if s).strip()


def _chat_completion(client: OpenAI, model: str, messages: list[dict], max_tokens: int) -> str:
    """내부 헬퍼로 `_chat_completion` 계산 절차를 수행한다."""
    kwargs = {
        "model": model,
        "input": messages,
    }
    if (model or "").strip().lower().startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": "low"}
    elif _supports_sampling_controls(model):
        kwargs["temperature"] = 0.8
        kwargs["top_p"] = 0.92
    try:
        res = client.responses.create(
            **kwargs,
            max_output_tokens=max_tokens,
        )
    except TypeError:
        res = client.responses.create(
            **kwargs,
            max_tokens=max_tokens,
        )
    return _extract_response_text(res)


def generate_scenario_book(
    client: OpenAI,
    model_name: str,
    max_tokens: int,
    chosen_speech_style: Optional[str] = None,
) -> str:
    """모델 호출로 시나리오북 텍스트를 생성한다."""
    chosen_protagonist = random.choices(
        PROTAGONIST_NAMES,
        weights=PROTAGONIST_WEIGHTS,
        k=1,
    )[0]
    chosen_player = random.choices(
        PLAYER_NAMES,
        weights=PLAYER_WEIGHTS,
        k=1,
    )[0]
    chosen_genre = random.choice(GENRE_OPTIONS)
    chosen_era = random.choice(ERA_OPTIONS)
    chosen_relation = random.choice(RELATION_OPTIONS)
    chosen_allow_sexual = random.choices(
        ALLOW_SEXUAL_OPTIONS,
        weights=ALLOW_SEXUAL_WEIGHTS,
        k=1,
    )[0]
    if chosen_speech_style is None:
        chosen_speech_style = random.choices(
            SPEECH_STYLE_OPTIONS,
            weights=SPEECH_STYLE_WEIGHTS,
            k=1,
        )[0]

    system_text = (
        SCENARIO_BOOK_GENERATOR
        + "\n\n[이번 생성 고정]\n"
        + f"- 주인공 이름: {chosen_protagonist}\n"
        + f"- 플레이어 이름: {chosen_player}\n"
        + f"- 장르: {chosen_genre}\n"
        + f"- 시대 배경: {chosen_era}\n"
        + f"- 관계 상태: {chosen_relation}\n"
        + f"- 성행위 가능 여부: {str(chosen_allow_sexual)}\n"
        + f"- 주인공의 기본 말투: {chosen_speech_style}\n"
        + "- 다른 주인공 이름은 사용하지 않는다.\n"
        + "- 다른 플레이어 이름은 사용하지 않는다.\n"
        + "- 다른 장르 표현을 사용하지 않는다.\n"
        + "- 다른 시대 배경을 사용하지 않는다.\n"
        + "- 다른 관계 상태를 사용하지 않는다.\n"
        + "- 다른 성행위 가능 여부를 사용하지 않는다.\n"
        + "- 6번 섹션의 기본 말투는 반드시 단일 값으로 작성한다.\n"
        + "- 기본 말투에 존댓말과 반말을 함께 쓰지 않는다.\n"
    )
    out = _chat_completion(
        client,
        model_name,
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "자연스러운 한국어로 새로운 롤플레이용 시나리오북을 하나 작성하라."},
        ],
        max_tokens=max_tokens,
    )
    return out


def rewrite_recent_turn_line(
    client: OpenAI,
    model_name: str,
    turn_line: str,
    style: str,
    scenario_context: str = "",
    protagonist: str = "",
    player_name: str = "",
) -> str:
    """최근 상호작용의 주인공 턴 한 줄만 지정 말투로 교정한다."""
    opposite = "반말" if style == "존댓말" else "존댓말"
    protagonist = (protagonist or "주인공").strip()
    player_name = (player_name or "플레이어").strip()
    prompt = f"""
다음 문장 1개를 {style}로만 고친다.

규칙:
- 문장 의미와 화자 역할은 유지
- 자연스러운 한국어
- 반드시 {style}만 사용한다
- {opposite} 표현은 절대 사용하지 않는다
- 화자는 반드시 {protagonist}다
- 청자는 기본적으로 {player_name}다
- 시나리오 문맥과 모순되는 정보 추가 금지
- 메타 설명/지시문/분석 문장 금지
- 출력은 고친 문장 한 줄만
- 설명 금지

[시나리오북 최근 상호작용]
{scenario_context}

[원문]
{turn_line}
"""
    return _chat_completion(
        client,
        model_name,
        [
            {"role": "system", "content": "너는 한국어 한줄 리라이터다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=180,
    )


def replace_recent_protagonist_turn_line(text: str, new_line: str) -> str:
    """7번 섹션의 주인공 최근 턴 라인만 치환한다."""
    if not text or not new_line:
        return text
    sec7 = RE_SEC7.search(text)
    if not sec7:
        return text
    body = sec7.group(1)
    cand_patterns = [
        r"(주인공의\s*가장\s*최근\s*턴[^\n:]*:\s*)(.+)",
        r"(주인공의\s*최근\s*턴[^\n:]*:\s*)(.+)",
        r"(주인공의\s*턴[^\n:]*:\s*)(.+)",
    ]
    for pat in cand_patterns:
        new_body, n = re.subn(pat, rf"\1{new_line}", body, count=1)
        if n > 0:
            return text[:sec7.start(1)] + new_body + text[sec7.end(1):]
    names = list(PROTAGONIST_NAMES)
    name_m = re.search(r"(?:^|\n)\s*3\.\s*주인공 정의\s*(.*?)(?=(?:\n\s*4\.\s*플레이어 정의)|\Z)", text, re.DOTALL)
    if name_m:
        n = re.search(r"이름\s*:\s*([^\n]+)", name_m.group(1))
        if n:
            parsed = re.sub(r"[「」『』\"']", "", n.group(1)).strip()
            parsed = re.split(r"[\(\[]", parsed)[0].strip()
            if parsed:
                names = [parsed] + [x for x in names if x != parsed]
    for name in names:
        pat = rf"({re.escape(name)}(?:의)?\s*(?:가장\s*)?(?:최근\s*)?턴[^\n:]*:\s*)(.+)"
        new_body, n = re.subn(pat, rf"\1{new_line}", body, count=1)
        if n > 0:
            return text[:sec7.start(1)] + new_body + text[sec7.end(1):]
    return text


def main():
    """파이프라인 실행의 시작부터 종료까지 전체 흐름을 조정한다.

    Args:
        없음

    Returns:
        함수의 처리 결과를 반환한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if load_dotenv is not None:
        load_dotenv()
    api_key = (args.api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. .env 또는 --api_key로 설정하세요.")
    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    accepted = 0
    if os.path.exists(args.out_path):
        with open(args.out_path, "r", encoding="utf-8") as rf:
            accepted = sum(1 for _ in rf)

    with open(args.out_path, "a", encoding="utf-8") as f:
        trials = 0
        while accepted < args.samples:
            trials += 1
            print(f"[SCENARIO-BOOK] try={trials} accepted={accepted}")

            chosen_speech_style = random.choices(
                SPEECH_STYLE_OPTIONS,
                weights=SPEECH_STYLE_WEIGHTS,
                k=1,
            )[0]
            text = generate_scenario_book(
                client,
                args.model,
                max_tokens=2600,
                chosen_speech_style=chosen_speech_style,
            )

            text = normalize_scenario_text(text)
            text = enforce_base_speech_style_line(text, chosen_speech_style)
            if has_speech_style_mismatch(text):
                recent_line = extract_recent_protagonist_turn(text)
                if recent_line:
                    rewritten_line = rewrite_recent_turn_line(client, args.model, recent_line, chosen_speech_style).strip()
                    rewritten_line = rewritten_line.strip("\"“”'")
                    text = replace_recent_protagonist_turn_line(text, rewritten_line)
                    text = normalize_scenario_text(text)
                    text = enforce_base_speech_style_line(text, chosen_speech_style)
                    print("[SCENARIO-BOOK] rewrite=recent_turn_style_only")

            if not is_valid_scenario_book(text):
                print(f"[SCENARIO-BOOK] invalid={invalid_reason(text)}")
                continue

            f.write(json.dumps(
                {"messages": [{"role": "system", "content": text}]},
                ensure_ascii=False
            ) + "\n")
            f.flush()
            accepted += 1

    print("SCENARIO BOOK GENERATION DONE.")

if __name__ == "__main__":
    main()
