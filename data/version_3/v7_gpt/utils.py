#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data/v7_gpt_multiturn_gen.py
=========================================================
LangGraph-orchestrated VN multi-turn generator (FSM-driven)
(ROLE-SEPARATION HARDENED, NO "SIMPLIFY" REGRESSION)

- ScenarioBook (jsonl) -> Multi-turn RP (jsonl)
- FSM: QwenFSMEngine (v7_qwen_fsm_engine)
- Orchestration: LangGraph controls the whole loop (real)
- Evaluation: GPT API JSON scoring -> FSM inputs (no keyword rule scoring)
- Memory: explicit summary + EmbeddingMemory (BGE-m3-ko) anti-repeat

Run:
uv run data/v7_gpt_multiturn_gen.py \
  --openai_model gpt-5-mini \
  --scenario_path /mnt/d/rp_data/gpt/rp_scenario.jsonl \
  --out_path /mnt/d/rp_data/gpt/rp_datum.jsonl \
  --fsm_path data/version_3/v7_qwen/state_fsm.yaml \
  --action_fsm_path data/version_3/v7_qwen/action_fsm.yaml \
  --turns 3
=========================================================

핵심 변경(요구사항 반영):
- 플레이어는 시나리오북에서 추출한 "플레이어 이름"을 사용한다.
- 주인공은 시나리오북에서 추출한 "주인공 이름"을 사용한다.
- 출력에 별표(*)/장식문자/괄호()/메타(FSM, system 등) 언급을 강하게 차단한다.
- EmbeddingMemory.is_repetitive()는 keyword-only이므로 절대 positional로 넘기지 않는다.
- (선택) 서술/대사 노드 분리를 "통합 생성 1회"로 변경하여 톤 싱크/비용을 최적화한다.
  -> 다만 메모리(kind)는 narration/dialogue/assistant로 분리 저장한다.
"""

from __future__ import annotations

import re
import difflib
from typing import Dict, List, Optional, Any

import numpy as np
from embedding_utils import EmbeddingMemory



RE_QUOTE = re.compile(r"\"([^\"]+)\"")
RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)
RE_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?…])\s+")

# MOD: 별표(*) 및 다양한 장식문자/메타 단어를 더 강하게 차단하도록 확장
RE_META = re.compile(
    r"(\*+|프롬프트|요청하신|요청된|"
    r"시나리오북|헌법|검증|메타|"
    r"role\s*check|system\s*check|langgraph|fsm|phase|노드|"
    r"그래프|평가|json|prompt|instruction|guideline)",
    flags=re.IGNORECASE,
)

# MOD: 메타 전개/선택지류 문구 차단
RE_META_BLOCK = re.compile(
    r"(장면\s*전환|다음\s*장면|다음\s*장면\s*전개|다음\s*행동|"
    r"선택지|선택하세요|다음\s*전개|전개\s*요약|요약\s*:|결론\s*:|"
    r"소마의\s*다음\s*행동.*|.+의\s*선택\s*:)",
    flags=re.IGNORECASE,
)
RE_META_BLOCK_LINE = re.compile(
    r"^\s*(---+|장면\s*전환|다음\s*장면.*|다음\s*장면\s*전개:.*|"
    r"선택지.*|선택하세요.*|.*의\s*선택\s*:.*|다음\s*행동.*)\s*$",
    flags=re.IGNORECASE,
)

# Dialogue-leading directive labels (e.g., "요구: ...") that cause meta-like outputs.
RE_DIALOGUE_LABEL_PREFIX = re.compile(
    r"^\s*(요구|선언|결정|선택|지시|조건|판정|명령)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
RE_DIALOGUE_COLON_HEAD = re.compile(r"^\s*[:：]")
RE_COLON_ANY = re.compile(r"[:：]")

RE_POLICY_LEAK_LINE = re.compile(
    r"^\s*(?:[-•·]\s*)?(?:\d+\.\s*)?"
    r"(장르|관계\s*상태|관계\s*변화\s*의도|현재\s*상태|현재\s*국면|행위/사건\s*상태|"
    r"응답\s*모드|성행위\s*허용|정서의존\s*허용도|정신\s*불안정|친밀도|위협도|압박|탐색|"
    r"시나리오북|주인공\s*이름|플레이어\s*이름|출력\s*규칙|형식\s*예시|단계별\s*힌트|"
    r"이야기\s*방향과\s*시대|역할\s*선언|세계와\s*상황\s*설정|주인공\s*정의|플레이어\s*정의|"
    r"관계\s*및\s*변화\s*규칙|발화와\s*분위기\s*규칙|가장\s*최근\s*상호작용|"
    r"관능적\s*묘사|수위|톤|허용|성행위\s*규칙)\s*[:：]",
    flags=re.IGNORECASE,
)

# MOD: 괄호() 금지 규칙을 validator에서 강제하기 위한 패턴
RE_PAREN = re.compile(r"[\(\)]")

# MOD: 플레이어 이름 참조(legacy)
RE_USER_PLACEHOLDER = re.compile(r"\{\{user\}\}")
RE_LINE_LABEL = re.compile(
    r"^\s*[-•·]?\s*(서술|대사|행동|반응|대답|정보\s*공개|Action|Narration|Dialogue|Speech|Player'?s\s*Action)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
RE_FORMAL_ENDING = re.compile(r"(습니다|합니다|됩니|십니다|세요|이에요|예요|에요|니다)(?=[\s\.\!\?…]|$)")
RE_POLITE_MARK_ANY = re.compile(r"(?:[가-힣]{2,}요)(?=[,\s\.\!\?…]|$)")
RE_INFORMAL_ENDING = re.compile(r"(야|어|아|지|네|냐|군|거야|할래|했어|됐어|몰라)(?=[\s\.\!\?…]|$)")
RE_BRACKET_LABEL = re.compile(r"^\s*[\[\(].*?(서술|대사|행동|출력|정보|첫\s*번째|첫째|둘째).*?[\]\)]\s*$", flags=re.IGNORECASE)
RE_COLON_LABEL = re.compile(r"^\s*[:：]+\s*")
RE_PROTAGONIST_WORD = re.compile(r"주인공")
RE_SQUARE_BRACKET = re.compile(r"\[([^\[\]]+)\]")
PLAYER_NAME_CANDIDATES = ["하야토", "카즈키", "소마"]
RE_GENRE_LINE = re.compile(r"장르\s*:\s*([^\n]+)")
RE_PLAYER_SUBJECT = re.compile(r"^({name})(이|가|은|는)\b")
RE_SPEAKER_LABEL = re.compile(r"^\s*\"?\s*({name})\s*[:：]\s*")
RE_GENERIC_SPEAKER_LABEL = re.compile(
    r'^\s*"?\s*(플레이어|주인공|user|assistant|player|protagonist)\s*[:：]\s*',
    flags=re.IGNORECASE,
)


def normalize_space(text: str) -> str:
    """입력 텍스트를 `normalize_space` 규칙에 맞게 정규화한다."""
    return re.sub(r"\s+", " ", (text or "").strip())


def quotes_balanced(text: str) -> bool:
    """
    큰따옴표 균형 검사

    입력 문자열에서 큰따옴표 개수가 짝수인지 확인해
    인용부호가 닫히지 않은 형식 오류를 빠르게 탐지한다.

    Args:
        text: 검사 대상 문자열.

    Returns:
        bool: 따옴표 균형이 맞으면 True, 아니면 False.
    """
    return (text or "").count('"') % 2 == 0


def extract_quotes(text: str) -> List[str]:
    """입력 데이터에서 `extract_quotes`에 필요한 부분 문자열 또는 객체를 추출한다."""
    return [s.strip() for s in RE_QUOTE.findall(text or "") if s.strip()]


def extract_single_quote(text: str) -> Optional[str]:
    """입력 데이터에서 `extract_single_quote`에 필요한 부분 문자열 또는 객체를 추출한다."""
    segs = extract_quotes(text)
    return segs[0] if len(segs) == 1 else None


def has_speaker_label(line: str, names: List[str]) -> bool:
    """문장 시작에 화자 라벨이 있는지 확인한다."""
    if not line:
        return False
    if RE_GENERIC_SPEAKER_LABEL.match(line):
        return True
    if not names:
        return False
    for n in names:
        if not n:
            continue
        if re.match(RE_SPEAKER_LABEL.pattern.format(name=re.escape(n)), line):
            return True
    return False


def force_single_quote_line(text: str) -> str:
    """
    단일 대사 라인 강제 정규화

    입력 문자열에서 대사 부분만 추출해 큰따옴표 1쌍으로 감싼 형태로 맞춘다.
    비어 있는 경우 기본 침묵 대사로 대체한다.

    Args:
        text: 원본 대사 또는 혼합 문자열.

    Returns:
        str: `"..."` 형식의 단일 대사 라인.
    """
    t = (text or "").strip()
    if t.count('"') >= 2:
        first = t.find('"')
        last = t.rfind('"')
        inner = t[first + 1:last].strip()
        return f"\"{inner}\"" if inner else "\"...\""
    return f"\"{t}\"" if t else "\"...\""


def remove_all_quotes(text: str) -> str:
    """입력 텍스트에서 `remove_all_quotes` 대상 요소를 제거한다."""
    return (text or "").replace('"', "").strip()


def extract_dialogue_core(text: str) -> str:
    """
    텍스트에서 대사 핵심 문자열 추출

    2줄 형식(서술+대사) 또는 혼합 텍스트에서
    마지막 대사 문장을 우선 추출해 비교 가능한 문자열로 반환한다.

    Args:
        text: 원본 턴 텍스트.

    Returns:
        str: 따옴표를 제거한 대사 핵심 문자열.
    """
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if lines:
        q = extract_single_quote(lines[-1])
        if q is not None:
            return normalize_space(q)
    q_all = extract_single_quote(text or "")
    if q_all is not None:
        return normalize_space(q_all)
    if lines:
        return normalize_space(remove_all_quotes(lines[-1]))
    return ""


def dialogue_echo_score(a: str, b: str) -> float:
    """
    대사 간 에코 유사도 계산

    문자 유사도(SequenceMatcher)와 토큰 자카드 유사도를 함께 계산해
    user/assistant가 서로 문장을 베끼는 패턴을 감지할 점수를 만든다.

    Args:
        a: 비교 대상 대사 A.
        b: 비교 대상 대사 B.

    Returns:
        float: 0~1 범위 유사도 점수(클수록 에코 가능성 높음).
    """
    x = normalize_space(remove_all_quotes(a)).lower()
    y = normalize_space(remove_all_quotes(b)).lower()
    if not x or not y:
        return 0.0

    seq_ratio = difflib.SequenceMatcher(None, x, y).ratio()
    toks_x = {t for t in re.findall(r"[가-힣A-Za-z0-9]+", x) if len(t) >= 2}
    toks_y = {t for t in re.findall(r"[가-힣A-Za-z0-9]+", y) if len(t) >= 2}
    if not toks_x or not toks_y:
        return seq_ratio
    jac = len(toks_x & toks_y) / max(len(toks_x | toks_y), 1)
    return max(seq_ratio, jac)


def is_dialogue_echo(a: str, b: str, *, threshold: float = 0.78) -> bool:
    """
    대사 에코 여부 판정

    높은 유사도 + 포함 관계를 함께 사용해
    직전 턴 대사를 반복하거나 재진술한 출력을 차단한다.

    Args:
        a: 현재 생성 대사.
        b: 비교 기준 대사(직전 상대 대사 등).
        threshold: 에코 판정 임계값.

    Returns:
        bool: 에코로 판단되면 True.
    """
    x = normalize_space(remove_all_quotes(a)).lower()
    y = normalize_space(remove_all_quotes(b)).lower()
    if not x or not y:
        return False
    score = dialogue_echo_score(x, y)
    if score >= threshold:
        return True
    if len(x) >= 24 and (x in y or y in x):
        return True
    return False


def assistant_sampling_params(
    retry_assistant: int,
    *,
    base_temperature: float,
    base_top_p: float,
) -> Dict[str, float]:
    """
    Assistant 재시도 샘플링 정책

    assistant 생성이 같은 형식/역할 오류로 반복될 때, 재시도 횟수에 따라
    temperature와 top_p를 단계적으로 낮춰 출력 안정성을 높인다.
    기본 시도에서는 호출자가 준 base 값을 유지하고, 실패가 누적되면
    보수적인 파라미터로 점진적으로 수렴시킨다.

    Args:
        retry_assistant: 현재 assistant 재시도 누적 횟수.
        base_temperature: 0~2 범위의 기본 온도 값.
        base_top_p: 0~1 범위의 기본 top_p 값.

    Returns:
        Dict[str, float]: {"temperature": float, "top_p": float} 형태의 샘플링 파라미터.
    """
    if retry_assistant >= 12:
        return {"temperature": 0.36, "top_p": 0.68}
    if retry_assistant >= 7:
        return {"temperature": 0.44, "top_p": 0.74}
    if retry_assistant >= 3:
        return {"temperature": 0.50, "top_p": 0.78}
    return {"temperature": base_temperature, "top_p": base_top_p}


def enforce_assistant_two_line(text: str) -> str:
    """
    Assistant 출력 2줄 형식 강제

    모델 출력이 2줄 규칙(서술 1줄 + 대사 1줄)을 벗어나도, 검증 전에
    일관된 포맷으로 보정한다. 단순 침묵 대사로 덮어쓰지 않고 원문 정보를
    최대한 유지해 `asst_requires_two_lines`와 `asst_dia_silent` 반복을 줄인다.

    Args:
        text: 원본 assistant 출력 문자열.

    Returns:
        str: 반드시 2줄 형식으로 정규화된 assistant 텍스트.
    """
    src = strip_line_labels_multiline(text or "")
    lines = [x.strip() for x in src.splitlines() if x.strip()]
    if not lines:
        return "숨이 짧게 새어 나왔다.\n\"...\""

    if len(lines) == 1:
        only = lines[0]
        q = extract_single_quote(only)
        if q is not None and q.strip() not in ("...", "…"):
            return f"숨이 짧게 새어 나왔다.\n\"{q.strip()}\""
        return f"숨이 짧게 새어 나왔다.\n{force_single_quote_line(remove_all_quotes(only))}"

    narration = remove_all_quotes(lines[0]).strip() or "숨이 짧게 새어 나왔다."
    quote_line = ""
    for line in lines[1:]:
        q = extract_single_quote(line)
        if q is not None and q.strip() and q.strip() not in ("...", "…"):
            quote_line = f"\"{q.strip()}\""
            break
    if not quote_line:
        quote_line = force_single_quote_line(
            remove_all_quotes(lines[1] if len(lines) > 1 else lines[0])
        )
    return f"{narration}\n{quote_line}"


def first_sentence(text: str) -> str:
    """
    문자열에서 첫 문장만 추출

    후처리 단계에서 출력 길이 폭주를 막기 위해 문장 분리를 수행하고,
    첫 번째 문장만 남긴다.

    Args:
        text: 원본 문자열.

    Returns:
        str: 첫 문장 문자열.
    """
    t = normalize_space(text)
    if not t:
        return ""
    parts = [p.strip() for p in RE_SENTENCE_SPLIT.split(t) if p.strip()]
    if parts:
        return parts[0]
    return t


def enforce_two_line_one_sentence(
    text: str,
    *,
    default_narration: str = "잠시 숨을 고른다.",
) -> str:
    """
    2줄/줄당 1문장 형식 강제

    입력 텍스트를 `서술 1줄 + 대사 1줄` 형태로 정규화하고,
    각 줄은 첫 문장만 남겨 길이 과다 출력을 제어한다.

    Args:
        text: 원본 턴 텍스트.
        default_narration: 서술이 비었을 때 사용할 기본 문장.

    Returns:
        str: 2줄 고정 + 줄당 1문장으로 정규화된 텍스트.
    """
    src = strip_line_labels_multiline(text or "")
    lines = [x.strip() for x in src.splitlines() if x.strip()]
    if not lines:
        return f"{default_narration}\n\"...\""

    if len(lines) == 1:
        q = extract_single_quote(lines[0])
        dia_raw = q if q is not None else remove_all_quotes(lines[0])
        dia = first_sentence(dia_raw)
        narr = first_sentence(default_narration)
        return f"{narr}\n{force_single_quote_line(dia)}"

    narr_raw = remove_all_quotes(lines[0])
    q = extract_single_quote(lines[-1])
    dia_raw = q if q is not None else remove_all_quotes(lines[-1])

    narr = first_sentence(narr_raw) or first_sentence(default_narration)
    dia = first_sentence(dia_raw)
    return f"{narr}\n{force_single_quote_line(dia)}"


def strip_line_label(text: str) -> str:
    """입력 텍스트에서 `strip_line_label` 규칙의 불필요 요소를 제거한다."""
    t = (text or "").strip()
    if RE_BRACKET_LABEL.search(t):
        return ""
    t = re.sub(r"^\s*ACT\s*\d+\s*[:：]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", t)
    t = re.sub(r"^[^\s]{1,12}의\s*(서술|대사|행동|반응|대답|신체적\s*반응|행동/신체적\s*반응(?:\s*및\s*대사)?)\s*[:：]?\s*", "", t, flags=re.IGNORECASE)
    t = RE_COLON_LABEL.sub("", t).strip()
    t = RE_LINE_LABEL.sub("", t).strip()
    # Drop bare label-only lines like "사야의 행동/신체적 반응 및 대사"
    if re.fullmatch(r"[^\s]{1,12}의\s*(서술|대사|행동|반응|대답|신체적\s*반응|행동/신체적\s*반응(?:\s*및\s*대사)?)", t, flags=re.IGNORECASE):
        return ""
    return t


def strip_line_labels_multiline(text: str) -> str:
    """입력 텍스트에서 `strip_line_labels_multiline` 규칙의 불필요 요소를 제거한다."""
    lines = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        cleaned = strip_line_label(raw)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def strip_square_brackets(text: str) -> str:
    """입력 텍스트에서 `strip_square_brackets` 규칙의 불필요 요소를 제거한다."""
    if not text:
        return ""
    return RE_SQUARE_BRACKET.sub(r"\1", text)


def normalize_protagonist_refs(text: str, protagonist: str) -> str:
    """입력 텍스트를 `normalize_protagonist_refs` 규칙에 맞게 정규화한다."""
    if not text:
        return ""
    t = text
    if protagonist:
        t = RE_PROTAGONIST_WORD.sub(protagonist, t)
        # 이름 글자 사이에 공백이 끼는 출력(예: 카 즈키)을 원형으로 복원한다.
        spaced = r"\s*".join(map(re.escape, list(protagonist)))
        t = re.sub(spaced, protagonist, t)
    return t


def normalize_user_refs(text: str, player_name: str) -> str:
    """입력 텍스트를 `normalize_user_refs` 규칙에 맞게 정규화한다."""
    if not text:
        return ""
    name = player_name or "플레이어"
    t = text.replace("{{user}}", name)
    t = re.sub(r"(?<!\{)\{user\}(?!\})", name, t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\{)(사용자|플레이어|유저)(?!\})", name, t)
    t = re.sub(r"(?<!\{)\buser\b(?!\})", name, t, flags=re.IGNORECASE)
    if player_name:
        # 이름 글자 사이에 공백이 끼는 출력(예: 카 즈키)을 원형으로 복원한다.
        spaced = r"\s*".join(map(re.escape, list(player_name)))
        t = re.sub(spaced, player_name, t)
        for cand in PLAYER_NAME_CANDIDATES:
            if cand != player_name:
                t = t.replace(cand, player_name)
    return t


def strip_meta(text: str) -> str:
    """입력 텍스트에서 `strip_meta` 규칙의 불필요 요소를 제거한다."""
    if not text:
        return ""
    t = text.replace("[[", "").replace("]]", "")
    kept: List[str] = []
    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        if RE_META_BLOCK_LINE.search(line):
            continue
        if RE_META_BLOCK.search(line):
            continue
        if RE_META.search(line):
            continue
        if RE_POLICY_LEAK_LINE.search(line):
            continue
        kept.append(raw)
    t = "\n".join(kept)
    t = re.sub(r"\bACT\s*\d+\b", "", t, flags=re.IGNORECASE)
    # 세미콜론/대시를 마침표로 통일하고, 마침표 주변 공백을 보정
    t = t.replace(";", ".").replace("；", ".").replace("—", ".")
    t = re.sub(r"\s+\.", ".", t)  # 마침표 앞 공백 제거
    t = re.sub(r"(?<!\d)\.(?=[^\s\n\.\d])", ". ", t)  # 마침표 뒤 공백 1칸
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_genre(system_lore: str) -> str:
    """입력 텍스트에서 `parse_genre`에 해당하는 구조화 값을 파싱한다."""
    m = RE_GENRE_LINE.search(system_lore or "")
    if not m:
        return ""
    return normalize_space(m.group(1))


def is_player_subject(narration: str, player_name: str) -> bool:
    """문장이 플레이어 주어 형식인지 확인한다."""
    if not narration or not player_name:
        return False
    pattern = RE_PLAYER_SUBJECT.pattern.format(name=re.escape(player_name))
    return re.search(pattern, narration) is not None


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    """정수를 지정 범위로 클램프한다."""
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def safe_bool(x: Any) -> bool:
    """다양한 입력을 안전한 불리언 값으로 변환한다."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return False


def has_question(text: str) -> bool:
    """텍스트에 질문형 패턴이 포함되어 있는지 확인한다."""
    if not text:
        return False
    return "?" in text


RELATION_KEYWORDS = ["적대", "거리감", "친밀", "사랑"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """유사도 계산 규칙에 따라 `cosine_sim` 점수를 계산한다."""
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def last_role_text(messages: List[Dict[str, str]], role: str) -> str:
    """
    최근 role 메시지 1건 조회

    메시지 배열의 끝에서부터 탐색하여 지정 role의 최신 content를 반환한다.

    Args:
        messages: 채팅 메시지 목록.
        role: 조회할 역할명(user/assistant/system 등).

    Returns:
        str: 최신 role content, 없으면 빈 문자열.
    """
    for m in reversed(messages or []):
        if m.get("role") == role and m.get("content"):
            return m["content"]
    return ""


def extract_recent_history(messages: List[Dict[str, str]], max_turns: int = 8) -> str:
    """입력 데이터에서 `extract_recent_history`에 필요한 부분 문자열 또는 객체를 추출한다."""
    hist: List[str] = []
    for m in reversed(messages):
        if m["role"] == "system":
            continue
        hist.append(f"{m['role']}: {m['content']}")
        if len(hist) >= max_turns * 2:
            break

    if not hist:
        return "이전 대화 없음."
    return "\n".join(reversed(hist))


def extract_last_assistant_output(messages: List[Dict[str, str]]) -> str:
    """입력 데이터에서 `extract_last_assistant_output`에 필요한 부분 문자열 또는 객체를 추출한다."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "").strip() or "없음."
    return "없음."


def parse_protagonist_name(system_lore: str) -> str:
    """입력 텍스트에서 `parse_protagonist_name`에 해당하는 구조화 값을 파싱한다."""
    t = system_lore or ""
    m1 = re.search(r"당신은\s*이제\s*「([^」]{1,20})」", t)
    if m1:
        return m1.group(1).strip()

    m2 = re.search(r"(?:[-•]\s*)?이름\s*[:：]\s*([^\n]{1,24})", t)
    if m2:
        name = m2.group(1).strip()
        name = re.split(r"[\(\[]", name)[0].strip()
        if name:
            return name[:20]
    return "주인공"


def parse_player_name(system_lore: str) -> str:
    """입력 텍스트에서 `parse_player_name`에 해당하는 구조화 값을 파싱한다."""
    t = system_lore or ""
    m_block = re.search(r"4\.\s*플레이어 정의([\s\S]*?)(?:\n\s*5\.\s*|$)", t)
    if m_block:
        block = m_block.group(1)
        m_name = re.search(r"(?:[-•]\s*)?이름\s*[:：]\s*([^\n]{1,24})", block)
        if m_name:
            name = m_name.group(1).strip()
            name = re.sub(r"[「」『』《》\"']", "", name)
            name = re.split(r"[\(\[]", name)[0].strip()
            if name:
                return name[:20]

    m2 = re.search(r"플레이어의\s*이름은\s*([^\s,\.]{1,12})", t)
    if m2:
        name = re.sub(r"[「」『』《》\"']", "", m2.group(1).strip())
        return name[:20]

    m3 = re.search(r"플레이어(?:인|는)\s*[\"'「]?([^\s,\.]{1,12})[\"'」]?", t)
    if m3:
        name = re.sub(r"[「」『』《》\"']", "", m3.group(1).strip())
        return name[:20]

    return "플레이어"


def _extract_section7(text: str) -> str:
    """내부 헬퍼로 `_extract_section7` 계산 절차를 수행한다."""
    if not text:
        return ""
    m = re.search(r"\n\s*7\.\s*가장\s*최근\s*상호작용.*?(?=\n\s*\d+\.\s|\Z)", text, flags=re.DOTALL)
    return m.group(0) if m else ""


def parse_relation_status(system_lore: str) -> str:
    """입력 텍스트에서 `parse_relation_status`에 해당하는 구조화 값을 파싱한다."""
    t = system_lore or ""
    sec7 = _extract_section7(t)
    if sec7:
        m7 = re.search(r"관계\s*상태\s*[:：]\s*([^\n]{1,24})", sec7)
        if m7:
            line = m7.group(1)
            for kw in RELATION_KEYWORDS:
                if kw in line:
                    return kw
    m = re.search(r"관계\s*상태\s*[:：]\s*([^\n]{1,24})", t)
    if m:
        line = m.group(1)
        for kw in RELATION_KEYWORDS:
            if kw in line:
                return kw
    return ""


def parse_protagonist_speech_formal(system_lore: str) -> Optional[bool]:
    """
    return:
      - True  : 존댓말 우선
      - False : 반말 우선
      - None  : 판별 불가
    """
    t = system_lore or ""
    m6 = re.search(r"6\.\s*발화와\s*분위기\s*규칙([\s\S]*?)(?:\n\s*7\.\s*|$)", t)
    block = m6.group(1) if m6 else t
    if re.search(r"(기본\s*말투|말투\s*범주|말투)\s*[:：]?\s*존댓말", block):
        return True
    if re.search(r"(기본\s*말투|말투\s*범주|말투)\s*[:：]?\s*반말", block):
        return False

    if "존댓말" in block:
        return True
    if "반말" in block:
        return False
    return None


def parse_sexual_condition_relation(system_lore: str) -> str:
    """입력 텍스트에서 `parse_sexual_condition_relation`에 해당하는 구조화 값을 파싱한다."""
    t = system_lore or ""
    m5 = re.search(r"5\.\s*관계\s*및\s*변화\s*규칙([\s\S]*?)(?:\n\s*6\.\s*|$)", t)
    block = m5.group(1) if m5 else t
    cond = ""
    m = re.search(r"성행위\s*가능\s*여부\s*True\s*시\s*조건[^\n]*", block)
    if m:
        cond = m.group(0)
    else:
        m2 = re.search(r"성행위\s*가능\s*여부\s*True[^\n]*", block)
        if m2:
            cond = m2.group(0)
    if not cond:
        lines = block.splitlines()
        for i, line in enumerate(lines):
            if "성행위" in line and "True" in line:
                if i + 1 < len(lines):
                    cond = lines[i + 1]
                else:
                    cond = line
                break
    for kw in RELATION_KEYWORDS:
        if kw in cond:
            return kw
    return ""


def resolve_allow_sexual(flag_value: Any, system_lore: str) -> bool:
    """
    성행위 허용 플래그 확정

    FSM 플래그와 시나리오북 텍스트를 함께 해석해
    sexual 분기 가능 여부를 최종 bool 값으로 결정한다.

    Args:
        flag_value: FSM이 전달한 allow_sexual 값(bool 또는 특수 문자열).
        system_lore: 시나리오북 원문.

    Returns:
        bool: 성행위 허용 여부.
    """
    if isinstance(flag_value, bool):
        return flag_value

    if isinstance(flag_value, str) and flag_value.strip().lower() == "scenario_defined":
        # MOD: FSM 엔진이 global_flags로 allow_sexual을 강제 false/true 할 수 있으나,
        # 여기서는 scenario_defined일 때 시나리오북 텍스트에서도 fallback 판정이 되게 유지.
        t = system_lore or ""
        m = re.search(r"(성행위\s*허용|성행위\s*가능\s*여부|allow_sexual)\s*[:：]\s*(true|false)", t, flags=re.IGNORECASE)
        if m:
            return m.group(2).lower() == "true"
        if re.search(r"\b(nsfl|nsfw|adult|18\+)\b", t, flags=re.IGNORECASE):
            return True
        if re.search(r"(장르\s*[:：]\s*헨타이|무삭제|성인물)", t):
            return True
        return False

    return safe_bool(flag_value)
