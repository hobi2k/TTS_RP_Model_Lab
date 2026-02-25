from __future__ import annotations
import re
from typing import Optional

try:
    from .utils import (
        normalize_space, quotes_balanced, extract_quotes, extract_single_quote,
        has_speaker_label, strip_line_labels_multiline,
    )
except ImportError:
    from utils import (
        normalize_space, quotes_balanced, extract_quotes, extract_single_quote,
        has_speaker_label, strip_line_labels_multiline,
    )

# NOTE: validator constants/functions extracted from v5 cores.

RE_META = re.compile(
    r"(\*+|프롬프트|요청하신|요청된|"
    r"시나리오북|헌법|검증|메타|"
    r"role\s*check|system\s*check|langgraph|fsm|phase|노드|"
    r"그래프|평가|json|prompt|instruction|guideline)",
    flags=re.IGNORECASE,
)


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


RE_PAREN = re.compile(r"[\(\)]")


RE_FORMAL_ENDING = re.compile(r"(습니다|합니다|됩니|십니다|세요|이에요|예요|에요|니다)(?=[\s\.\!\?…]|$)")


RE_SPEAKER_LABEL = re.compile(r"^\s*({name})\s*[:：]\s*")


RE_NARR_DIALOGUE_ENDING = re.compile(
    r"(요|죠|네요|군요|할게|줄게|가자|하자|해줘|알겠어|알았어|걱정\s*마)$"
)


PROTAGONIST_NAMES = ["사야", "마이", "코하루"]


PLAYER_NAMES = ["하야토", "카즈키", "소마"]


RE_ASCII_WORD = re.compile(r"\b[A-Za-z]{4,}\b")


def _narration_has_dialogue_tone(narr: str, protagonist: str, player_name: str) -> bool:
    """서술 줄에 대사체가 섞였는지 검사한다.

    검출 규칙:
    - 물음표/느낌표 존재
    - 이름 직접 호명(예: `소마, ...`)
    - 문장 끝이 대사체 어미/표현으로 끝남
    """
    if not narr:
        return False

    if "?" in narr or "!" in narr:
        return True

    names = [n for n in (protagonist, player_name) if n]
    if names:
        name_pat = "|".join(re.escape(n) for n in names)
        if re.search(rf"\b(?:{name_pat})\s*,", narr):
            return True

    parts = [x.strip() for x in re.split(r"[\.…]\s*", narr) if x.strip()]
    for p in parts:
        if RE_NARR_DIALOGUE_ENDING.search(p):
            return True
    return False


def is_valid_scenario_book(text: str) -> bool:
    """시나리오북 기본 구조 유효성을 빠르게 판정한다.

    검증 기준:
    - 최소 길이 충족
    - 주인공/플레이어 이름이 각각 정확히 1종류만 포함
    - 0~7 섹션 헤더 존재
    """
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

    # Require sections 0..7
    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return False

    return True


def invalid_reason(text: str) -> str:
    """시나리오북 유효성 실패 원인을 코드 문자열로 반환한다."""
    t = (text or "").strip()
    if len(t) < 350:
        return "too_short"
    # Require sections 0..7
    for i in range(0, 8):
        if re.search(rf"^\s*{i}\.\s+", t, flags=re.MULTILINE) is None:
            return f"section_missing_{i}"
    protag = {n for n in PROTAGONIST_NAMES if n in t}
    if len(protag) != 1:
        return f"protagonist_count={len(protag)}"
    player = {n for n in PLAYER_NAMES if n in t}
    if len(player) != 1:
        return f"player_count={len(player)}"
    return "unknown"


def _has_forbidden_chars(text: str) -> bool:
    """출력 금지 문자(괄호/별표) 포함 여부를 반환한다."""
    if not text:
        return False
    if RE_PAREN.search(text):
        return True
    if "*" in text:
        return True
    return False


def user_invalid_reason(text: str, *, player_name: str = "", protagonist: str = "") -> Optional[str]:
    """사용자 턴 형식/정책 위반 사유를 반환한다.

    반환값:
    - 위반 없음: `None`
    - 위반 있음: 구체적인 에러 코드 문자열
    """
    t = (text or "").strip()
    if not t:
        return "user_empty"
    if _has_forbidden_chars(t):
        return "user_forbidden_chars"
    if RE_COLON_ANY.search(t):
        return "user_colon_char"
    if not quotes_balanced(t):
        return "user_unbalanced_quotes"
    q = extract_quotes(t)
    if len(q) != 1:
        return "user_requires_single_quote"
    if q[0].strip() in ("...", "…"):
        return "user_quote_silent"
    if RE_DIALOGUE_COLON_HEAD.search(q[0]):
        return "user_dialogue_colon_head"
    if RE_DIALOGUE_LABEL_PREFIX.search(q[0]):
        return "user_dialogue_label_prefix"
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if any(RE_POLICY_LEAK_LINE.search(x) for x in lines):
        return "user_policy_leak"
    if lines and has_speaker_label(lines[0], [player_name, protagonist]):
        return "user_speaker_label"
    if len(lines) > 2:
        return "user_too_many_lines"
    if extract_single_quote(lines[-1]) is None:
        return "user_last_line_must_be_quote"
    if lines and RE_FORMAL_ENDING.search(lines[0]):
        return "user_action_formal"
    return None


def assistant_invalid_reason_integrated(
    text: str, *, protagonist: str = "", player_name: str = "", prefer_formal: Optional[bool] = None
) -> Optional[str]:
    """통합 assistant 출력(2줄) 전용 유효성 검사기.

    기대 형식:
    - 1줄: 서술(따옴표 없음)
    - 2줄: 대사(큰따옴표 1쌍)
    """
    t = (text or "").strip()
    if not t:
        return "asst_empty"
    if _has_forbidden_chars(t):
        return "asst_forbidden_chars"
    if RE_COLON_ANY.search(t):
        return "asst_colon_char"
    if RE_META.search(t) or RE_META_BLOCK.search(t):
        return "asst_meta"
    if any(RE_META_BLOCK_LINE.search(x) for x in t.splitlines() if x.strip()):
        return "asst_meta"
    if not quotes_balanced(t):
        return "asst_unbalanced_quotes"

    lines = [x.strip() for x in strip_line_labels_multiline(t).splitlines() if x.strip()]
    if len(lines) != 2:
        return "asst_requires_two_lines"
    if any(RE_POLICY_LEAK_LINE.search(x) for x in lines):
        return "asst_policy_leak"

    narr = lines[0]
    dia = lines[1]
    if has_speaker_label(narr, [protagonist, player_name]):
        return "asst_speaker_label"
    if has_speaker_label(dia, [protagonist, player_name]):
        return "asst_speaker_label"

    if RE_FORMAL_ENDING.search(narr):
        return "asst_narr_formal"
    if '"' in narr:
        return "asst_narr_has_quote"
    if _narration_has_dialogue_tone(narr, protagonist, player_name):
        return "asst_narr_dialogue_tone"
    q = extract_quotes(dia)
    if len(q) != 1:
        return "asst_dia_requires_single_quote"
    if q[0].strip() in ("...", "…"):
        return "asst_dia_silent"
    if RE_DIALOGUE_COLON_HEAD.search(q[0]):
        return "asst_dia_colon_head"
    if RE_DIALOGUE_LABEL_PREFIX.search(q[0]):
        return "asst_dia_label_prefix"
    if re.fullmatch(r"\s*\d+\s*\.?\s*", q[0].strip()):
        return "asst_dialogue_numeric"
    # Dialogue must not be narration (e.g., "사야는 ...했다")
    if protagonist:
        dia_txt = q[0].strip()
        if protagonist in dia_txt and re.search(r"(했다|되었다|있었다|이었다|였다)\b", dia_txt):
            return "asst_dialogue_narration"
    if len(normalize_space(narr)) < 6:
        return "asst_narr_too_short"
    return None


def crisis_invalid_reason(text: str) -> Optional[str]:
    """CRISIS 응답 전용 형식 위반 사유를 반환한다."""
    t = (text or "").strip()
    if not t:
        return "crisis_empty"
    if _has_forbidden_chars(t):
        return "crisis_forbidden_chars"
    if RE_COLON_ANY.search(t):
        return "crisis_colon_char"
    if not quotes_balanced(t):
        return "crisis_unbalanced_quotes"
    # 너무 길면 컷(안전장치)
    if len(t) > 800:
        return "crisis_too_long"
    return None


__all__ = [
    "is_valid_scenario_book",
    "invalid_reason",
    "_has_forbidden_chars",
    "user_invalid_reason",
    "assistant_invalid_reason_integrated",
    "crisis_invalid_reason",
]
