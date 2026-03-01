from __future__ import annotations
import re
from typing import Any, List, Optional

RE_QUOTE = re.compile(r"\"([^\"]+)\"")


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


RE_USER_PLACEHOLDER = re.compile(r"\{\{user\}\}")


RE_LINE_LABEL = re.compile(
    r"^\s*[-•·]?\s*(서술|대사|행동|반응|대답|정보\s*공개|Action|Narration|Dialogue|Speech|Player'?s\s*Action)\s*[:：]\s*",
    flags=re.IGNORECASE,
)


RE_FORMAL_ENDING = re.compile(r"(습니다|합니다|됩니|십니다|세요|이에요|예요|에요|니다)(?=[\s\.\!\?…]|$)")


RE_BRACKET_LABEL = re.compile(r"^\s*[\[\(].*?(서술|대사|행동|출력|정보|첫\s*번째|첫째|둘째).*?[\]\)]\s*$", flags=re.IGNORECASE)


RE_COLON_LABEL = re.compile(r"^\s*[:：]+\s*")


RE_PROTAGONIST_WORD = re.compile(r"주인공")


RE_SQUARE_BRACKET = re.compile(r"\[([^\[\]]+)\]")


PLAYER_NAME_CANDIDATES = ["하야토", "카즈키", "소마"]


RE_GENRE_LINE = re.compile(r"장르\s*:\s*([^\n]+)")


RE_PLAYER_SUBJECT = re.compile(r"^({name})(이|가|은|는)\b")


RE_SPEAKER_LABEL = re.compile(r"^\s*({name})\s*[:：]\s*")


def normalize_space(text: str) -> str:
    """입력 문자열을 규칙에 맞게 정규화한다."""
    return re.sub(r"\s+", " ", (text or "").strip())


def quotes_balanced(text: str) -> bool:
    """큰따옴표 개수가 짝수인지 확인한다."""
    return (text or "").count('"') % 2 == 0


def extract_quotes(text: str) -> List[str]:
    """텍스트 또는 메시지에서 필요한 항목을 추출한다."""
    return [s.strip() for s in RE_QUOTE.findall(text or "") if s.strip()]


def extract_single_quote(text: str) -> Optional[str]:
    """텍스트 또는 메시지에서 필요한 항목을 추출한다."""
    segs = extract_quotes(text)
    return segs[0] if len(segs) == 1 else None


def has_speaker_label(line: str, names: List[str]) -> bool:
    """조건 충족 여부를 판정해 불리언 값을 반환한다."""
    if not line or not names:
        return False
    for n in names:
        if not n:
            continue
        if re.match(RE_SPEAKER_LABEL.pattern.format(name=re.escape(n)), line):
            return True
    return False


def force_single_quote_line(text: str) -> str:
    """
    대사 1줄만 강제 (큰따옴표 1쌍)
    """
    t = (text or "").strip()
    if t.count('"') >= 2:
        first = t.find('"')
        last = t.rfind('"')
        inner = t[first + 1:last].strip()
        return f"\"{inner}\"" if inner else "\"...\""
    return f"\"{t}\"" if t else "\"...\""


def remove_all_quotes(text: str) -> str:
    """문자열에서 모든 큰따옴표를 제거한다."""
    return (text or "").replace('"', "").strip()


def strip_line_label(text: str) -> str:
    """한 줄 텍스트의 화자 라벨을 제거한다."""
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
    """여러 줄 텍스트의 화자 라벨을 제거한다."""
    lines = []
    for raw in (text or "").splitlines():
        if not raw.strip():
            continue
        cleaned = strip_line_label(raw)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def strip_square_brackets(text: str) -> str:
    """대괄호 메타 표기를 제거한다."""
    if not text:
        return ""
    return RE_SQUARE_BRACKET.sub(r"\1", text)


def normalize_protagonist_refs(text: str, protagonist: str) -> str:
    """입력 문자열을 규칙에 맞게 정규화한다."""
    if not text:
        return ""
    if protagonist:
        return RE_PROTAGONIST_WORD.sub(protagonist, text)
    return text


def normalize_user_refs(text: str, player_name: str) -> str:
    """입력 문자열을 규칙에 맞게 정규화한다."""
    if not text:
        return ""
    name = player_name or "플레이어"
    t = text.replace("{{user}}", name)
    t = re.sub(r"(?<!\{)\{user\}(?!\})", name, t, flags=re.IGNORECASE)
    t = re.sub(r"(?<!\{)(사용자|플레이어|유저)(?!\})", name, t)
    t = re.sub(r"(?<!\{)\buser\b(?!\})", name, t, flags=re.IGNORECASE)
    if player_name:
        for cand in PLAYER_NAME_CANDIDATES:
            if cand != player_name:
                t = t.replace(cand, player_name)
    return t


def strip_meta(text: str) -> str:
    """메타 문구를 제거하고 대화 본문만 남긴다."""
    if not text:
        return ""
    t = text.replace("[[", "").replace("]]", "")
    # Remove whole meta/policy leakage lines first.
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
    t = RE_META.sub("", t)
    t = re.sub(r"\bACT\s*\d+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_genre(system_lore: str) -> str:
    """입력 텍스트에서 필요한 값을 파싱해 반환한다."""
    m = RE_GENRE_LINE.search(system_lore or "")
    if not m:
        return ""
    return normalize_space(m.group(1))


def is_player_subject(narration: str, player_name: str) -> bool:
    """조건 충족 여부를 판정해 불리언 값을 반환한다."""
    if not narration or not player_name:
        return False
    pattern = RE_PLAYER_SUBJECT.pattern.format(name=re.escape(player_name))
    return re.search(pattern, narration) is not None


def clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    """정수값을 주어진 최소·최대 범위로 제한한다."""
    try:
        v = int(x)
    except Exception:
        return default
    return max(lo, min(hi, v))


def safe_bool(x: Any) -> bool:
    """다양한 입력 타입을 안전하게 불리언으로 변환한다."""
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
    """조건 충족 여부를 판정해 불리언 값을 반환한다."""
    if not text:
        return False
    return "?" in text


def normalize_scenario_text(text: str) -> str:
    """
    생성 결과 후처리:
    - 별표(*) 제거
    - 특수 괄호 제거 (《》)
    - 연속 공백 정리
    """
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

    # 불필요한 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def cut_after_section7(text: str) -> str:
    """
    7번 섹션 이후 추가 안내/섹션이 붙는 경우를 잘라낸다.
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


__all__ = [
    "normalize_space",
    "quotes_balanced",
    "extract_quotes",
    "extract_single_quote",
    "has_speaker_label",
    "force_single_quote_line",
    "remove_all_quotes",
    "strip_line_label",
    "strip_line_labels_multiline",
    "strip_square_brackets",
    "normalize_protagonist_refs",
    "normalize_user_refs",
    "strip_meta",
    "parse_genre",
    "is_player_subject",
    "clamp_int",
    "safe_bool",
    "has_question",
    "normalize_scenario_text",
    "cut_after_section7",
]
