"""
data/v7_qwen_multiturn_gen.py

LangGraph-orchestrated VN multi-turn generator (FSM-driven)
(ROLE-SEPARATION HARDENED, NO "SIMPLIFY" REGRESSION)

- ScenarioBook (jsonl) -> Multi-turn RP (jsonl)
- FSM: QwenFSMEngine (v7_qwen_fsm_engine)
- Orchestration: LangGraph controls the whole loop (real)
- Evaluation: LLM(JSON) scoring -> FSM inputs (no keyword rule scoring)
- Memory: explicit summary + EmbeddingMemory (BGE-m3-ko) anti-repeat

사용 예시:
uv run data/v7_qwen_multiturn_gen.py \
  --model_path data/generator/Tri-7B \
  --scenario_path /mnt/d/rp_data/qwen/rp_scenario.jsonl \
  --out_path /mnt/d/rp_data/qwen/rp_datum.jsonl \
  --fsm_path data/version_3/v7_qwen/state_fsm.yaml \
  --action_fsm_path data/version_3/v7_qwen/action_fsm.yaml \
  --turns 3 \
  --use_4bit

핵심 변경:
- 플레이어는 시나리오북에서 추출한 "플레이어 이름"을 사용한다.
- 주인공은 시나리오북에서 추출한 "주인공 이름"을 사용한다.
- 출력에 별표(*)/장식문자/괄호()/메타(FSM, system 등) 언급을 강하게 차단한다.
- EmbeddingMemory.is_repetitive()는 keyword-only이므로 절대 positional로 넘기지 않는다.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple

from backend import generate_text
from utils import (
    extract_quotes,
    extract_single_quote,
    force_single_quote_line,
    has_speaker_label,
    normalize_space,
    remove_all_quotes,
    quotes_balanced,
    strip_line_labels_multiline,
)



RE_QUOTE = re.compile(r"\"([^\"]+)\"")
RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)

# 별표(*) 및 다양한 장식문자/메타 단어를 더 강하게 차단하도록 확장
RE_META = re.compile(
    r"(\*+|프롬프트|요청하신|요청된|"
    r"시나리오북|헌법|검증|메타|"
    r"role\s*check|system\s*check|langgraph|fsm|phase|노드|"
    r"그래프|평가|json|prompt|instruction|guideline)",
    flags=re.IGNORECASE,
)

# 메타 전개/선택지류 문구 차단
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
RE_POLICY_LEAK_PHRASE = re.compile(
    r"(장르와\s*행위/?상태\s*반영|형식\s*골격|판정\s*기준|말투\s*규칙|출력\s*규칙|"
    r"단계\s*힌트|관계\s*상태|관계\s*변화\s*의도|현재\s*국면|FSM|"
    r"(적대|거리감|친밀|사랑)\s*단계|사랑의\s*관계|관계로\s*연결)",
    flags=re.IGNORECASE,
)

# 괄호() 금지 규칙을 validator에서 강제하기 위한 패턴
RE_PAREN = re.compile(r"[\(\)]")

# 플레이어 이름 참조(legacy)
RE_USER_PLACEHOLDER = re.compile(r"\{\{user\}\}")
RE_LINE_LABEL = re.compile(
    r"^\s*[-•·]?\s*(서술|대사|행동|반응|대답|정보\s*공개|Action|Narration|Dialogue|Speech|Player'?s\s*Action)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
RE_FORMAL_ENDING = re.compile(r"(습니다|합니다|됩니|십니다|세요|이에요|예요|에요|니다)(?=[\s\.\!\?…]|$)")
RE_POLITE_MARK_ANY = re.compile(r"(?:[가-힣]{2,}요)(?=[,\s\.\!\?…]|$)")
RE_BRACKET_LABEL = re.compile(r"^\s*[\[\(].*?(서술|대사|행동|출력|정보|첫\s*번째|첫째|둘째).*?[\]\)]\s*$", flags=re.IGNORECASE)
RE_COLON_LABEL = re.compile(r"^\s*[:：]+\s*")
RE_PROTAGONIST_WORD = re.compile(r"주인공")
RE_SQUARE_BRACKET = re.compile(r"\[([^\[\]]+)\]")
PLAYER_NAME_CANDIDATES = ["하야토", "카즈키", "소마"]
RE_GENRE_LINE = re.compile(r"장르\s*:\s*([^\n]+)")
RE_PLAYER_SUBJECT = re.compile(r"^({name})(이|가|은|는)\b")
RE_SPEAKER_LABEL = re.compile(r"^\s*({name})\s*[:：]\s*")
RE_DANGLING_DIALOGUE_END = re.compile(
    r"(?:"
    r"[가-힣A-Za-z0-9]+(?:을|를|이|가|은|는|와|과|에게|한테|께)"
    r"|그리고|하지만|그런데|그래서|그러니까|혹은|또는|만약"
    r")$"
)


def _has_player_third_person_ref(dialogue: str, player_name: str) -> bool:
    """주인공 대사에서 플레이어를 3인칭으로 지칭하는 패턴을 탐지한다."""
    if not dialogue or not player_name:
        return False
    name = re.escape(player_name)
    particle_ref = re.search(
        rf"{name}\s*(이|가|을|를|에게|한테|께)(?=[^가-힣]|$)",
        dialogue,
    )
    return bool(particle_ref)


def llm_classify_dialogue_style_local(
    model,
    tokenizer,
    dialogue_text: str,
) -> str:
    """LLM 호출로 `llm_classify_dialogue_style_local` 판정 결과를 계산한다."""
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 말투 분류기다. "
                "입력 대사의 화자 말투를 다음 중 하나로만 분류하라: "
                "FORMAL 또는 INFORMAL. "
                "존댓말이면 FORMAL, 반말이면 INFORMAL로 분류한다. "
                "문장 끝 일부만 존댓말이고 나머지가 반말이면 INFORMAL로 분류한다. "
                "반드시 라벨 1개만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 FORMAL 또는 INFORMAL 중 하나만 출력하라. "
                "그 외 텍스트/설명 금지.\n"
                f"[대사]\n{dialogue_text}"
            ),
        },
    ]
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "INFORMAL" in s:
        return "informal"
    if "FORMAL" in s:
        return "formal"
    if "반말" in raw:
        return "informal"
    if "존댓말" in raw:
        return "formal"
    return "unknown"


def llm_speech_style_mismatch_local(
    model,
    tokenizer,
    dialogue_text: str,
    prefer_formal: Optional[bool],
) -> bool:
    """LLM 호출로 `llm_speech_style_mismatch_local` 판정 결과를 계산한다."""
    if prefer_formal is None:
        return False
    style = llm_classify_dialogue_style_local(model, tokenizer, dialogue_text)
    if style == "unknown":
        return True
    if prefer_formal and style != "formal":
        return True
    if (prefer_formal is False) and style != "informal":
        return True
    return False


def llm_narration_plain_mismatch_local(
    model,
    tokenizer,
    narration_text: str,
) -> bool:
    """서술 1줄이 평어체가 아닌 경우를 LLM 말투 분류로 판정한다."""
    if not normalize_space(narration_text):
        return False
    style = llm_classify_dialogue_style_local(model, tokenizer, narration_text)
    return style != "informal"


def llm_dialogue_quality_fail_local(
    model,
    tokenizer,
    *,
    system_lore: str,
    relation_status: str,
    action_state: str,
    player_name: str,
    user_text: str,
    assistant_text: str,
) -> bool:
    """LLM 호출로 `llm_dialogue_quality_fail_local` 판정 결과를 계산한다."""
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 대화 품질 판정기다. "
                "assistant의 응답이 사람 간 상호작용처럼 자연스럽고, "
                "직전 user 발화와 맥락적으로 연결되며, "
                "장면이 의미 있게 전진하면 PASS, 아니면 FAIL만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 PASS 또는 FAIL 중 하나만 출력하라.\n"
                f"[관계 상태]\n{relation_status}\n\n"
                f"[행위/사건 상태]\n{action_state}\n\n"
                f"[플레이어 이름]\n{player_name}\n\n"
                f"[시나리오북]\n{system_lore}\n\n"
                f"[user]\n{user_text}\n\n"
                f"[assistant]\n{assistant_text}"
            ),
        },
    ]
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "PASS" in s:
        return False
    if "FAIL" in s:
        return True
    return True


def llm_role_conflict_fail_local(
    model,
    tokenizer,
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    user_text: str,
    assistant_text: str,
) -> bool:
    """생성 결과의 역할 혼동(주인공/플레이어 전도, 대필)을 LLM으로 판정한다."""
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 역할 충돌 판정기다. "
                "assistant 출력이 시나리오북의 주인공 관점을 유지하면 PASS, "
                "플레이어 대사를 대신 쓰거나 화자가 뒤바뀌면 FAIL만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 PASS 또는 FAIL 중 하나만 출력하라.\n"
                f"[시나리오북]\n{system_lore}\n\n"
                f"[주인공 이름]\n{protagonist}\n\n"
                f"[플레이어 이름]\n{player_name}\n\n"
                f"[user]\n{user_text}\n\n"
                f"[assistant]\n{assistant_text}\n\n"
                "[판정 기준]\n"
                "- assistant가 플레이어 입장에서 말하면 FAIL\n"
                "- assistant가 플레이어 대사/행동을 대필하면 FAIL\n"
                "- assistant가 주인공이 자기 이름을 호격으로 부르면 FAIL\n"
                "- 그 외 주인공 관점 유지면 PASS"
            ),
        },
    ]
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "PASS" in s:
        return False
    if "FAIL" in s:
        return True
    return True


def llm_user_role_conflict_fail_local(
    model,
    tokenizer,
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    user_text: str,
) -> bool:
    """
    user 출력의 역할 혼동을 LLM으로 판정한다.

    플레이어 턴인데 주인공 시점으로 서술하거나, 플레이어 자신을 타인처럼 서술하면 FAIL로 본다.

    Args:
        model: 로컬 생성 모델 인스턴스.
        tokenizer: 로컬 토크나이저 인스턴스.
        system_lore: 시나리오북 원문.
        protagonist: 주인공 이름.
        player_name: 플레이어 이름.
        user_text: 검사 대상 user 출력.

    Returns:
        bool: 역할 혼동이면 True, 아니면 False.
    """
    lines = [ln.strip() for ln in (user_text or "").splitlines() if ln.strip()]
    narr_line = lines[0] if lines else (user_text or "").strip()
    dia_line = lines[1] if len(lines) > 1 else ""

    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 한국어 역할 충돌 판정기다. "
                "판정 대상은 user 출력의 서술 1줄과 대사 1줄이다. "
                "두 줄이 모두 플레이어 시점을 유지하면 PASS, 아니면 FAIL만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "반드시 PASS 또는 FAIL 중 하나만 출력하라.\n"
                f"[시나리오북]\n{system_lore}\n\n"
                f"[주인공 이름]\n{protagonist}\n\n"
                f"[플레이어 이름]\n{player_name}\n\n"
                f"[user 전체]\n{user_text}\n\n"
                f"[판정 대상: 서술 1줄]\n{narr_line}\n\n"
                f"[판정 대상: 대사 1줄]\n{dia_line}\n\n"
                "[판정 기준]\n"
                "- 서술 1줄은 플레이어 시점이면 PASS이며, 주어 생략은 허용한다\n"
                f"- 서술 1줄이 '{protagonist}'를 주어로 두면 FAIL\n"
                f"- 서술 1줄에서 '{player_name}를/을/에게/한테/께/와/과'처럼 플레이어를 타인처럼 목적어 처리하면 FAIL\n"
                f"- 대사 1줄이 {protagonist}의 발화처럼 보이면 FAIL\n"
                f"- 대사 1줄이 {player_name} 시점이 아니라 {protagonist} 시점 고백/판단이면 FAIL\n"
                "- 그 외 플레이어 관점 유지면 PASS"
            ),
        },
    ]
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_msgs,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    # user 역할 혼동 판정은 과검출이 치명적이므로
    # 명시적으로 FAIL이 확인될 때만 실패 처리한다.
    if "FAIL" in s:
        return True
    if "PASS" in s:
        return False
    return False


def _has_forbidden_chars(text: str) -> bool:
    """내부 헬퍼로 `_has_forbidden_chars` 계산 절차를 수행한다."""
    if not text:
        return False
    if RE_PAREN.search(text):
        return True
    if "*" in text:
        return True
    return False


def _has_policy_leak_phrase(text: str) -> bool:
    """
    정책 문구 누출 여부를 판정한다.

    콜론 기반 정책 라벨이 아니어도, 프롬프트 지시 문구가
    본문/대사에 노출되면 메타 누출로 처리한다.
    """
    return bool(RE_POLICY_LEAK_PHRASE.search(text or ""))


def _looks_truncated_dialogue(text: str) -> bool:
    """
    대사 토큰 컷(미완성 끝맺음) 여부를 판정한다.

    조사/접속사로 끝나는 대사를 잘린 출력으로 간주해 재시도시킨다.
    """
    t = normalize_space(remove_all_quotes(text))
    if len(t) < 8:
        return False
    return bool(RE_DANGLING_DIALOGUE_END.search(t))


def user_invalid_reason(text: str, *, player_name: str = "", protagonist: str = "") -> Optional[str]:
    """
    user 출력 유효성 판정

    user 턴이 형식(2줄/대사 인용), 금지 문자, 역할 규칙을 만족하는지 검사하고,
    실패 시 세부 오류 코드를 반환한다.

    Args:
        text: 검사할 user 출력 원문.
        player_name: 플레이어 이름.
        protagonist: 주인공 이름.

    Returns:
        Optional[str]: 오류 코드, 통과 시 None.
    """
    t = (text or "").strip()
    if not t:
        return "user_empty"
    # 괄호/별표 금지 강제
    if _has_forbidden_chars(t):
        return "user_forbidden_chars"
    if RE_COLON_ANY.search(t):
        return "user_colon_char"
    # 플레이어 이름이 누락될 수 있어도 형식 위반은 아니다.
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
    if lines and protagonist:
        if re.search(rf"^\s*{re.escape(protagonist)}\s*(?:이|가|은|는)\b", lines[0]):
            return "user_role_conflict"
    if any(RE_POLICY_LEAK_LINE.search(x) for x in lines):
        return "user_policy_leak"
    if _has_policy_leak_phrase(t):
        return "user_policy_leak"
    if lines and has_speaker_label(lines[0], [player_name, protagonist]):
        return "user_speaker_label"
    if len(lines) > 2:
        return "user_too_many_lines"
    if extract_single_quote(lines[-1]) is None:
        return "user_last_line_must_be_quote"
    if _looks_truncated_dialogue(q[0]):
        return "user_dialogue_truncated"
    if lines and RE_FORMAL_ENDING.search(lines[0]):
        return "user_action_formal"
    return None


def assistant_invalid_reason_integrated(
    text: str, *, protagonist: str = "", player_name: str = "", prefer_formal: Optional[bool] = None
) -> Optional[str]:
    """
    assistant 통합 유효성 판정

    서술+대사 2줄 형식, 역할 일관성, 메타 누수, 말투/침묵 규칙을 한 번에 검사하고
    첫 번째 실패 원인을 오류 코드로 반환한다.

    Args:
        text: 검사할 assistant 출력 원문.
        protagonist: 주인공 이름.
        player_name: 플레이어 이름.
        prefer_formal: 주인공 대사 말투 강제값(True=존댓말, False=반말, None=미강제).

    Returns:
        Optional[str]: 오류 코드, 통과 시 None.
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
    if _has_policy_leak_phrase(t):
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
    dia_txt = q[0].strip()
    if _looks_truncated_dialogue(dia_txt):
        return "asst_dialogue_truncated"
    if _has_player_third_person_ref(dia_txt, player_name):
        return "asst_player_third_person"
    # Dialogue must not be narration (e.g., "사야는 ...했다")
    if protagonist:
        if protagonist in dia_txt and re.search(r"(했다|되었다|있었다|이었다|였다)\b", dia_txt):
            return "asst_dialogue_narration"
    if len(normalize_space(narr)) < 6:
        return "asst_narr_too_short"
    return None


def crisis_invalid_reason(text: str, *, player_name: str = "") -> Optional[str]:
    """
    crisis 출력 유효성 판정

    crisis 노드 출력에 대해 금지 문자, 따옴표 균형, 과도한 길이 등을 검사해
    위험한 출력이 데이터에 들어가지 않도록 막는다.

    Args:
        text: 검사할 crisis 출력 문자열.

    Returns:
        Optional[str]: 오류 코드, 통과 시 None.
    """
    t = (text or "").strip()
    if not t:
        return "crisis_empty"
    if _has_forbidden_chars(t):
        return "crisis_forbidden_chars"
    if RE_COLON_ANY.search(t):
        return "crisis_colon_char"
    if not quotes_balanced(t):
        return "crisis_unbalanced_quotes"
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if len(lines) != 2:
        return "crisis_requires_two_lines"
    narr = lines[0]
    dia = lines[1]
    if narr.startswith('"') and narr.endswith('"'):
        return "crisis_no_narr_line"
    if not (dia.startswith('"') and dia.endswith('"')):
        return "crisis_requires_single_quote"
    dia_txt = remove_all_quotes(dia).strip()
    if _has_player_third_person_ref(dia_txt, player_name):
        return "crisis_player_third_person"
    return None


def detect_sexual_request(user_text: str) -> bool:
    """입력 텍스트에서 `detect_sexual_request` 조건 충족 여부를 판별한다."""
    t = user_text or ""
    keywords = ["안아", "키스", "벗어", "벗겨", "가슴", "팬티", "속옷", "허벅지", "허리", "엉덩이", "몸", "옷을", "옷 벗", "밀착", "포옹", "끌어안", "껴안", "입맞춤", "성기", "성행위"]
    return any(w in t for w in keywords)


def split_integrated_assistant(text: str) -> Tuple[str, str, str]:
    """입력 텍스트를 `split_integrated_assistant` 규칙에 따라 분해한다."""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) >= 2:
        narr = lines[0]
        dia = force_single_quote_line(lines[1])
        merged = (narr + "\n" + dia).strip()
        return narr, dia, merged
    if len(lines) == 1:
        # fallback: 서술이 비었을 경우 최소 서술 추가
        dia = force_single_quote_line(lines[0])
        narr = "숨이 짧게 새어 나왔다."
        merged = (narr + "\n" + dia).strip()
        return narr, dia, merged
    narr = "숨이 짧게 새어 나왔다."
    dia = "\"...\""
    merged = (narr + "\n" + dia).strip()
    return narr, dia, merged


def is_redundant_turn_dialogue(current: str, previous: str, *, min_overlap: int = 8) -> bool:
    """
    현재 대사가 직전 대사와 사실상 반복인지 판정한다.

    문장부호/공백을 제거한 뒤 긴 공통 구절 또는 높은 유사도면 반복으로 본다.
    """
    cur = normalize_space(remove_all_quotes(current or ""))
    prev = normalize_space(remove_all_quotes(previous or ""))
    if not cur or not prev:
        return False

    cur_key = re.sub(r"[\s\.\,\!\?\~\"'`…]+", "", cur)
    prev_key = re.sub(r"[\s\.\,\!\?\~\"'`…]+", "", prev)
    if len(cur_key) < min_overlap or len(prev_key) < min_overlap:
        return False

    if cur_key in prev_key or prev_key in cur_key:
        return True
    if SequenceMatcher(None, cur_key, prev_key).ratio() >= 0.86:
        return True
    return False


def llm_stall_detect_local(
    model,
    tokenizer,
    *,
    system_lore: str,
    action_state: str,
    prev_user_dialogue: str,
    prev_asst_dialogue: str,
    user_dialogue: str,
    asst_dialogue: str,
) -> bool:
    """직전 턴 대비 전개 정체(반복/의도 고착)를 LLM으로 판정한다."""
    sexual_mode = str(action_state or "").startswith("SEXUAL_")
    strict_rule = (
        "- SEXUAL 국면에서는 새 행동/새 결정/새 정보 중 2개 이상이 아니면 STALL.\n"
        "- SEXUAL 국면에서는 직전과 같은 접촉·요청·리듬 표현이 반복되면 STALL.\n"
        if sexual_mode
        else ""
    )
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "너는 VN 전개 정체 판정기다. "
                "직전 턴 대비 실질 진행이 부족하거나 반복이면 STALL, "
                "명확한 전진이 있으면 ADVANCE를 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                "다음 체크리스트를 내부 판정에 반드시 사용하라.\n"
                "- 새 행동 있음: Y/N\n"
                "- 새 결정 있음: Y/N\n"
                "- 새 정보 있음: Y/N\n"
                "- 같은 말하기 방식 반복: Y/N\n"
                "- 같은 의도/요청 반복: Y/N\n"
                "- 직전 상대의 거절/유보를 무시한 반복: Y/N\n"
                "- 직전 상대의 동의/수락 뒤 재확인 반복: Y/N\n"
                "- 하드 규칙: 새 항목(Y) 합계가 2 미만이면 STALL.\n"
                "- 하드 규칙: 반복 항목(Y)이 1개 이상이면 STALL.\n"
                "- 하드 규칙: 직전 상대가 거절/유보/난색을 보였는데 현재 화자가 같은 요구를 다시 밀면 STALL.\n"
                "- 하드 규칙: 직전 상대가 동의/수락했는데 현재 화자가 같은 동의 확인을 반복하면 STALL.\n"
                f"{strict_rule}"
                "출력 형식:\n"
                "1행: STALL 또는 ADVANCE\n"
                "2행: 12자 이내 코드(예: NEW1_REP1)\n"
                f"[시나리오북]\n{system_lore}\n\n"
                f"[행위/사건 상태]\n{action_state}\n\n"
                f"[직전 user 대사]\n{prev_user_dialogue}\n\n"
                f"[직전 assistant 대사]\n{prev_asst_dialogue}\n\n"
                f"[현재 user 대사]\n{user_dialogue}\n\n"
                f"[현재 assistant 대사]\n{asst_dialogue}"
            ),
        },
    ]
    raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=prompt_msgs,
        max_new_tokens=24,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    s = normalize_space(raw).upper()
    if "ADVANCE" in s:
        return False
    if "STALL" in s:
        return True
    return False
