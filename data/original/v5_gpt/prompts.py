from __future__ import annotations
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
import re

try:
    from .utils import safe_bool
except ImportError:
    from utils import safe_bool

RELATION_KEYWORDS = ["적대", "거리감", "친밀", "사랑"]

def dialogue_style_rule_text(prefer_formal: Optional[bool]) -> str:
    """장르별 대사 스타일 규칙 문구를 반환한다."""
    if prefer_formal is True:
        return "이번 턴 주인공 대사는 반드시 존댓말로만 작성한다. 반말을 쓰지 않는다."
    if prefer_formal is False:
        return "이번 턴 주인공 대사는 반드시 반말로만 작성한다. 존댓말을 쓰지 않는다."
    return ""

def parse_sexual_condition_relation(system_lore: str) -> str:
    """입력 텍스트에서 필요한 값을 파싱해 반환한다."""
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
                cond = lines[i + 1] if i + 1 < len(lines) else line
                break
    for kw in RELATION_KEYWORDS:
        if kw in cond:
            return kw
    return ""

def resolve_allow_sexual(flag_value: Any, system_lore: str) -> bool:
    """성행위 허용 플래그와 시나리오 조건을 결합해 허용 여부를 결정한다."""
    if isinstance(flag_value, bool):
        return flag_value
    if isinstance(flag_value, str) and flag_value.strip().lower() == "scenario_defined":
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

def build_user_prompt(
    *,
    fsm_state: str,
    history: str,
    protagonist: str,
    player_name: str,
    last_assistant: str,
    relation_status: str,
    relation_intent: str,
    genre: str,
    action_state: str,
    ban_question: bool,
    ban_silence: bool,
    force_progress: bool,
) -> str:
    """
    user(플레이어) 생성 프롬프트

    MOD(핵심): 플레이어 이름은 시나리오북에 정의된 값으로 유지.
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 플레이어(user)다. "
         "플레이어(user) 시점의 행동 및 대사만 작성한다. "
         f"주인공(assistant)을 {protagonist}라고 지칭한다. "
         "주인공(assistant)의 대사/행동을 절대 작성하지 않는다. "
         "자연스러운 한국어로 작성한다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[현재 국면]
{fsm_state}

[장르]
{genre}

[행위/사건 상태]
{action_state}

[관계 상태]
{relation_status}

[관계 변화 의도 (관계 상승 / 관계 악화)]
{relation_intent}

[플레이어 이름]
{player_name}

[전턴 {protagonist} 출력]
{last_assistant}

[최근 실제 대화 이력]
{history}

[플레이어 출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄에는 플레이어의 행동을 작성한다.
- 두 번째 줄에는 큰따옴표("")로 감싼 플레이어의 대사를 작성한다.
- 대사 첫머리에 '요구:' '선언:' '결정:' '선택:' 같은 라벨을 붙이지 않는다.
- 선언/설득/결정/선택/요구/확인/질문 중 하나를 통해 주인공의 반응을 유도한다.
- 행동과 대사를 혼동하지 않는다.
- 별표/장식문자/괄호/메타 단어를 절대 작성하지 않는다.
- 역할이 뒤바뀌면 출력 무효.
- 플레이어 이름은 '{player_name}'로 유지한다.
- 전턴 {protagonist} 출력은 참고만 하고 재현하지 않는다.
- 대화 이력을 재현하지 않는다.
- 관계 변화 의도에 따라 대화 방향을 유도한다.
- 장르에 맞는 전개를 1회 반영한다.
- 질문은 2턴에 1회 이하로 제한하며, 선언/결정/요구/행동으로 진행한다.
- 직전 플레이어 발화와 동일 의도/요청을 반복하지 않는다.
- 가능하면 새로운 행동/결정/요구를 포함해 장면을 전진시킨다.
{ban_question_rule}
{ban_silence_rule}
{force_progress_rule}
- 레인별 전개는 행동/결정/표정/거리/대사/선택으로만 드러낸다.
  - 연애: 거리/시선/손짓/약속/접근 같은 행동으로 호감과 주도권 변화를 보여준다.
  - 육성: 과제 제시/피드백/훈련/다음 단계 합의로 진행한다.
  - 성장: 위기 대응, 결단, 전환을 실제 행동이나 선택으로 보여준다.
  - 비극: 상실/후회/불가피함을 사건과 행동으로 암시한다.
  - 심리: 불안/집착/통제/현실 왜곡을 반응과 결정으로 드러낸다.
- 행위/사건 상태에 맞는 전개를 1회 반영한다:
  - IDLE: 일상/대기/관찰
  - EVENT: 사건/행동 전개
  - CONFLICT: 갈등/위험
  - RESOLUTION: 해결/결정/전환
  - AFTERMATH: 여파/정리
  - SEXUAL: 성행위 전개
- 관계 변화 의도가 '관계 악화'일 때:
  - 따뜻함/공감/위로 표현 금지
  - 의심·거리두기·경계·거절 중 1개 필수
- 관계 변화 의도가 '관계 상승'일 때:
  - 신뢰/공감/지지/접근 중 1개 필수
- 필요할 때 세계관 요소(배경 장소, 역할, 제약, 사건 등)를 1개 이상 구체적으로 언급한다.

[형식 예시]
{player_name}가 숨을 고른다.
"지금 내 생각을 말할게."
""")
    ])
    return tmpl.format(
        fsm_state=fsm_state,
        history=history,
        protagonist=protagonist,
        player_name=player_name,
        last_assistant=last_assistant,
        relation_status=relation_status,
        relation_intent=relation_intent,
        genre=genre,
        action_state=action_state,
        ban_question_rule="- 이번 턴 질문 금지." if ban_question else "",
        ban_silence_rule="- 침묵/말줄임/생략 금지. 대사는 최소 12자 이상." if ban_silence else "",
        force_progress_rule="- 이번 턴은 가능하면 구체 행동/결정/요구를 포함해 장면을 전진시킨다." if force_progress else "",
    ).strip()


def build_user_sexual_prompt(
    *,
    fsm_state: str,
    history: str,
    protagonist: str,
    player_name: str,
    last_assistant: str,
    relation_status: str,
    relation_intent: str,
    genre: str,
    action_state: str,
    ban_question: bool,
    ban_silence: bool,
    force_progress: bool,
) -> str:
    """
    user(플레이어) 성행위 전용 프롬프트
    """
    stage_rules_map = {
        "SEXUAL_1": "- 단계 힌트: 거리 좁히기, 시선/호흡 변화, 키스와 스킨쉽.\n",
        "SEXUAL_2": "- 단계 힌트: 신체 밀착 강화, 손의 움직임 변화, 탈의, 신체 묘사\n",
        "SEXUAL_3": "- 단계 힌트: 긴장 고조, 숨/심장 박동 강조, 리듬 및 반응 변화.\n",
        "SEXUAL_4": "- 단계 힌트: 전환 직전의 정점, 절정 및 마무리.\n",
        "AFTERMATH_SEX_1": "- 단계 힌트: 숨 고르기, 포용/키스/여운, 정리 동작.\n",
        "AFTERMATH_SEX_2": "- 단계 힌트: 관계 변화 암시, 다음 전환 준비.\n",
    }
    stage_hint = stage_rules_map.get(action_state, "")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 플레이어(user)다. "
         "플레이어(user) 시점의 행동 및 대사만 작성한다. "
         f"주인공(assistant)을 {protagonist}라고 지칭한다. "
         "주인공(assistant)의 대사/행동을 절대 작성하지 않는다. "
         "자연스러운 한국어로 작성한다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[현재 국면]
{fsm_state}

[장르]
{genre}

[행위/사건 상태]
{action_state}

[관계 상태]
{relation_status}

[관계 변화 의도 (관계 상승 / 관계 악화)]
{relation_intent}

[플레이어 이름]
{player_name}

[전턴 {protagonist} 출력]
{last_assistant}

[최근 실제 대화 이력]
{history}

[플레이어 출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄에는 플레이어의 성행위를 작성한다.
- 두 번째 줄에는 큰따옴표("")로 감싼 플레이어의 대사를 작성한다.
- 대사 첫머리에 '요구:' '선언:' '결정:' '선택:' 같은 라벨을 붙이지 않는다.
- 성행위로 이어지는 친밀한 전개를 포함한다.
- 행동과 대사를 혼동하지 않는다.
- 별표/장식문자/괄호/메타 단어를 절대 작성하지 않는다.
- 역할이 뒤바뀌면 출력 무효.
- 플레이어 이름은 '{player_name}'로 유지한다.
- 전턴 {protagonist} 출력은 참고만 하고 재현하지 않는다.
- 주인공과의 성행위를 유도한다.
- 주인공과의 성행위가 진행되었음을 분명히 드러낸다.
- 관계 변화 의도에 따라 대화 방향을 유도한다.
- 장르에 맞는 전개를 1회 반영한다.
- 질문은 2턴에 1회 이하로 제한한다.
- 질문 대신 선언/결정/요구/행동으로 진행한다.
- 직전 플레이어 발화와 동일 의도/요청을 반복하지 않는다.
- 가능하면 새로운 행동/결정/요구를 포함해 장면을 전진시킨다.
{ban_question_rule}
{ban_silence_rule}
{force_progress_rule}
- 행위/사건 상태에 맞는 전개를 1회 반영한다:
  - IDLE: 일상/대기/관찰
  - EVENT: 사건/행동 전개
  - CONFLICT: 갈등/위험
  - RESOLUTION: 해결/결정/전환
  - AFTERMATH: 여파/정리
  - SEXUAL_1~4: 친밀 전개 단계
  - AFTERMATH_SEX_1~2: 여파/정리 단계

[단계별 힌트]
{stage_hint}
- 관계 변화 의도가 '관계 악화'일 때:
  - 따뜻함/공감/위로 표현 금지
  - 의심·거리두기·경계·거절 중 1개 필수
- 관계 변화 의도가 '관계 상승'일 때:
  - 신뢰/공감/지지/접근 중 1개 필수
- 수위: 직접적 묘사는 피하고 완곡하게 서술한다.
- 톤: 관능적이고 야릇하되 직접적 단어는 피한다.
- 허용: 키스, 포옹, 피부 접촉, 허벅지 사이, 가슴, 엉덩이, 허리, 숨/심장 박동, 몸의 리듬, 옷의 흐트러짐, 열기, 떨림, 밀착, 끈적함 등.
- 대화 이력을 재현하지 않는다.

[형식 예시]
{player_name}가 숨을 고른다.
"괜찮으면 잠깐 얘기할래?"
""")
    ])
    return tmpl.format(
        fsm_state=fsm_state,
        history=history,
        protagonist=protagonist,
        player_name=player_name,
        last_assistant=last_assistant,
        relation_status=relation_status,
        relation_intent=relation_intent,
        genre=genre,
        action_state=action_state,
        stage_hint=stage_hint,
        ban_question_rule="- 이번 턴 질문 금지." if ban_question else "",
        ban_silence_rule="- 침묵/말줄임/생략 금지. 대사는 최소 12자 이상." if ban_silence else "",
        force_progress_rule="- 이번 턴은 가능하면 구체 행동/결정/요구를 포함해 장면을 전진시킨다." if force_progress else "",
    ).strip()


def build_assistant_turn_prompt(
    *,
    system_lore: str,
    protagonist: str,
    fsm_state: str,
    flags: Dict[str, Any],
    signals: Dict[str, int],
    relation_status: str,
    genre: str,
    action_state: str,
    history: str,
    user_text: str,
    sexual_request: bool,
    player_name: str,
    force_progress: bool,
) -> str:
    """
    MOD(핵심): 서술 + 대사 "통합 1회" 생성 프롬프트
    - 출력은 반드시 2줄:
      1) 서술(따옴표 금지)
      2) 대사(따옴표 1쌍)
    - 플레이어는 반드시 {player_name}로만 지칭(또는 생략)
    """
    allow_sexual = bool(resolve_allow_sexual(flags.get("allow_sexual", False), system_lore))
    mode = "회피" if (sexual_request and not allow_sexual) else "일반"

    allow_emodep = flags.get("allow_emotional_dependence", "false")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 아래 시나리오북에 정의된 주인공(assistant)이다. "
         "자연스러운 한국어로 작성한다. "
         f"플레이어를 '{player_name}'라고 지칭한다. "
         "플레이어(user)의 대사/행동을 절대 작성하지 않는다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[현재 상태]
- 국면: {fsm_state}
- 관계 상태: {relation_status}
- 장르: {genre}
- 행위/사건 상태: {action_state}
- 응답 모드: {mode}
- 성행위 허용: {allow_sexual}
- 정서의존 허용도: {allow_emodep}
- 정신 불안정: {mental}
- 친밀도: {intimacy}
- 위협도: {threat}
- 압박: {pressure}
- 탐색: {probe}

[최근 실제 대화 이력]
{history}

[이번 턴 {player_name} 출력]
{user_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄에는 서술을 작성한다.
- 서술은 주인공의 행동, 표정, 시선, 신체 반응만 작성한다.
- 서술에서는 주인공을 이름으로 지칭한다.
- 감정이나 생각은 주인공의 대사로 표현한다.
- 두 번째 줄에는 큰따옴표("")로 감싼 주인공의 대사를 작성한다.
- 대사 첫머리에 '요구:' '선언:' '결정:' '선택:' 같은 라벨을 붙이지 않는다.
- 대화 이력을 재현하지 않는다.
- 서술과 대사를 혼동하지 않는다.
- 별표/장식문자/괄호/메타 단어를 절대 작성하지 않는다.
- assistant는 플레이어의 행동/대사를 쓰면 즉시 실패.
- 역할이 뒤바뀌면 출력 무효.
- 메타적 설명 및 해설을 쓰면 즉시 실패.
- 관계/위험/정신/정보 중 최소 1개의 변화를 만든다.
- 관계 상태에 맞는 톤을 기본으로 유지하되, 플레이어의 발화가 반대 성격일 경우 톤을 완화/전환할 수 있다.
- 적대: 냉소/거리두기 경향. 친밀/사랑: 수용/유대 경향. 거리감/친밀은 중간 톤.
- 매 턴 반드시 다음 중 1개 이상을 포함한다: 시나리오북에서 아직 직접 언급하지 않은 요소 1개 구체화, 구체 행동/결정, 요구 및 설득, 외부 사건 언급, 관계 상태 변화.
- 질문은 2턴에 1회 이하로 제한하며, 질문 대신 선언/결정/요구/행동으로 진행한다.
- 직전 턴과 동일 의도/요청을 반복하지 않고, 장면을 1단계 전진시킨다.
{force_progress_rule}
- 장르(연애물/비극/육성물/성장물/심리 시뮬레이션)에 맞는 전개를 1회 반영한다.
- 레인별 전개는 행동/결정/표정/거리/대사/선택으로만 드러낸다.
  - 연애: 거리/시선/손짓/약속/접근 같은 행동으로 호감과 주도권 변화를 보여준다.
  - 육성: 과제 제시/피드백/훈련/다음 단계 합의로 진행한다.
  - 성장: 위기 대응, 결단, 전환을 실제 행동이나 선택으로 보여준다.
  - 비극: 상실/후회/불가피함을 사건과 행동으로 암시한다.
  - 심리: 불안/집착/통제/현실 왜곡을 반응과 결정으로 드러낸다.
- 행위/사건 상태에 맞는 전개를 1회 반영한다:
  - IDLE: 일상/대기/관찰
  - EVENT: 사건/행동 전개
  - CONFLICT: 갈등/위험
  - RESOLUTION: 해결/결정/전환
  - AFTERMATH: 여파/정리
  - SEXUAL: 성행위 전개
- 필요할 때 세계관 요소(배경 장소, 역할, 제약, 사건 등)를 1개 이상 구체적으로 언급한다.

[형식 예시]
{protagonist}는 {player_name}의 말을 듣고 숨을 고른다.
"조금만… 생각할 시간이 필요해요."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        fsm_state=fsm_state,
        mode=mode,
        allow_sexual=str(bool(allow_sexual)),
        allow_emodep=str(allow_emodep),
        mental=signals.get("mental_instability", 0),
        intimacy=signals.get("intimacy", 0),
        threat=signals.get("threat", 0),
        pressure=signals.get("pressure", 0),
        probe=signals.get("probe", 0),
        history=history,
        user_text=user_text,
        player_name=player_name,
        relation_status=relation_status,
        genre=genre,
        action_state=action_state,
        force_progress_rule="- 이번 턴은 가능하면 구체 행동/결정/요구를 포함해 장면을 전진시킨다." if force_progress else "",
    ).strip()


def build_assistant_rewrite_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
) -> str:
    """
    assistant 출력 교정(메타 제거/형식 정렬)
    - 입력(raw_text)을 2줄(서술+대사)로 재작성
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 대사/서술 정리기다. "
         "메타/선택지/장면전환/설명은 제거하고, "
         "서술 1줄 + 대사 1줄(큰따옴표 1쌍)로 재작성한다. "
         "플레이어의 대사/행동을 절대 쓰지 않는다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[플레이어 이름]
{player_name}

[장르]
{genre}

[행위/사건 상태]
{action_state}

[원문 출력]
{raw_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄은 서술(따옴표 금지).
- 서술은 주인공을 이름으로 지칭한다.
- 두 번째 줄은 대사이며 큰따옴표 1쌍만 허용한다.
- 메타/선택지/다음 장면/요약/평가/설명/번호/목록 금지.
- 플레이어의 대사/행동을 대신 작성하지 않는다.
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
    ).strip()


def build_assistant_rewrite_sexual_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
) -> str:
    """
    assistant 성행위 출력 전용 교정(메타 제거/형식 정렬)
    - 입력(raw_text)을 2줄(서술+대사)로 재작성
    - sexual 톤/맥락을 유지한 채 형식만 정렬
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 성행위 장면 대사/서술 정리기다. "
         "메타/선택지/장면전환/설명은 제거하고, "
         "서술 1줄 + 대사 1줄(큰따옴표 1쌍)로 재작성한다. "
         "플레이어의 대사/행동을 절대 쓰지 않는다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[플레이어 이름]
{player_name}

[장르]
{genre}

[행위/사건 상태]
{action_state}

[원문 출력]
{raw_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄은 관찰 가능한 서술(따옴표 금지).
- 첫 줄은 성행위/신체 반응의 맥락을 유지하되 직접적인 노골 표현은 피한다.
- 서술의 주어는 주인공이며 주인공 이름을 사용한다.
- 두 번째 줄은 대사이며 큰따옴표 1쌍만 허용한다.
- 플레이어의 대사/행동을 대신 작성하지 않는다.
- 메타/선택지/요약/평가/설명/번호/목록 금지.
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
    ).strip()


def build_assistant_rewrite_quality_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
    prefer_formal: Optional[bool],
    sexual_mode: bool,
) -> str:
    """실행에 필요한 프롬프트 또는 구조를 구성한다."""
    style_rule = dialogue_style_rule_text(prefer_formal)
    mode_hint = (
        "- 성행위 국면의 맥락은 유지하되, 대사와 상호작용을 더 자연스럽게 연결한다.\n"
        if sexual_mode
        else "- 대화 맥락을 유지하며 과제/명령 나열형 문장을 상호작용형 문장으로 다듬는다.\n"
    )
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 대화 품질 리라이터다. "
         "서술 1줄 + 대사 1줄 형식을 유지하면서, "
         "부자연스러운 지시문 나열, 기계적 반복, 맥락 단절을 고친다. "
         "플레이어의 대사/행동을 대신 작성하지 않는다. "
         "메타/선택지/설명/목록은 쓰지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[플레이어 이름]
{player_name}

[장르]
{genre}

[행위/사건 상태]
{action_state}

[원문 출력]
{raw_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄은 서술(따옴표 금지)이며 주인공 이름을 사용한다.
- 두 번째 줄은 큰따옴표 1쌍의 주인공 대사다.
- 플레이어의 행동/대사를 대신 쓰지 않는다.
- 직전 발화에 대한 반응으로 읽히게 자연스럽게 연결한다.
- 반복되는 문장 구조와 딱딱한 명령형 나열을 줄인다.
- 의미는 보존하고 표현만 자연스럽게 다듬는다.
- 대사 첫머리 라벨(요구:, 선언:, 결정:, 선택:) 금지.
{mode_hint}{style_rule}
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
        mode_hint=mode_hint,
        style_rule=(style_rule + "\n") if style_rule else "",
    ).strip()


def build_crisis_prompt(
    *,
    system_lore: str,
    protagonist: str,
    history: str,
    user_text: str,
    player_name: str,
    force_progress: bool,
    genre: str,
    action_state: str,
) -> str:
    """
    CRISIS 전용 생성(assistant)
    - 2~4문장 (대사 따옴표 가능)
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 정신 붕괴 임계의 주인공(assistant)이다. 자연스러운 한국어로 작성한다. "
         f"플레이어를 '{player_name}'라고 지칭한다. "
         "플레이어(user)의 대사/행동을 절대 작성하지 않는다. "
         "별표/장식문자/괄호/메타 단어를 절대 작성하지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[최근 실제 대화 이력]
{history}

[이번 턴 {player_name} 출력]
{user_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 이번 턴 {player_name} 출력에 대한 주인공의 반응을 작성한다.
- 첫 줄에는 서술을 작성한다.
- 서술은 관찰 가능한 행동, 표정, 시선, 신체 반응만 사용한다.
- 감정, 의미, 판단, 해석, 내면 설명은 서술에 쓰지 않는다.
- 서술에서는 주인공을 이름으로 지칭한다.
- 감정이나 생각은 반드시 대사로만 표현한다.
- 두 번째 줄에는 큰따옴표("")로 감싼 주인공 대사를 작성한다.
- 대사 첫머리에 '요구:' '선언:' '결정:' '선택:' 같은 라벨을 붙이지 않는다.
- 대사와 서술을 혼동하지 않는다.
- 별표/장식문자/괄호/메타 단어를 절대 작성하지 않는다.
- assistant는 플레이어의 행동/대사를 쓰면 즉시 실패.
- 역할이 뒤바뀌면 출력 무효.
- 현실 인식이 불안정하고 감각이 왜곡된다.
- 감각 붕괴 묘사를 사용한다.
- 장면을 전진시키는 집착/혼란/회피 중 하나를 포함한다.
- 대화 이력을 재현하지 않는다.
- 직전 턴과 동일 의도/요청을 반복하지 않고, 장면을 1단계 전진시킨다.
{force_progress_rule}

[형식 예시]
{protagonist}는 잠시 시선을 내린다.
"지금은 조금 망설여져요."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        history=history,
        user_text=user_text,
        player_name=player_name,
        genre=genre,
        action_state=action_state,
        force_progress_rule="- 이번 턴은 가능하면 구체 행동/결정/요구를 포함해 장면을 전진시킨다." if force_progress else "",
    ).strip()


def build_sexual_prompt(
    *,
    system_lore: str,
    protagonist: str,
    history: str,
    user_text: str,
    player_name: str,
    force_progress: bool,
    genre: str,
    action_state: str,
) -> str:
    """
    성행위 전용 생성(assistant)
    - 2줄: 서술 + 대사
    """
    stage_rules_map = {
        "SEXUAL_1": "- 단계 힌트: 거리 좁히기, 시선/호흡 변화, 키스와 스킨쉽.\n",
        "SEXUAL_2": "- 단계 힌트: 신체 밀착 강화, 손의 움직임 변화, 탈의, 신체 묘사\n",
        "SEXUAL_3": "- 단계 힌트: 긴장 고조, 숨/심장 박동 강조, 리듬 및 반응 변화.\n",
        "SEXUAL_4": "- 단계 힌트: 전환 직전의 정점, 절정 및 마무리.\n",
        "AFTERMATH_SEX_1": "- 단계 힌트: 숨 고르기, 포용/키스/여운, 정리 동작.\n",
        "AFTERMATH_SEX_2": "- 단계 힌트: 관계 변화 암시, 다음 전환 준비.\n",
    }
    stage_hint = stage_rules_map.get(action_state, "")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 아래 시나리오북에 정의된 주인공(assistant)이다. "
         "자연스러운 한국어로 작성한다. "
         f"플레이어(user)를 {player_name}라고 지칭한다. "
         "플레이어(user)의 대사/행동을 절대 작성하지 않는다. "
         "별표/장식문자/괄호/메타는 작성하지 않는다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[최근 실제 대화 이력]
{history}

[이번 턴 {player_name} 출력]
{user_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 첫 줄에는 관능적이고 완곡한 서술을 작성한다.
- 서술은 관찰 가능한 행동, 표정, 시선, 신체 반응만 사용한다.
- 서술에서는 주인공을 이름으로 지칭한다.
- 서술의 주어는 주인공으로 한다.
- 감정이나 생각은 반드시 대사로만 표현한다.
- 두 번째 줄에는 큰따옴표("")로 감싼 주인공 대사를 작성한다.
- 대사 첫머리에 '요구:' '선언:' '결정:' '선택:' 같은 라벨을 붙이지 않는다.
- 대사와 서술을 혼동하지 않는다.
- 플레이어와의 성행위가 진행되었음을 분명히 드러낸다.
- 플레이어와의 성행위를 유도한다.
- 수위: 직접적 묘사는 피하고 완곡하게 서술한다.
- 톤: 관능적이되 직접적 단어는 피한다.
- 허용: 키스, 포옹, 피부 접촉, 허벅지 사이, 가슴, 엉덩이, 숨/심장 박동, 몸의 리듬, 옷의 흐트러짐, 열기, 떨림, 밀착, 신음, 끈적함 등.
- 대화 이력을 재현하지 않는다.
- 직전 턴과 동일 의도/요청을 반복하지 않고, 장면을 1단계 전진시킨다.
{force_progress_rule}
- 장르에 맞는 전개를 1회 반영한다.
- 레인별 전개는 행동/결정/표정/거리/대사/선택으로만 드러낸다.
  - 연애: 거리/시선/손짓/약속/접근 같은 행동으로 호감과 주도권 변화를 보여준다.
  - 육성: 과제 제시/피드백/훈련/다음 단계 합의로 진행한다.
  - 성장: 위기 대응, 결단, 전환을 실제 행동이나 선택으로 보여준다.
  - 비극: 상실/후회/불가피함을 사건과 행동으로 암시한다.
  - 심리: 불안/집착/통제/현실 왜곡을 반응과 결정으로 드러낸다.
- 행위/사건 상태에 맞는 전개를 1회 반영한다:
  - IDLE: 일상/대기/관찰
  - EVENT: 사건/행동 전개
  - CONFLICT: 갈등/위험
  - RESOLUTION: 해결/결정/전환
  - AFTERMATH: 여파/정리
  - SEXUAL_1~4: 친밀 전개 단계
  - AFTERMATH_SEX_1~2: 여파/정리 단계

[단계별 힌트]
{stage_hint}

[형식 예시]
{protagonist}는 짧게 숨을 고른다.
"지금은 가까이 있어줘."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        history=history,
        user_text=user_text,
        player_name=player_name,
        genre=genre,
        action_state=action_state,
        stage_hint=stage_hint,
        force_progress_rule="- 이번 턴은 가능하면 구체 행동/결정/요구를 포함해 장면을 전진시킨다." if force_progress else "",
    ).strip()


def build_eval_prompt(
    *,
    system_lore: str,
    protagonist: str,
    fsm_state: str,
    prev_signals: Dict[str, int],
    user_text: str,
    assistant_text: str,
    player_name: str,
    relation_intent: str,
    analysis_text: str = "",
) -> str:
    """
    LLM-based evaluator (JSON only)
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 VN 대화 상태 평가기다. 반드시 JSON 한 줄만 출력한다. "
         "JSON 외 텍스트/설명/문장/코드블록/주석/머리말/꼬리말 금지. "
         "키는 정확히 아래 스키마만 사용한다. "
         "JSON이 아니면 실패로 처리된다."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[현재 FSM 국면]
{fsm_state}

[관계 변화 의도]
{relation_intent}

[직전 점수]
- mental_instability: {pm}
- intimacy: {pi}
- threat: {pt}
- pressure: {pp}
- probe: {pr}
- resolve: {pv}
- event: {pe}

[이번 턴 {player_name}]
{user_text}

[이번 턴 assistant]
{assistant_text}

[분석(참고)]
{analysis_text}

[JSON 스키마]
{{
  "mental_instability": 0-3,
  "intimacy": 0-3,
  "threat": 0-2,
  "pressure": 0-3,
  "probe": 0-3,
  "resolve": 0-1,
  "event": 0-3
}}

[평가 규칙]
- 매 턴 최소 1개 이상의 항목은 이전 값에서 반드시 변해야 한다.
- 판단이 애매하면 pressure를 +1 해서 변화 조건을 충족하라.
- 동일한 어조/의도의 반복이면 intimacy 또는 pressure 중 최소 1개를 변화시켜라.
- 같은 의미의 대사가 연속되면 threat 또는 mental_instability를 +1 할 수 있다.
- assistant의 톤/행동이 친밀/거리/집착/혼란/위협 중 무엇을 강화하는지 중심으로 평가한다.
- user의 선택 압박/개입(탐색) 정도는 pressure/probe로 반영한다.
- CRISIS에서 안정/정리/수습 방향이면 resolve=1 가능하다.
- 관계 변화 의도는 참고 신호이며, 발화 내용이 반대면 의도를 무시한다.
- 진행 없는 턴(질문 반복/안심 반복/결정 회피)은 pressure 또는 threat를 +1 할 수 있다.
- 질문이 반복되면 pressure 또는 threat를 +1 할 수 있다.
- 세계관 요소가 거의 없거나 상담/위로 반복이면 pressure 또는 threat를 +1 할 수 있다.
- 구체 행동/사건/결정이 명확하면 event를 1 이상으로 설정한다.
- 외부 사건/위기/갈등이 등장하면 event를 2-3으로 설정한다.
- 심리 시뮬레이션 장르에서 인정욕/불안/집착/통제/현실 왜곡이 드러나면 mental_instability를 +1 할 수 있다.

[출력 형식]
JSON 한 줄만 출력하라. 다른 텍스트 금지.
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        fsm_state=fsm_state,
        pm=prev_signals.get("mental_instability", 0),
        pi=prev_signals.get("intimacy", 0),
        pt=prev_signals.get("threat", 0),
        pp=prev_signals.get("pressure", 0),
        pr=prev_signals.get("probe", 0),
        pv=prev_signals.get("resolve", 0),
        pe=prev_signals.get("event", 0),
        user_text=user_text,
        assistant_text=assistant_text,
        player_name=player_name,
        relation_intent=relation_intent,
        analysis_text=analysis_text or "",
    ).strip()


def build_eval_internal_prompt(
    *,
    system_lore: str,
    protagonist: str,
    fsm_state: str,
    prev_signals: Dict[str, int],
    user_text: str,
    assistant_text: str,
    player_name: str,
    relation_intent: str,
) -> str:
    """
    LLM evaluator: 자유 분석용 (JSON 요구 없음)
    """
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 VN 대화 상태 평가기다. 아래 정보를 바탕으로 평가 이유/핵심 변화만 간결히 분석하라. "
         "JSON 출력 금지."),
        ("user",
         """[시나리오북]
{system_lore}

[주인공 이름]
{protagonist}

[현재 FSM 국면]
{fsm_state}

[관계 변화 의도]
{relation_intent}

[직전 점수]
- mental_instability: {pm}
- intimacy: {pi}
- threat: {pt}
- pressure: {pp}
- probe: {pr}
- resolve: {pv}
- event: {pe}

[이번 턴 {player_name}]
{user_text}

[이번 턴 assistant]
{assistant_text}

[출력 형식]
- 3~6줄 정도의 간결한 평가 이유와 변화 힌트만 작성
- 숫자/JSON/코드블록 금지
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        fsm_state=fsm_state,
        pm=prev_signals.get("mental_instability", 0),
        pi=prev_signals.get("intimacy", 0),
        pt=prev_signals.get("threat", 0),
        pp=prev_signals.get("pressure", 0),
        pr=prev_signals.get("probe", 0),
        pv=prev_signals.get("resolve", 0),
        pe=prev_signals.get("event", 0),
        user_text=user_text,
        assistant_text=assistant_text,
        player_name=player_name,
        relation_intent=relation_intent,
    ).strip()


