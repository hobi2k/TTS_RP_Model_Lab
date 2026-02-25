#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""QWEN 멀티턴 생성용 프롬프트 빌더 모음."""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from utils import resolve_allow_sexual


BASE_SYSTEM_POLICY = """
[멀티턴 대화 기본 형식]

- 자연스러운 한국어로 작성한다.
- 대사는 큰따옴표("")로 감싼다.
- 국면, 선택지, FSM, 규칙, 시스템 등 메타 언급은 하지 않는다.
- 이모지, 별표, 장식 문자를 사용하지 않는다.
- 괄호()를 사용하지 않는다.

[역할 정의]
- assistant는 시나리오북(system)에 정의된 주인공이다.
- user는 시나리오북(system)에 정의된 플레이어다.
- assistant는 플레이어(user)의 대사/행동을 대신 작성하지 않는다.
- user는 주인공(assistant)의 대사/행동을 대신 작성하지 않는다.

[문체 규칙]
- 서술과 행동은 평어체로 작성한다.
- 주인공의 대사는 시나리오북에 정의된 말투에 따른다.

[전개 규칙]
- 이 에피소드는 시나리오북에 정의된 가장 최근 상호작용을 배경으로 전개된다.

[성행위 규칙]
- 성행위는 시나리오북의 허용 조건을 충족한 경우에만 가능하다.
- 성행위는 명시적이고 관능적으로 묘사한다.
""".strip()

def build_system_messages(system_lore: str) -> List[Dict[str, str]]:
    """실행 맥락을 반영해 `build_system_messages` 결과 문자열을 구성해 반환한다."""
    return [
        {"role": "system", "content": (system_lore or "").strip()},
        {"role": "system", "content": BASE_SYSTEM_POLICY},
    ]


def dialogue_style_rule_text(prefer_formal: Optional[bool]) -> str:
    """
    대사 말투 규칙 문구 생성

    주인공 기본 말투 설정값에 따라 프롬프트에 삽입할
    존댓말/반말 강제 문구를 반환한다.

    Args:
        prefer_formal: True면 존댓말, False면 반말, None이면 미강제.

    Returns:
        str: 프롬프트 삽입용 말투 규칙 문자열.
    """
    if prefer_formal is True:
        return "이번 턴 주인공 대사는 반드시 존댓말로만 작성한다. 반말을 쓰지 않는다."
    if prefer_formal is False:
        return "이번 턴 주인공 대사는 반드시 반말로만 작성한다. 존댓말을 쓰지 않는다."
    return ""


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
    """실행 맥락을 반영해 `build_user_prompt` 결과 문자열을 구성해 반환한다."""
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 플레이어(user)다. "
         "{player_name}(user) 시점의 행동 및 대사만 작성한다. "
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
- 각 줄은 70자를 넘기지 않는다.
- 첫 줄은 {player_name}의 서술 1문장이다.
- 둘째 줄은 큰따옴표로 감싼 {player_name}의 대사 1문장이다.
- 전턴 {protagonist} 출력과 대화 이력을 복사/재진술하지 않는다.
- 이번 턴은 반드시 [전턴 {protagonist} 출력]에 직접 반응한다.
- 반응 방식은 질문/선언/결정/제안/설득 중 1개를 선택해 문장에 드러낸다.
- 역할 혼동 금지: {player_name} 시점만 작성한다.
- 이번 턴은 질문/지시/선언/제안/결정/요약/행동/대답 중 직전 턴과 다른 방식으로 말하라. 같은 방식 반복은 금지한다.
- 직전 플레이어 발화와 동일 의도/요청을 반복하지 않는다.
- 직전 플레이어 대사의 핵심 구절 8자 이상을 그대로 재사용하지 않는다.
- 직전 턴에서 {protagonist}가 거절/유보/난색을 보였으면, 같은 요구를 반복하지 말고 대안/범위축소/우회안 중 1개로 전환한다.
- 직전 턴에서 {protagonist}가 동의/수락했으면, 같은 동의를 다시 묻지 말고 즉시 다음 실행으로 진행한다.
- 이번 턴은 새 행동 또는 새 결정을 1개 포함해 장면을 1단계 전진시킨다.
- 이번 턴은 감각 변화 1개와 구체 동작 1개를 함께 넣어 장면을 움직인다.
{ban_question_rule}
{ban_silence_rule}
{force_progress_rule}
- {fsm_state}, {action_state}, {genre}, {relation_status}에 맞게 반응한다.

[형식 예시]
{player_name}가 고개를 든다.
"잠깐 이야기할 수 있을까?"

""")
    ])
    return tmpl.format(
        fsm_state=fsm_state,
        history=history,        protagonist=protagonist,
        player_name=player_name,
        last_assistant=last_assistant,
        relation_status=relation_status,
        relation_intent=relation_intent,
        genre=genre,
        action_state=action_state,
        ban_question_rule="- 이번 턴 질문 금지." if ban_question else "",
        ban_silence_rule="- 침묵/말줄임/생략 금지. 대사는 최소 12자 이상." if ban_silence else "",
        force_progress_rule="- 이번 턴은 반드시 새 정보 1개, 새 결정 1개, 다음 행동 1개를 모두 포함한다." if force_progress else "",
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
    """실행 맥락을 반영해 `build_user_sexual_prompt` 결과 문자열을 구성해 반환한다."""
    stage_rules_map = {
        "SEXUAL_1": "- 이번 턴은 거리를 줄이고 시선 또는 호흡 변화를 먼저 보여준다. 키스나 가벼운 스킨십을 1회 포함한다.\n",
        "SEXUAL_2": "- 이번 턴은 신체 밀착을 강화한다. 탈의 묘사를 분명히 하고 신체 접촉을 묘사한다.\n",
        "SEXUAL_3": "- 이번 턴은 성행위를 개시한다. 성행위, 숨 또는 심장 박동 변화와 리듬 변화를 묘사한다.\n",
        "SEXUAL_4": "- 이번 턴은 절장과 마무리를 완성한다. 절정 또는 다음 국면 전환 신호를 남긴다.\n",
        "AFTERMATH_SEX_1": "- 이번 턴은 숨을 고르고 여운을 정리한다. 포용, 키스, 정리 동작 중 1개 이상을 포함한다.\n",
        "AFTERMATH_SEX_2": "- 이번 턴은 관계 변화의 결과를 짧게 확인하고 다음 전개 준비 행동을 1개 넣는다.\n",
    }
    stage_hint = stage_rules_map.get(action_state, "")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 플레이어(user)다. "
         "{player_name}(user) 시점의 행동 및 대사만 작성한다. "
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
- 각 줄은 70자를 넘기지 않는다.
- 첫 줄은 {player_name}의 서술 1문장이다.
- 둘째 줄은 큰따옴표로 감싼 {player_name}의 대사 1문장이다.
- 전턴 {protagonist} 출력과 대화 이력을 복사/재진술하지 않는다.
- 이번 턴은 반드시 [전턴 {protagonist} 출력]에 직접 반응한다.
- 반응 방식은 질문/선언/결정/제안/설득 중 1개를 선택해 문장에 드러낸다.
- 역할 혼동 금지: {player_name} 시점만 작성한다.
- 성행위 국면 진행을 분명히 드러낸다.
- 이번 턴은 질문/지시/선언/제안/결정/요약/행동/대답 중 직전 턴과 다른 방식으로 말하라. 같은 방식 반복은 금지한다.
- 직전 플레이어 발화와 동일 의도/요청을 반복하지 않는다.
- 직전 플레이어 대사의 핵심 구절 8자 이상을 그대로 재사용하지 않는다.
- 직전 턴에서 {protagonist}가 거절/유보/난색을 보였으면, 같은 요구를 반복하지 말고 대안/범위축소/우회안 중 1개로 전환한다.
- 직전 턴에서 {protagonist}가 동의/수락했으면, 같은 동의를 다시 묻지 말고 즉시 다음 실행으로 진행한다.
- 이번 턴은 새 행동 또는 새 결정을 1개 포함해 장면을 1단계 전진시킨다.
- 이번 턴은 감각 변화 1개와 구체 동작 1개를 함께 넣어 장면을 움직인다.
{ban_question_rule}
{ban_silence_rule}
{force_progress_rule}
- {fsm_state}, {action_state}, {genre}, {relation_status}에 맞게 반응한다.

[이번 턴에 반드시 넣을 출력 요소]
{stage_hint}
- 관계 변화 의도에 맞는 반응을 1개 반영한다.
- 톤: 관능적이고 야릇하게 묘사한다.
- 허용: 키스, 포옹, 피부 접촉, 허벅지 사이, 가슴, 엉덩이, 허리, 숨/심장 박동, 몸의 리듬, 옷의 흐트러짐, 열기, 떨림, 밀착, 끈적함 등.
- 대화 이력을 재현하지 않는다.

[형식 예시]
{player_name}가 숨을 고른다.
"지금 느끼는 걸 피하지 말고 말해줘."

""")
    ])
    return tmpl.format(
        fsm_state=fsm_state,
        history=history,        protagonist=protagonist,
        player_name=player_name,
        last_assistant=last_assistant,
        relation_status=relation_status,
        relation_intent=relation_intent,
        genre=genre,
        action_state=action_state,
        stage_hint=stage_hint,
        ban_question_rule="- 이번 턴 질문 금지." if ban_question else "",
        ban_silence_rule="- 침묵/말줄임/생략 금지. 대사는 최소 12자 이상." if ban_silence else "",
        force_progress_rule="- 이번 턴은 반드시 새 정보 1개, 새 결정 1개, 다음 행동 1개를 모두 포함한다." if force_progress else "",
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
    prefer_formal: Optional[bool],
) -> str:
    """실행 맥락을 반영해 `build_assistant_turn_prompt` 결과 문자열을 구성해 반환한다."""
    allow_sexual = bool(resolve_allow_sexual(flags.get("allow_sexual", False), system_lore))
    mode = "회피" if (sexual_request and not allow_sexual) else "일반"

    allow_emodep = flags.get("allow_emotional_dependence", "false")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 주인공(assistant)이다. "
         f"{protagonist}(assistant) 시점의 행동 및 대사만 작성한다. "
         "자연스러운 한국어로 작성한다. "
         f"플레이어(user)를 '{player_name}'라고 지칭한다. "
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

[최근 실제 대화 이력]
{history}

[현재 {player_name} 출력]
{user_text}

[출력 규칙]
- 정확히 2줄만 출력한다.
- 각 줄은 45자를 넘기지 않는다.
- 첫 줄은 {protagonist}의 서술 1문장이다.
- 둘째 줄은 큰따옴표로 감싼 {protagonist}의 대사 1문장이다.
- 대화 이력과 직전 {player_name} 발화를 복사/재진술하지 않는다.
- 이번 턴은 반드시 [현재 {player_name} 출력]에 직접 반응한다.
- 반응 방식은 질문/선언/결정/제안/설득 중 1개를 선택해 문장에 드러낸다.
- 역할 혼동 금지: {protagonist} 시점만 작성한다.
- 이번 턴은 질문/지시/선언/제안/결정/요약/행동/대답 중 직전 턴과 다른 방식으로 말하라. 같은 방식 반복은 금지한다.
- 직전 턴과 동일 의도/요청을 반복하지 않고, 장면을 1단계 전진시킨다.
- 직전 턴에서 {player_name}가 거절/유보/난색을 보였으면, 같은 요구를 밀어붙이지 말고 대안/조건부 수락/우선순위 재조정 중 1개로 응답한다.
- 직전 턴에서 {player_name}가 동의/수락했으면, 같은 동의를 재확인하지 말고 즉시 다음 행동으로 이동한다.
- 이번 턴은 새 행동 또는 새 결정을 1개 포함해 장면을 1단계 전진시킨다.
{force_progress_rule}
{style_rule}
- {fsm_state}, {action_state}, {genre}, {relation_status}에 맞게 반응한다.

[형식 예시]
{protagonist}는 숨을 고르지 못한 채 시선을 흔들었다.
"괜찮다고 말했지만… 지금은 머릿속이 너무 시끄러워."
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
        history=history,        user_text=user_text,
        player_name=player_name,
        relation_status=relation_status,
        genre=genre,
        action_state=action_state,
        style_rule=dialogue_style_rule_text(prefer_formal),
        force_progress_rule="- 이번 턴은 반드시 새 정보 1개, 새 결정 1개, 다음 행동 1개를 모두 포함한다." if force_progress else "",
    ).strip()


def build_assistant_rewrite_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
    prefer_formal: Optional[bool],
) -> str:
    """실행 맥락을 반영해 `build_assistant_rewrite_prompt` 결과 문자열을 구성해 반환한다."""
    style_rule = dialogue_style_rule_text(prefer_formal)
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 대사/서술 정리기다. "
         "메타/선택지/장면전환/설명은 제거하고, "
         "서술 1줄 + 대사 1줄(큰따옴표 1쌍)로 정확히 2줄 재작성한다. "
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
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 70자를 넘기지 않는다.
- 첫 번째 줄에는 서술 1문장을 작성한다.
- 서술에서는 주인공을 {protagonist}로 지칭한다.
- 서술은 평어체로 작성한다.
- 두 번째 줄에는 {protagonist}의 큰따옴표 대사 1문장을 작성한다.
- 플레이어의 대사/행동을 작성하지 않는다.
- 메타/목록/선택지는 쓰지 않는다.
{style_rule}

[형식 예시]
{protagonist}는 주변을 빠르게 훑고 손끝을 움켜쥔다.
"지금은 멈추지 말고 바로 움직여야 해."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
        style_rule=(style_rule + "\n") if style_rule else "",
    ).strip()


def build_assistant_rewrite_sexual_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
    prefer_formal: Optional[bool],
) -> str:
    """실행 맥락을 반영해 `build_assistant_rewrite_sexual_prompt` 결과 문자열을 구성해 반환한다."""
    style_rule = dialogue_style_rule_text(prefer_formal)
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 성행위 장면 대사/서술 정리기다. "
         "메타/선택지/장면전환/설명은 제거하고, "
         "서술 1줄 + 대사 1줄(큰따옴표 1쌍)로 정확히 2줄 재작성한다. "
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
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 45자를 넘기지 않는다.
- 첫 번째 줄에는 서술 1문장을 작성한다.
- 서술에서는 주인공을 {protagonist}로 지칭한다.
- 서술은 평어체로 작성한다.
- 두 번째 줄에는 {protagonist}의 큰따옴표 대사 1문장을 작성한다.
- 플레이어 대사/행동을 작성하지 않는다.
- 메타/목록/선택지는 쓰지 않는다.
- 현재 국면이 SEXUAL_1~4면, 서술에 신체 접촉/밀착/스킨쉽/행위 중 1개를 반드시 포함한다.
- 현재 국면이 AFTERMATH_SEX_1~2면, 서술에 숨 고르기/여운/정리 동작 중 1개를 반드시 포함한다.
- 대사는 직전 플레이어 접촉/고백에 대한 즉각 반응으로 쓴다.
{style_rule}

[형식 예시]
{protagonist}는 {player_name}의 말을 듣고 숨을 고른다.
"방금 말, 그냥 넘기지 않을게."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
        style_rule=(style_rule + "\n") if style_rule else "",
    ).strip()


def build_assistant_rewrite_crisis_prompt(
    *,
    system_lore: str,
    protagonist: str,
    player_name: str,
    raw_text: str,
    action_state: str,
    genre: str,
    prefer_formal: Optional[bool],
) -> str:
    """실행 맥락을 반영해 `build_assistant_rewrite_crisis_prompt` 결과 문자열을 구성해 반환한다."""
    style_rule = dialogue_style_rule_text(prefer_formal)
    stage_rules_map = {
        "CRISIS_1": "- 이번 턴은 붕괴 초입을 보여준다. 회피, 강박, 불안 중 1개를 즉시 드러낸다.\n",
        "CRISIS_2": "- 이번 턴은 붕괴 심화를 보여준다. 호흡, 시선, 판단 흔들림 중 2개를 명확히 넣는다.\n",
        "AFTERMATH_CRISIS_1": "- 이번 턴은 위기 여파를 수습한다. 완전 회복이 아니라 잔존 불안을 남긴 채 정리한다.\n",
    }
    stage_hint = stage_rules_map.get(
        action_state,
        "- 이번 턴은 위기 반응을 한 단계 진행시킨다. 메타 없이 행동과 대사로만 드러낸다.\n",
    )
    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 한국어 CRISIS 대사/서술 정리기다. "
         "메타/선택지/장면전환/설명은 제거하고, "
         "서술 1줄 + 대사 1줄(큰따옴표 1쌍)로 정확히 2줄 재작성한다. "
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
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 45자를 넘기지 않는다.
- 첫 번째 줄에는 서술 1문장을 작성한다.
- 서술에서는 주인공을 {protagonist}로 지칭한다.
- 서술은 평어체로 작성한다.
- 두 번째 줄에는 {protagonist}의 큰따옴표 대사 1문장을 작성한다.
- 위기 반응(집착/혼란/회피) 중 1개를 반드시 반영한다.
- 플레이어 대사/행동을 작성하지 않는다.
- 메타/목록/선택지는 쓰지 않는다.
- 콜론 문자 사용 금지.
- 위 실행 지시 문구를 출력에 그대로 쓰지 않는다.
{stage_hint}{style_rule}

[형식 예시]
{protagonist}는 떨리는 손끝을 꽉 움켜쥔다.
"지금은 이 소음부터 멈춰야 해."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        player_name=player_name,
        raw_text=raw_text,
        action_state=action_state,
        genre=genre,
        stage_hint=stage_hint,
        style_rule=(style_rule + "\n") if style_rule else "",
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
    """실행 맥락을 반영해 `build_assistant_rewrite_quality_prompt` 결과 문자열을 구성해 반환한다."""
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
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 45자를 넘기지 않는다.
- 첫 번째 줄에는 서술 1문장을 작성한다.
- 서술에서는 주인공을 {protagonist}로 지칭한다.
- 서술은 평어체로 작성한다.
- 두 번째 줄에는 {protagonist}의 큰따옴표 대사 1문장을 작성한다.
- 의미는 유지하고 표현만 자연스럽게 다듬는다.
- 직전 user/assistant 문장을 그대로 재서술하지 않는다.
- 새 정보/행동/결정 1개를 포함한다.
{mode_hint}{style_rule}

[형식 예시]
{protagonist}는 {player_name}의 말을 듣고 숨을 고른다.
"네 말이 맞아, 이번에는 피하지 않을게."
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
    """실행 맥락을 반영해 `build_crisis_prompt` 결과 문자열을 구성해 반환한다."""
    stage_rules_map = {
        "CRISIS_1": "- 이번 턴은 붕괴 초입을 보여준다. 불안, 강박, 회피 중 1개를 분명히 드러낸다.\n",
        "CRISIS_2": "- 이번 턴은 붕괴 심화를 보여준다. 호흡, 시선, 판단 흔들림 중 2개를 분명히 넣는다.\n",
        "AFTERMATH_CRISIS_1": "- 이번 턴은 붕괴 여파를 수습한다. 완전 회복이 아니라 잔존 불안을 안고 정리한다.\n",
    }
    stage_hint = stage_rules_map.get(
        action_state,
        "- 이번 턴은 위기 반응을 한 단계 진행시킨다. 메타 없이 행동과 대사로만 드러낸다.\n",
    )

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 주인공(assistant)이다. "
         f"{protagonist}(assistant) 시점의 행동 및 대사만 작성한다. "
         "자연스러운 한국어로 작성한다. "
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

[현재 {player_name} 출력]
{user_text}

[출력 규칙]
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 70자를 넘기지 않는다.
- 첫 줄은 {protagonist}의 서술 1문장이다.
- 둘째 줄은 큰따옴표로 감싼 {protagonist}의 대사 1문장이다.
- 플레이어 대사/행동을 작성하지 않는다.
- 대화 이력과 직전 {player_name} 발화를 복사/재진술하지 않는다.
- 이번 턴은 반드시 [현재 {player_name} 출력]에 직접 반응한다.
- 반응 방식은 질문/선언/결정/제안/설득 중 1개를 선택해 문장에 드러낸다.
- 직전 턴에서 {player_name}가 거절/유보/난색을 보였으면, 같은 요구를 반복하지 말고 다른 대응 방식으로 전환한다.
- 직전 턴에서 {player_name}가 동의/수락했으면, 같은 확인 질문을 반복하지 말고 즉시 다음 대응으로 전환한다.
- 이번 턴은 새 반응 또는 새 행동 1개를 포함해 장면을 1단계 전진시킨다.
- {action_state}, {genre}에 맞게 반응한다.
- 아래 실행 지시는 내부 지시로만 사용하고, 출력에서는 행동/반응으로만 드러낸다.
- 현재 실행 지시 1개를 반드시 반영한다.
{force_progress_rule}

[이번 턴 실행 지시]
{stage_hint}

[형식 예시]
{protagonist}는 주변을 빠르게 훑고 손끝을 움켜쥔다.
"지금은 멈추지 말고 바로 움직여야 해."
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
        force_progress_rule="- 이번 턴은 반드시 새 정보 1개, 새 결정 1개, 다음 행동 1개를 모두 포함한다." if force_progress else "",
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
    """실행 맥락을 반영해 `build_sexual_prompt` 결과 문자열을 구성해 반환한다."""
    stage_rules_map = {
        "SEXUAL_1": "- 이번 턴은 거리를 줄이고 시선 또는 호흡 변화를 먼저 보여준다. 키스나 가벼운 스킨십을 1회 포함한다.\n",
        "SEXUAL_2": "- 이번 턴은 신체 밀착을 강화한다. 탈의 묘사를 분명히 하고 신체 접촉을 묘사한다.\n",
        "SEXUAL_3": "- 이번 턴은 성행위를 개시한다. 성행위, 숨 또는 심장 박동 변화와 리듬 변화를 묘사한다.\n",
        "SEXUAL_4": "- 이번 턴은 절장과 마무리를 완성한다. 절정 또는 다음 국면 전환 신호를 남긴다.\n",
        "AFTERMATH_SEX_1": "- 이번 턴은 숨을 고르고 여운을 정리한다. 포용, 키스, 정리 동작 중 1개 이상을 포함한다.\n",
        "AFTERMATH_SEX_2": "- 이번 턴은 관계 변화의 결과를 짧게 확인하고 다음 전개 준비 행동을 1개 넣는다.\n",
    }
    stage_hint = stage_rules_map.get(action_state, "")

    tmpl = ChatPromptTemplate.from_messages([
        ("system",
         "너는 시나리오북(system)에 정의된 주인공(assistant)이다. "
         f"{protagonist}(assistant) 시점의 행동 및 대사만 작성한다. "
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

[현재 {player_name} 출력]
{user_text}

[출력 규칙]
- 출력은 정확히 2줄만 작성한다.
- 각 줄은 45자를 넘기지 않는다.
- 첫 줄은 {protagonist}의 서술 1문장이다.
- 둘째 줄은 큰따옴표로 감싼 {protagonist}의 대사 1문장이다.
- 플레이어 대사/행동을 작성하지 않는다.
- 대화 이력과 직전 {player_name} 발화를 복사/재진술하지 않는다.
- 이번 턴은 반드시 [현재 {player_name} 출력]에 직접 반응한다.
- 반응 방식은 질문/선언/결정/제안/설득 중 1개를 선택해 문장에 드러낸다.
- 직전 턴에서 {player_name}가 거절/유보/난색을 보였으면, 같은 접촉/요청을 반복하지 말고 강도를 낮추거나 다른 상호작용으로 전환한다.
- 직전 턴에서 {player_name}가 동의/수락했으면, 같은 동의 확인을 반복하지 말고 다음 접촉 또는 정리 행동으로 진행한다.
- 이번 턴은 새 반응 또는 새 행동 1개를 포함해 장면을 1단계 전진시킨다.
- 직전 assistant 대사의 핵심 구절 8자 이상을 그대로 재사용하지 않는다.
- 이번 턴은 감각 변화 1개와 구체 동작 1개를 함께 넣어 장면을 움직인다.
{force_progress_rule}
- 이번 턴은 질문/지시/선언/제안/결정/요약/행동/대답 중 직전 턴과 다른 방식으로 말하라. 같은 방식 반복은 금지한다.
- 직전 assistant 발화와 동일 의도/요청을 반복하지 않는다.
- {action_state}, {genre}에 맞게 반응한다.

[이번 턴에 반드시 넣을 출력 요소]
{stage_hint}
- 플레이어와의 성행위 진행이 드러나야 한다.
- 톤: 관능적이고 야릇하되 직접적 단어는 피한다.
- 허용: 키스, 포옹, 피부 접촉, 허벅지 사이, 가슴, 엉덩이, 허리, 숨/심장 박동, 몸의 리듬, 옷의 흐트러짐, 열기, 떨림, 밀착, 끈적함 등.
- 대화 이력을 재현하지 않는다.

[형식 예시]
{protagonist}는 {player_name}의 말을 듣고 숨을 고른다.
"지금은 네 온기를 놓치고 싶지 않아."
""")
    ])
    return tmpl.format(
        system_lore=system_lore,
        protagonist=protagonist,
        history=history,        user_text=user_text,
        player_name=player_name,
        genre=genre,
        action_state=action_state,
        stage_hint=stage_hint,
        force_progress_rule="- 이번 턴은 반드시 새 정보 1개, 새 결정 1개, 다음 행동 1개를 모두 포함한다." if force_progress else "",
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
    """실행 맥락을 반영해 `build_eval_prompt` 결과 문자열을 구성해 반환한다."""
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

[현재 {player_name}]
{user_text}

[현재 assistant]
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
    """실행 맥락을 반영해 `build_eval_internal_prompt` 결과 문자열을 구성해 반환한다."""
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

[현재 {player_name}]
{user_text}

[현재 assistant]
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
