"""QWEN 기반 v6 멀티턴 생성 파이프라인."""

from __future__ import annotations

import argparse
import random
import os
import json
import re
from typing import Dict, List, Optional, TypedDict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langgraph.graph import StateGraph, END

from fsm_engine import QwenFSMEngine
from embedding_utils import EmbeddingMemory


RE_JSON = re.compile(r"\{.*?\}", flags=re.DOTALL)

from utils import (
    clamp_int,
    extract_last_assistant_output,
    extract_recent_history,
    extract_single_quote,
    force_single_quote_line,
    has_question,
    is_player_subject,
    normalize_protagonist_refs,
    normalize_space,
    normalize_user_refs,
    parse_genre,
    parse_player_name,
    parse_protagonist_name,
    parse_protagonist_speech_formal,
    parse_relation_status,
    parse_sexual_condition_relation,
    remove_all_quotes,
    resolve_allow_sexual,
    strip_line_labels_multiline,
    strip_meta,
    strip_square_brackets,
)
from prompts import (
    build_assistant_rewrite_crisis_prompt,
    build_assistant_rewrite_prompt,
    build_assistant_rewrite_sexual_prompt,
    build_assistant_turn_prompt,
    build_crisis_prompt,
    build_eval_internal_prompt,
    build_eval_prompt,
    build_sexual_prompt,
    build_system_messages,
    build_user_prompt,
    build_user_sexual_prompt,
    dialogue_style_rule_text,
)
from backend import generate_text, generate_json
from validators import (
    assistant_invalid_reason_integrated,
    crisis_invalid_reason,
    detect_sexual_request,
    is_redundant_turn_dialogue,
    llm_dialogue_quality_fail_local,
    llm_role_conflict_fail_local,
    llm_stall_detect_local,
    llm_user_role_conflict_fail_local,
    split_integrated_assistant,
    user_invalid_reason,
)

class VNState(TypedDict):
    """그래프 노드 간 공유되는 상태 스키마를 정의한다."""
    system_lore: str
    protagonist: str
    player_name: str
    relation_status: str
    protagonist_speech_formal: Optional[bool]
    sexual_condition_rel: str
    relation_intent: str
    action_state: str
    last_user_question: bool
    last_asst_question: bool
    last_asst_sexual: bool
    action_state_hist: List[str]
    crisis_lock: int
    crisis_turns: int
    sexual_turns: int
    sexual_lock: int
    aftermath_turns: int
    user_idle_streak: int
    stall_count: int
    last_user_intent: str
    last_asst_intent: str
    last_user_signature: str
    last_asst_signature: str
    last_user_dialogue: str
    last_asst_dialogue: str

    turns_target: int
    turn_index: int

    messages: List[Dict[str, str]]
    user_text: str
    assistant_text: str
    raw_assistant: str

    narration_text: str
    dialogue_text: str

    sexual_request: bool
    allow_sexual: bool
    sexual_ready: bool

    signals: Dict[str, int]

    retry_user: int
    retry_assistant: int
    retry_rewrite: int
    retry_quality: int
    retry_sexual: int
    retry_eval: int
    retry_crisis: int

    eval_internal_text: str

    rewrite_mode: str

    error: str


RELATION_INTIMACY_MAP = {"적대": 0, "거리감": 1, "친밀": 2, "사랑": 3}
REWRITE_MAX_ATTEMPTS = 3
ACTION_STATES = ["IDLE", "EVENT", "CONFLICT", "RESOLUTION", "AFTERMATH"]
SEXUAL_STATES = {"SEXUAL_1", "SEXUAL_2", "SEXUAL_3", "SEXUAL_4"}
AFTERMATH_SEX_STATES = {"AFTERMATH_SEX_1", "AFTERMATH_SEX_2"}
CRISIS_STATES = {"CRISIS_1", "CRISIS_2"}
AFTERMATH_CRISIS_STATES = {"AFTERMATH_CRISIS_1"}

# Sexual/aftermath pacing controls
MIN_SEXUAL_TURNS = 4
# crisis 국면은 최소 2턴 유지해 급격한 이탈을 막는다.
MIN_CRISIS_TURNS = 2
AFTERMATH_TURNS_MIN = 1
AFTERMATH_TURNS_MAX = 2

GENRE_ACTION_BOOST = {
    "연애물": {"EVENT": 0.3, "RESOLUTION": 0.2},
    "육성물": {"EVENT": 0.3, "RESOLUTION": 0.2},
    "성장물": {"CONFLICT": 0.4, "RESOLUTION": 0.3},
    "비극": {"CONFLICT": 0.4, "AFTERMATH": 0.3},
    "심리 시뮬레이션": {"CONFLICT": 0.3, "AFTERMATH": 0.3},
}

def _pick_action_state(
    current: str,
    history: list,
    genre: str,
    sexual_ready: bool,
) -> str:
    """현재 상태와 최근 이력을 기반으로 `_pick_action_state` 후보를 선택한다."""
    candidates = list(ACTION_STATES)
    if sexual_ready:
        candidates.append("SEXUAL")

    if len(history) >= 2 and history[-1] == history[-2]:
        if history[-1] in candidates and len(candidates) > 1:
            candidates.remove(history[-1])

    weights = {s: 1.0 for s in candidates}

    recent = history[-4:]
    for s in recent:
        if s in weights:
            weights[s] -= 0.25

    for s in candidates:
        if recent.count(s) == 0:
            weights[s] += 0.5

    boosts = GENRE_ACTION_BOOST.get(genre, {})
    for s, b in boosts.items():
        if s in weights:
            weights[s] += b

    if current in weights:
        weights[current] += 0.1

    for s in list(weights.keys()):
        if weights[s] <= 0:
            weights[s] = 0.1

    total = sum(weights.values())
    r = random.random() * total
    acc = 0.0
    for s, w in weights.items():
        acc += w
        if r <= acc:
            return s
    return current if current in candidates else candidates[0]


def log_retry(node: str, retry: int, reason: str, max_retry: int):
    """실행 중 상태를 `log_retry` 형식 로그로 출력한다."""
    print(
        f"[RETRY] node={node} retry={retry}/{max_retry} reason={reason}",
        flush=True
    )


def log_step(node: str, msg: str):
    """실행 중 상태를 `log_step` 형식 로그로 출력한다."""
    print(f"[STEP] {node}: {msg}", flush=True)


def log_text_payload(text: str) -> str:
    """멀티라인 텍스트를 로그 한 줄 형태로 변환한다."""
    return (text or "").replace("\n", "\\n")


def _extract_dialogue_line(text: str) -> str:
    """2줄 출력에서 대사 라인을 추출한다."""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if len(lines) >= 2:
        return remove_all_quotes(lines[-1]).strip()
    if lines:
        return remove_all_quotes(lines[0]).strip()
    return ""


def _intent_key(dialogue: str) -> str:
    """대사 문장을 낮은 해상도 의도 키로 정규화한다."""
    t = normalize_space(dialogue)
    if not t:
        return ""
    if "?" in t or "까" in t or "나요" in t:
        return "question"
    if any(x in t for x in ("하자", "해보자", "결정", "정하")):
        return "decision"
    if any(x in t for x in ("해줘", "해주세요", "부탁", "원해", "필요")):
        return "request"
    return "statement"


def _content_signature(dialogue: str) -> str:
    """반복 감지를 위한 간단한 내용 서명 문자열을 만든다."""
    t = normalize_space(dialogue)
    if not t:
        return ""
    t = re.sub(r"[\"'“”‘’!?.,…\s]", "", t)
    return t[:40]


def _must_force_progress(state: VNState) -> bool:
    """정체 감지 시 강한 전개 제약을 적용할지 반환한다."""
    return bool(state.get("stall_count", 0) >= 1 or state.get("user_idle_streak", 0) >= 1)


def choose_relation_intent(state: VNState, fsm_state: str, action_state: str) -> str:
    """
    관계 변화 의도 결정

    매 턴 랜덤 의도를 주면 대화 결이 흔들리므로, 국면/신호 기반 기본값을 사용하고
    소량 확률만 섞어 전개의 자연스러움을 유지한다.

    Args:
        state: 현재 VN 실행 상태.
        fsm_state: 관계 FSM 상태.
        action_state: 액션 FSM 상태.

    Returns:
        str: "관계 상승" 또는 "관계 악화".
    """
    if action_state.startswith("SEXUAL_"):
        return "관계 상승"
    signals = state.get("signals", {}) or {}
    threat = int(signals.get("threat", 0) or 0)
    pressure = int(signals.get("pressure", 0) or 0)
    instability = int(signals.get("mental_instability", 0) or 0)
    if fsm_state == "CRISIS" or threat >= 1 or pressure >= 2 or instability >= 2:
        return "관계 악화" if random.random() < 0.8 else "관계 상승"
    return "관계 상승" if random.random() < 0.85 else "관계 악화"


def build_graph(
    model,
    tokenizer,
    fsm_rel: QwenFSMEngine,
    fsm_act: QwenFSMEngine,
    embed_memory: EmbeddingMemory,
) -> Any:
    """실행 맥락을 반영해 `build_graph` 결과 문자열을 구성해 반환한다."""
    def init_node(state: VNState) -> VNState:
        """
        초기 상태 구성

        시나리오북에서 주인공/플레이어/관계 정보를 파싱하고,
        메시지 버퍼와 신호값, 재시도 카운터를 학습 가능한 기본값으로 초기화한다.

        Args:
            state: 그래프가 전달한 VN 실행 상태 딕셔너리.

        Returns:
            VNState: 초기화가 반영된 상태 객체.
        """
        log_step("INIT", "start")
        protagonist_name = parse_protagonist_name(state["system_lore"])
        player_name = parse_player_name(state["system_lore"])
        system_msgs = build_system_messages(state["system_lore"])

        state["protagonist"] = protagonist_name
        state["player_name"] = player_name
        state["relation_status"] = parse_relation_status(state["system_lore"])
        # 주인공 말투는 INIT에서 1회만 확정해 전 턴에서 동일 기준을 사용한다.
        # 노드마다 system_lore를 재파싱하면 말투 기준이 흔들려 첫 턴부터 붕괴가 발생할 수 있다.
        parsed_protagonist_style = parse_protagonist_speech_formal(state["system_lore"])
        state["protagonist_speech_formal"] = parsed_protagonist_style
        flags = fsm_rel.get_flags()
        rel_status = flags.get("relation_status")
        if rel_status:
            state["relation_status"] = rel_status
        state["sexual_condition_rel"] = parse_sexual_condition_relation(state["system_lore"])
        state["action_state"] = fsm_act.get_state()
        state["messages"] = [*system_msgs]
        state["user_text"] = ""
        state["assistant_text"] = ""
        state["raw_assistant"] = ""
        state["narration_text"] = ""
        state["dialogue_text"] = ""

        state["sexual_request"] = False
        state["allow_sexual"] = False
        state["sexual_ready"] = False
        state["last_user_question"] = False
        state["last_asst_question"] = False
        state["last_asst_sexual"] = False

        state["signals"] = {
            "mental_instability": 0,
            "intimacy": RELATION_INTIMACY_MAP.get(state["relation_status"], 0),
            "threat": 0,
            "pressure": 0,
            "probe": 0,
            "resolve": 0,
            "event": 0,
        }

        state["turn_index"] = 0
        state["retry_user"] = 0
        state["retry_assistant"] = 0
        state["retry_rewrite"] = 0
        state["retry_quality"] = 0
        state["retry_sexual"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["rewrite_mode"] = ""
        state["error"] = ""
        log_step(
            "INIT",
            f"protagonist_speech_style="
            f"{'formal' if state['protagonist_speech_formal'] is True else ('informal' if state['protagonist_speech_formal'] is False else 'unset')}",
        )
        # 이름 파싱 결과를 초기 로그에 남겨, 역할 혼동 원인(이름 추출 실패/역전)을 즉시 확인한다.
        log_step("INIT", f"names protagonist={state['protagonist']} player={state['player_name']}")
        log_step("INIT", "done")
        return state

    def gen_user_node(state: VNState) -> VNState:
        """
        일반 user 턴 생성

        최근 이력과 FSM 상태를 바탕으로 플레이어 턴을 생성하고,
        형식/역할/반복/진행성 검증을 통과한 경우 메시지 히스토리에 반영한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: user 생성 결과 또는 재시도 오류가 반영된 상태.
        """
        log_step("GEN_USER", "start")

        history = extract_recent_history(state["messages"], max_turns=4)
        last_assistant = extract_last_assistant_output(state["messages"])
        log_step("GEN_USER", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()

        relation_intent = choose_relation_intent(state, fsm_state, action_state)
        state["relation_intent"] = relation_intent
        genre = parse_genre(state["system_lore"])
        task = build_user_prompt(
            fsm_state=fsm_state,
            history=history,
            protagonist=state["protagonist"],
            player_name=state["player_name"],
            last_assistant=last_assistant,
            relation_status=state.get("relation_status", ""),
            relation_intent=relation_intent,
            genre=genre,
            action_state=action_state,
            # user 말투 기준은 INIT에서 고정된 값을 그대로 사용한다.
            ban_question=bool(state.get("last_user_question", False)),
            ban_silence=bool(state.get("retry_user", 0) > 0),
            force_progress=_must_force_progress(state),
        )

        # v5 동작과 동일하게 user 생성 샘플링을 고정한다.
        log_step("GEN_USER", "sampling=temp:0.65,top_p:0.85")

        # task는 user role로 주입한다 (system으로 넣지 말 것)
        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=170,
            temperature=0.65,
            top_p=0.85,
            repetition_penalty=1.15,
        )
        log_step("GEN_USER", "generated")
        raw_lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not raw_lines:
            state["error"] = "user_generation_empty"
            log_step("GEN_USER", "empty")
            return state

        q = extract_single_quote(raw)
        if not q or q.strip() in ("...", "…"):
            state["retry_user"] += 1
            state["error"] = "user_generation_silent"
            log_step("GEN_USER", "silent")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        speech = f"\"{q}\""
        first = raw_lines[0] if raw_lines else ""
        if not first:
            first = "잠시 망설인다."
        if '"' in first:
            text = speech
        else:
            text = first + "\n" + speech

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = user_invalid_reason(
            text, player_name=state.get("player_name", ""), protagonist=state.get("protagonist", "")
        )
        if err:
            state["retry_user"] += 1
            state["error"] = err
            log_step("GEN_USER", f"invalid={err}")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=err,
            )
            return state
        if llm_user_role_conflict_fail_local(
            model=model,
            tokenizer=tokenizer,
            system_lore=state.get("system_lore", ""),
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            user_text=text,
        ):
            state["retry_user"] += 1
            state["error"] = "user_role_conflict_llm"
            log_step("GEN_USER", "invalid=user_role_conflict_llm")
            log_retry(
                node="GEN_USER",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        curr_user_dia = extract_single_quote(text) or ""
        if is_redundant_turn_dialogue(curr_user_dia, state.get("last_user_dialogue", "")):
            state["retry_user"] += 1
            state["error"] = "user_repeat_turn"
            log_step("GEN_USER_SEXUAL", "repeat=user_dialogue")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        state["user_text"] = text
        state["last_user_question"] = has_question(text)
        state["messages"].append({"role": "user", "content": text})
        state["sexual_request"] = detect_sexual_request(text)
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        narr_u = lines[0] if len(lines) >= 1 else ""
        dia_u = lines[-1] if len(lines) >= 2 else ""
        if narr_u:
            embed_memory.add(narr_u, kind="user_action")
        if dia_u and dia_u not in ("\"...\"", "\"…\""):
            embed_memory.add(dia_u, kind="user_dialogue")
        embed_memory.add(text, kind="user")

        state["retry_assistant"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("GEN_USER", f"output={log_text_payload(text)}")
        log_step("GEN_USER", "done")
        return state

    def gen_user_sexual_node(state: VNState) -> VNState:
        """
        sexual 국면 user 턴 생성

        sexual 전용 프롬프트로 user 턴을 만들고,
        일반 user 생성과 동일한 형식/반복/진행성 검증을 거쳐 상태에 기록한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: sexual user 생성 결과가 반영된 상태.
        """
        log_step("GEN_USER_SEXUAL", "start")

        history = extract_recent_history(state["messages"], max_turns=4)
        last_assistant = extract_last_assistant_output(state["messages"])
        log_step("GEN_USER_SEXUAL", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()

        relation_intent = choose_relation_intent(state, fsm_state, action_state)
        state["relation_intent"] = relation_intent
        genre = parse_genre(state["system_lore"])
        task = build_user_sexual_prompt(
            fsm_state=fsm_state,
            history=history,
            protagonist=state["protagonist"],
            player_name=state["player_name"],
            last_assistant=last_assistant,
            relation_status=state.get("relation_status", ""),
            relation_intent=relation_intent,
            genre=genre,
            action_state=action_state,
            ban_question=bool(state.get("last_user_question", False)),
            ban_silence=bool(state.get("retry_user", 0) > 0),
            force_progress=_must_force_progress(state),
        )

        # v5 동작과 동일하게 sexual user 생성 샘플링을 고정한다.
        log_step("GEN_USER_SEXUAL", "sampling=temp:0.65,top_p:0.85")

        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=170,
            temperature=0.65,
            top_p=0.85,
            repetition_penalty=1.15,
        )
        log_step("GEN_USER_SEXUAL", "generated")
        raw_lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not raw_lines:
            state["error"] = "user_generation_empty"
            log_step("GEN_USER_SEXUAL", "empty")
            return state
        q = extract_single_quote(raw)
        if not q or q.strip() in ("...", "…"):
            state["retry_user"] += 1
            state["error"] = "user_generation_silent"
            log_step("GEN_USER_SEXUAL", "silent")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        speech = f"\"{q}\""
        first = raw_lines[0] if raw_lines else ""
        if not first:
            first = "잠시 망설인다."
        if '"' in first:
            text = speech
        else:
            text = first + "\n" + speech

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        err = user_invalid_reason(
            text, player_name=state.get("player_name", ""), protagonist=state.get("protagonist", "")
        )
        if err:
            state["retry_user"] += 1
            state["error"] = err
            log_step("GEN_USER_SEXUAL", f"invalid={err}")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=err,
            )
            return state
        if llm_user_role_conflict_fail_local(
            model=model,
            tokenizer=tokenizer,
            system_lore=state.get("system_lore", ""),
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            user_text=text,
        ):
            state["retry_user"] += 1
            state["error"] = "user_role_conflict_llm"
            log_step("GEN_USER_SEXUAL", "invalid=user_role_conflict_llm")
            log_retry(
                node="GEN_USER_SEXUAL",
                retry=state["retry_user"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        state["user_text"] = text
        state["last_user_question"] = has_question(text)
        state["messages"].append({"role": "user", "content": text})
        state["sexual_request"] = detect_sexual_request(text)
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        narr_u = lines[0] if len(lines) >= 1 else ""
        dia_u = lines[-1] if len(lines) >= 2 else ""
        if narr_u:
            embed_memory.add(narr_u, kind="user_action")
        if dia_u and dia_u not in ("\"...\"", "\"…\""):
            embed_memory.add(dia_u, kind="user_dialogue")
        embed_memory.add(text, kind="user")

        state["retry_assistant"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("GEN_USER_SEXUAL", f"output={log_text_payload(text)}")
        log_step("GEN_USER_SEXUAL", "done")
        return state

    def detect_node(state: VNState) -> VNState:
        """입력 텍스트에서 `detect_node` 조건 충족 여부를 판별한다."""
        """입력 텍스트에서 `detect_node` 조건 충족 여부를 판별한다."""
        log_step("DETECT", "start")
        flags = fsm_rel.get_flags() or {}
        allow_flag = flags.get("allow_sexual", False)
        state["allow_sexual"] = resolve_allow_sexual(allow_flag, state["system_lore"])
        state["relation_status"] = parse_relation_status(state["system_lore"])
        rel_status = flags.get("relation_status")
        if rel_status:
            state["relation_status"] = rel_status
        state["sexual_condition_rel"] = parse_sexual_condition_relation(state["system_lore"])
        prev_action_state = state.get("action_state", "")
        if (
            prev_action_state not in SEXUAL_STATES
            and prev_action_state not in AFTERMATH_SEX_STATES
            and prev_action_state not in CRISIS_STATES
            and prev_action_state not in AFTERMATH_CRISIS_STATES
        ):
            state["action_state"] = fsm_act.get_state()
        state["sexual_ready"] = (
            bool(state["allow_sexual"])
            and bool(state["sexual_condition_rel"])
            and bool(state["relation_status"])
            and (state["sexual_condition_rel"] == state["relation_status"])
        )
        if prev_action_state in SEXUAL_STATES:
            state["sexual_ready"] = True
        log_step("DETECT", "done")
        return state

    def gen_assistant_node(state: VNState) -> VNState:
        """
        일반 assistant 턴 생성

        assistant 프롬프트로 서술+대사 2줄 출력을 생성하고,
        말투/역할/레인/반복/진행성 검증을 통과하면 메모리와 메시지 상태를 갱신한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: assistant 생성 성공 또는 재시도 상태가 반영된 객체.
        """
        log_step("GEN_ASSISTANT", "start")

        history = extract_recent_history(state["messages"], max_turns=4)
        log_step("GEN_ASSISTANT", f"history_lines={len(history.splitlines())}")
        fsm_state = fsm_rel.get_state()
        action_state = fsm_act.get_state()
        flags = fsm_rel.get_flags() or {}

        task = build_assistant_turn_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_state,
            flags=flags,
            signals=state["signals"],
            relation_status=state.get("relation_status", ""),
            genre=parse_genre(state["system_lore"]),
            action_state=action_state,
            history=history,
            user_text=state["user_text"],
            sexual_request=bool(state["sexual_request"]),
            player_name=state["player_name"],
            force_progress=_must_force_progress(state),
            prefer_formal=state.get("protagonist_speech_formal"),
        )

        # v5와 동일하게 assistant 기본 샘플링을 고정한다.
        log_step("GEN_ASSISTANT", "sampling=temp:0.70,top_p:0.85")
        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=140,
            temperature=0.70,
            top_p=0.85,
            repetition_penalty=1.12,
        )
        state["raw_assistant"] = raw
        log_step("GEN_ASSISTANT", "generated")

        if not normalize_space(raw):
            state["error"] = "assistant_generation_empty"
            log_step("GEN_ASSISTANT", "empty")
            return state

        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        if normalize_space(text) == normalize_space(state.get("user_text", "")):
            state["retry_assistant"] += 1
            state["error"] = "asst_echo_user"
            log_step("GEN_ASSISTANT", "invalid=asst_echo_user")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        narr_line = ""
        _lines = [x.strip() for x in text.splitlines() if x.strip()]
        if _lines:
            narr_line = _lines[0]
        if is_player_subject(narr_line, state.get("player_name", "")):
            state["retry_assistant"] += 1
            state["error"] = "asst_role_conflict"
            state["rewrite_mode"] = "normal"
            log_step("GEN_ASSISTANT", "invalid=asst_role_conflict")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=state.get("protagonist_speech_formal"),
        )
        if err:
            state["retry_assistant"] += 1
            state["error"] = err
            if err in ("asst_meta",):
                state["rewrite_mode"] = "normal"
            log_step("GEN_ASSISTANT", f"invalid={err}")
            log_retry(
                node="GEN_ASSISTANT",
                retry=state["retry_assistant"],
                max_retry=30,
                reason=err,
            )
            return state
        narr, dia, merged = split_integrated_assistant(text)
        # MOD(중요): kind별 repetition 검사, keyword-only 준수
        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        if narr:
            embed_memory.add(narr, kind="narration")
        if dia and dia not in ("\"...\"", "\"…\""):
            embed_memory.add(dia, kind="dialogue")
        embed_memory.add(merged, kind="assistant")
        state["error"] = ""
        log_step("GEN_ASSISTANT", f"output={log_text_payload(merged)}")
        log_step("GEN_ASSISTANT", "done")
        return state

    def gen_assistant_rewrite_node(state: VNState) -> VNState:
        """
        assistant 1차 리라이트

        생성 실패 원인(말투/메타/역할 충돌 등)을 보정하기 위해
        기존 assistant 출력을 rewrite 프롬프트로 재작성하고 재검증한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: rewrite 결과가 반영된 상태, 또는 실패 오류가 기록된 상태.
        """
        log_step("GEN_ASSISTANT_REWRITE", "start")

        raw_text = (state.get("raw_assistant") or state.get("assistant_text") or "").strip()
        if not raw_text:
            state["error"] = "rewrite_empty"
            log_step("GEN_ASSISTANT_REWRITE", "empty")
            return state

        rewrite_mode = (state.get("rewrite_mode") or "normal").strip().lower()
        prefer_formal = state.get("protagonist_speech_formal")
        if rewrite_mode == "sexual":
            task = build_assistant_rewrite_sexual_prompt(
                system_lore=state["system_lore"],
                protagonist=state["protagonist"],
                player_name=state["player_name"],
                raw_text=raw_text,
                action_state=state.get("action_state", ""),
                genre=parse_genre(state["system_lore"]),
                prefer_formal=prefer_formal,
            )
        elif rewrite_mode == "crisis":
            task = build_assistant_rewrite_crisis_prompt(
                system_lore=state["system_lore"],
                protagonist=state["protagonist"],
                player_name=state["player_name"],
                raw_text=raw_text,
                action_state=state.get("action_state", ""),
                genre=parse_genre(state["system_lore"]),
                prefer_formal=prefer_formal,
            )
        else:
            task = build_assistant_rewrite_prompt(
                system_lore=state["system_lore"],
                protagonist=state["protagonist"],
                player_name=state["player_name"],
                raw_text=raw_text,
                action_state=state.get("action_state", ""),
                genre=parse_genre(state["system_lore"]),
                prefer_formal=prefer_formal,
            )

        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=120,
            temperature=0.40,
            top_p=0.80,
            repetition_penalty=1.08,
        )
        log_step("GEN_ASSISTANT_REWRITE", "generated")

        if not normalize_space(raw):
            state["retry_rewrite"] += 1
            state["error"] = "rewrite_empty"
            log_step("GEN_ASSISTANT_REWRITE", "invalid=rewrite_empty")
            return state
        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        if rewrite_mode == "crisis":
            err = crisis_invalid_reason(text, player_name=state.get("player_name", ""))
        else:
            err = assistant_invalid_reason_integrated(
                text,
                protagonist=state.get("protagonist", ""),
                player_name=state.get("player_name", ""),
                prefer_formal=state.get("protagonist_speech_formal"),
            )
        if err:
            state["retry_rewrite"] += 1
            state["error"] = err
            log_step("GEN_ASSISTANT_REWRITE", f"invalid={err}")
            return state

        if rewrite_mode == "crisis":
            state["narration_text"] = ""
            state["dialogue_text"] = ""
            state["assistant_text"] = text
            state["last_asst_question"] = has_question(text)
            state["last_asst_sexual"] = False
            embed_memory.add(text, kind="assistant")
            out_text = text
        else:
            narr, dia, merged = split_integrated_assistant(text)
            state["narration_text"] = narr
            state["dialogue_text"] = dia
            state["assistant_text"] = merged
            state["last_asst_question"] = has_question(dia)
            state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
            if narr:
                embed_memory.add(narr, kind="narration")
            if dia and dia not in ("\"...\"", "\"…\""):
                embed_memory.add(dia, kind="dialogue")
            embed_memory.add(merged, kind="assistant")
            out_text = merged
        state["rewrite_mode"] = ""
        state["error"] = ""
        log_step("GEN_ASSISTANT_REWRITE", f"output={log_text_payload(out_text)}")
        log_step("GEN_ASSISTANT_REWRITE", "done")
        return state

    def eval_dialogue_quality_node(state: VNState) -> VNState:
        """
        대화 품질 LLM 평가

        assistant 출력이 역할 충돌/품질 저하를 포함하는지 별도 평가 모델로 확인하고,
        실패 시 quality rewrite 분기로 보낼 오류 코드를 기록한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: 품질 평가 결과가 반영된 상태.
        """
        log_step("EVAL_DIALOGUE_QUALITY", "start")
        text = (state.get("assistant_text") or "").strip()
        if not text:
            state["error"] = ""
            log_step("EVAL_DIALOGUE_QUALITY", "skip_empty")
            return state

        role_failed = llm_role_conflict_fail_local(
            model=model,
            tokenizer=tokenizer,
            system_lore=state.get("system_lore", ""),
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            user_text=state.get("user_text", ""),
            assistant_text=text,
        )
        if role_failed:
            # 품질 평가는 차단 게이트가 아니라 경고 신호로만 사용한다.
            log_step("EVAL_DIALOGUE_QUALITY", "warn=asst_role_conflict_llm")

        failed = llm_dialogue_quality_fail_local(
            model=model,
            tokenizer=tokenizer,
            system_lore=state.get("system_lore", ""),
            relation_status=state.get("relation_status", ""),
            action_state=state.get("action_state", ""),
            player_name=state.get("player_name", ""),
            user_text=state.get("user_text", ""),
            assistant_text=text,
        )
        if failed:
            log_step("EVAL_DIALOGUE_QUALITY", "warn=asst_dialogue_quality_llm")

        state["error"] = ""
        log_step("EVAL_DIALOGUE_QUALITY", "done")
        return state

    def gen_crisis_node(state: VNState) -> VNState:
        """
        crisis 국면 assistant 생성

        crisis 전용 프롬프트로 긴장 국면 출력을 만들고,
        crisis 형식 검증 및 반복 검증을 통과한 텍스트를 assistant 출력으로 반영한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: crisis 출력 또는 재시도 오류가 반영된 상태.
        """
        log_step("GEN_CRISIS", "start")
        state["retry_crisis"] += 1

        history = extract_recent_history(state["messages"], max_turns=4)
        log_step("GEN_CRISIS", f"history_lines={len(history.splitlines())}")

        task = build_crisis_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            history=history,
            user_text=state["user_text"],
            player_name=state["player_name"],
            force_progress=_must_force_progress(state),
            genre=parse_genre(state["system_lore"]),
            action_state=state.get("action_state", ""),
        )

        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=256,
            temperature=0.60,
            top_p=0.80,
            repetition_penalty=1.06,
            no_repeat_ngram_size=4,
        )
        text = raw.strip()
        state["raw_assistant"] = raw
        log_step("GEN_CRISIS", "generated")

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)
        err = crisis_invalid_reason(text, player_name=state.get("player_name", ""))
        if err:
            state["error"] = err
            log_step("GEN_CRISIS", f"invalid={err}")
            log_retry(
                node="GEN_CRISIS",
                retry=state["retry_crisis"],
                max_retry=30,
                reason=err,
            )
            return state

        state["assistant_text"] = text
        # CRISIS는 narration/dialogue 분리 안함
        state["narration_text"] = ""
        state["dialogue_text"] = ""
        embed_memory.add(text, kind="assistant")
        state["error"] = ""
        log_step("GEN_CRISIS", f"output={log_text_payload(text)}")
        log_step("GEN_CRISIS", "done")
        return state

    def gen_sexual_node(state: VNState) -> VNState:
        """
        sexual 국면 assistant 생성

        sexual 전용 프롬프트로 2줄 출력을 생성하고,
        역할/말투/레인/반복/진행성 검증 후 assistant 상태와 임베딩 메모리를 갱신한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: sexual 출력 또는 재시도 오류가 반영된 상태.
        """
        log_step("GEN_SEXUAL", "start")
        history = extract_recent_history(state["messages"], max_turns=4)
        log_step("GEN_SEXUAL", f"history_lines={len(history.splitlines())}")

        task = build_sexual_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            history=history,
            user_text=state["user_text"],
            player_name=state["player_name"],
            force_progress=_must_force_progress(state),
            genre=parse_genre(state["system_lore"]),
            action_state=state.get("action_state", ""),
        )
        style_rule = dialogue_style_rule_text(
            state.get("protagonist_speech_formal")
        )
        if style_rule:
            task = task + "\n\n[말투 규칙]\n- " + style_rule

        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=state["messages"] + [{"role": "user", "content": task}],
            max_new_tokens=180,
            temperature=0.68,
            top_p=0.85,
            repetition_penalty=1.08,
        )
        lines = [x for x in strip_line_labels_multiline(raw).splitlines() if x.strip()]
        if not lines:
            state["error"] = "assistant_generation_empty"
            log_step("GEN_SEXUAL", "empty")
            return state
        if len(lines) >= 2:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + force_single_quote_line(lines[1])).strip()
        elif len(lines) == 1:
            narr_line = remove_all_quotes(lines[0]).strip()
            text = (narr_line + "\n" + "\"...\"").strip()
        else:
            text = "숨이 짧게 새어 나왔다.\n\"...\""
        log_step("GEN_SEXUAL", "generated")

        text = strip_meta(text)
        text = strip_square_brackets(text)
        text = normalize_user_refs(text, state.get("player_name", "{{user}}"))
        text = normalize_protagonist_refs(text, state.get("protagonist", ""))
        text = strip_line_labels_multiline(text)

        narr_line = ""
        _lines = [x.strip() for x in text.splitlines() if x.strip()]
        if _lines:
            narr_line = _lines[0]
        if is_player_subject(narr_line, state.get("player_name", "")):
            state["retry_sexual"] += 1
            state["error"] = "asst_role_conflict"
            state["rewrite_mode"] = "sexual"
            log_step("GEN_SEXUAL", "invalid=asst_role_conflict")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=state["error"],
            )
            return state

        err = assistant_invalid_reason_integrated(
            text,
            protagonist=state.get("protagonist", ""),
            player_name=state.get("player_name", ""),
            prefer_formal=state.get("protagonist_speech_formal"),
        )
        if err:
            state["retry_sexual"] += 1
            state["error"] = err
            log_step("GEN_SEXUAL", f"invalid={err}")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=err,
            )
            return state
        narr, dia, merged = split_integrated_assistant(text)
        if is_redundant_turn_dialogue(dia, state.get("last_asst_dialogue", "")):
            state["retry_sexual"] += 1
            state["error"] = "asst_repeat_turn"
            state["rewrite_mode"] = "sexual"
            log_step("GEN_SEXUAL", "repeat=asst_dialogue")
            log_retry(
                node="GEN_SEXUAL",
                retry=state["retry_sexual"],
                max_retry=30,
                reason=state["error"],
            )
            return state
        state["narration_text"] = narr
        state["dialogue_text"] = dia
        state["assistant_text"] = merged
        state["last_asst_question"] = has_question(dia)
        state["last_asst_sexual"] = state.get("action_state", "").startswith("SEXUAL_")
        if narr:
            embed_memory.add(narr, kind="narration")
        if dia and dia not in ("\"...\"", "\"…\""):
            embed_memory.add(dia, kind="dialogue")
        embed_memory.add(merged, kind="assistant")
        state["error"] = ""
        log_step("GEN_SEXUAL", f"output={log_text_payload(merged)}")
        log_step("GEN_SEXUAL", "done")
        return state

    def eval_internal_node(state: VNState) -> VNState:
        """
        내부 해설 평가 생성

        직전 user/assistant 턴을 바탕으로 내부 평가 텍스트를 생성해
        신호 JSON 평가 전에 참고용 분석 컨텍스트를 채운다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: eval_internal_text가 반영된 상태.
        """
        log_step("EVAL_INTERNAL", "start")

        if (
            (not state["messages"])
            or (state["messages"][-1]["role"] != "assistant")
            or (state["messages"][-1]["content"] != state["assistant_text"])
        ):
            state["messages"].append({"role": "assistant", "content": state["assistant_text"]})

        task = build_eval_internal_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_rel.get_state(),
            prev_signals=state["signals"],
            user_text=state["user_text"],
            assistant_text=state["assistant_text"],
            player_name=state["player_name"],
            relation_intent=state.get("relation_intent", ""),
        )

        eval_messages = [
            {"role": "system", "content": "You are an evaluator. Provide brief analysis only."},
            {"role": "user", "content": task},
        ]

        analysis = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=eval_messages,
            max_new_tokens=450,
            temperature=0.3,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=True,
        )

        state["eval_internal_text"] = (analysis or "").strip()
        if state["eval_internal_text"]:
            print(f"[EVAL_INTERNAL] {state['eval_internal_text']}", flush=True)
        log_step("EVAL_INTERNAL", "done")
        return state

    def eval_json_node(state: VNState) -> VNState:
        """
        상태 신호 JSON 평가

        평가 프롬프트로 mental/intimacy/threat 등 신호를 JSON으로 생성하고,
        장르 드리프트와 증감 제한 규칙을 적용해 다음 FSM 입력 신호를 확정한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: signals 필드가 갱신된 상태.
        """
        log_step("EVAL_JSON", "start")
        state["retry_eval"] += 1

        task = build_eval_prompt(
            system_lore=state["system_lore"],
            protagonist=state["protagonist"],
            fsm_state=fsm_rel.get_state(),
            prev_signals=state["signals"],
            user_text=state["user_text"],
            assistant_text=state["assistant_text"],
            player_name=state["player_name"],
            relation_intent=state.get("relation_intent", ""),
            analysis_text=state.get("eval_internal_text", ""),
        )

        eval_messages = [
            {"role": "system", "content": "You are a JSON-only evaluator."},
            {"role": "user", "content": task},
        ]

        obj = generate_json(
            model=model,
            tokenizer=tokenizer,
            messages=eval_messages,
            max_new_tokens=256,
            temperature=0.1,
            top_p=1.0,
            do_sample=False,
            log_raw=True,
        )

        if obj is None:
            state["error"] = "eval_json_parse_fail"
            log_step("EVAL_JSON", "json_parse_fail")
            log_retry(
                node="EVAL_JSON",
                retry=state["retry_eval"],
                max_retry=40,
                reason="json_parse_fail",
            )
            return state

        new_signals = {
            "mental_instability": clamp_int(obj.get("mental_instability"), 0, 3, state["signals"].get("mental_instability", 0)),
            "intimacy": clamp_int(obj.get("intimacy"), 0, 3, state["signals"].get("intimacy", 0)),
            "threat": clamp_int(obj.get("threat"), 0, 2, state["signals"].get("threat", 0)),
            "pressure": clamp_int(obj.get("pressure"), 0, 3, state["signals"].get("pressure", 0)),
            "probe": clamp_int(obj.get("probe"), 0, 3, state["signals"].get("probe", 0)),
            "resolve": clamp_int(obj.get("resolve"), 0, 1, state["signals"].get("resolve", 0)),
            "event": clamp_int(obj.get("event"), 0, 3, state["signals"].get("event", 0)),
        }

        # - 관계 상승: 60% 상승 / 30% 유지 / 10% 하락
        # - 관계 악화: 60% 하락 / 30% 유지 / 10% 상승
        rel_intent = state.get("relation_intent", "")
        prev = state["signals"]
        prev_intimacy = prev.get("intimacy", 0)
        r = random.random()

        if rel_intent == "관계 상승":
            if r < 0.60:
                new_signals["intimacy"] = min(prev_intimacy + 1, 3)
            elif r < 0.90:
                new_signals["intimacy"] = prev_intimacy
            else:
                new_signals["intimacy"] = max(prev_intimacy - 1, 0)
        elif rel_intent == "관계 악화":
            if r < 0.60:
                new_signals["intimacy"] = max(prev_intimacy - 1, 0)
            elif r < 0.90:
                new_signals["intimacy"] = prev_intimacy
            else:
                new_signals["intimacy"] = min(prev_intimacy + 1, 3)

        genre = parse_genre(state["system_lore"])
        genre_weights = {
            "연애물": {
                "mental_instability": (0.15, 0.55, 0.30),
                "threat": (0.15, 0.55, 0.30),
                "pressure": (0.20, 0.50, 0.30),
                "probe": (0.25, 0.50, 0.25),
                "resolve": (0.20, 0.50, 0.30),
                "event": (0.35, 0.45, 0.20),
            },
            "육성물": {
                "mental_instability": (0.20, 0.55, 0.25),
                "threat": (0.15, 0.60, 0.25),
                "pressure": (0.35, 0.45, 0.20),
                "probe": (0.35, 0.45, 0.20),
                "resolve": (0.45, 0.40, 0.15),
                "event": (0.45, 0.40, 0.15),
            },
            "성장물": {
                "mental_instability": (0.30, 0.45, 0.25),
                "threat": (0.25, 0.50, 0.25),
                "pressure": (0.35, 0.45, 0.20),
                "probe": (0.25, 0.50, 0.25),
                "resolve": (0.40, 0.40, 0.20),
                "event": (0.50, 0.35, 0.15),
            },
            "비극": {
                "mental_instability": (0.50, 0.30, 0.20),
                "threat": (0.40, 0.35, 0.25),
                "pressure": (0.40, 0.35, 0.25),
                "probe": (0.20, 0.50, 0.30),
                "resolve": (0.15, 0.40, 0.45),
                "event": (0.45, 0.40, 0.15),
            },
            "심리 시뮬레이션": {
                "mental_instability": (0.45, 0.35, 0.20),
                "threat": (0.30, 0.45, 0.25),
                "pressure": (0.40, 0.40, 0.20),
                "probe": (0.40, 0.40, 0.20),
                "resolve": (0.25, 0.45, 0.30),
                "event": (0.30, 0.50, 0.20),
            },
        }

        def _drift_stat(key: str, lo: int, hi: int, probs: tuple[float, float, float]) -> None:
            """내부 헬퍼로 `_drift_stat` 계산 절차를 수행한다."""
            up, same, _ = probs
            v = clamp_int(new_signals.get(key), lo, hi, prev.get(key, 0))
            r2 = random.random()
            if r2 < up:
                v = min(v + 1, hi)
            elif r2 >= up + same:
                v = max(v - 1, lo)
            new_signals[key] = v

        weights = genre_weights.get(genre)
        if weights:
            _drift_stat("mental_instability", 0, 3, weights["mental_instability"])
            _drift_stat("threat", 0, 2, weights["threat"])
            _drift_stat("pressure", 0, 3, weights["pressure"])
            _drift_stat("probe", 0, 3, weights["probe"])
            _drift_stat("resolve", 0, 1, weights["resolve"])
            _drift_stat("event", 0, 3, weights["event"])

        def _limit_delta(key: str, lo: int, hi: int) -> int:
            """내부 헬퍼로 `_limit_delta` 계산 절차를 수행한다."""
            v = clamp_int(new_signals.get(key), lo, hi, prev.get(key, 0))
            p = prev.get(key, 0)
            if v > p + 1:
                return p + 1
            if v < p - 1:
                return p - 1
            return v

        new_signals["mental_instability"] = _limit_delta("mental_instability", 0, 3)
        new_signals["intimacy"] = _limit_delta("intimacy", 0, 3)
        new_signals["threat"] = _limit_delta("threat", 0, 2)
        new_signals["pressure"] = _limit_delta("pressure", 0, 3)
        new_signals["probe"] = _limit_delta("probe", 0, 3)
        new_signals["resolve"] = _limit_delta("resolve", 0, 1)
        new_signals["event"] = _limit_delta("event", 0, 3)
        # 정체 누적 시 다음 턴 전이를 위해 최소 1개 신호 변화를 강제한다.
        if int(state.get("stall_count", 0)) >= 1:
            prev_event = int(prev.get("event", 0))
            prev_pressure = int(prev.get("pressure", 0))
            if new_signals.get("event", prev_event) == prev_event:
                new_signals["event"] = min(prev_event + 1, 3)
            if (
                new_signals.get("event", prev_event) == prev_event
                and new_signals.get("pressure", prev_pressure) == prev_pressure
            ):
                new_signals["pressure"] = min(prev_pressure + 1, 3)

        state["signals"] = new_signals
        state["error"] = ""
        log_step("EVAL_JSON", "done")
        return state

    def fsm_step_node(state: VNState) -> VNState:
        """
        관계 FSM 및 액션 FSM 전이

        평가 신호를 기반으로 관계 FSM/액션 FSM을 전이시키고,
        sexual 최소 유지 턴, aftermath 전환, idle 보정 규칙을 반영해 action_state를 확정한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: action_state와 관련 카운터가 갱신된 상태.
        """
        log_step("FSM_STEP", "start")
        prev_action_state = state.get("action_state", "")
        action_hist = state.get("action_state_hist", [])
        # 정체가 누적되면 FSM 입력 신호를 상향 보정해 상태 전이를 가속한다.
        stall_level = int(state.get("stall_count", 0))
        fsm_signals = dict(state.get("signals", {}) or {})
        if stall_level >= 1:
            fsm_signals["event"] = max(int(fsm_signals.get("event", 0) or 0), 1)
            fsm_signals["probe"] = max(int(fsm_signals.get("probe", 0) or 0), 1)
        if stall_level >= 2:
            fsm_signals["event"] = max(int(fsm_signals.get("event", 0) or 0), 2)
            fsm_signals["pressure"] = max(int(fsm_signals.get("pressure", 0) or 0), 1)
        if stall_level >= 3:
            fsm_signals["resolve"] = max(int(fsm_signals.get("resolve", 0) or 0), 1)
        fsm_rel.step({
            "mental_instability": int(fsm_signals.get("mental_instability", 0)),
            "intimacy": int(fsm_signals.get("intimacy", 0)),
            "threat": int(fsm_signals.get("threat", 0)),
            "pressure": int(fsm_signals.get("pressure", 0)),
            "probe": int(fsm_signals.get("probe", 0)),
            "resolve": int(fsm_signals.get("resolve", 0)),
            "genre": parse_genre(state["system_lore"]),
        })
        fsm_act.step({
            "event": int(fsm_signals.get("event", 0)),
            "threat": int(fsm_signals.get("threat", 0)),
            "pressure": int(fsm_signals.get("pressure", 0)),
            "mental_instability": int(fsm_signals.get("mental_instability", 0)),
            "resolve": int(fsm_signals.get("resolve", 0)),
            "sexual_ready": bool(state.get("sexual_ready", False)),
        })
        # crisis는 관계 FSM 신호를 우선해 최소 유지 턴을 잠근다.
        crisis_lock = int(state.get("crisis_lock", 0))
        if fsm_rel.get_state() == "CRISIS":
            crisis_lock = MIN_CRISIS_TURNS
        elif crisis_lock > 0:
            crisis_lock = max(crisis_lock - 1, 0)
        base_action_state = fsm_act.get_state()
        log_step(
            "FSM_STEP",
            f"pre action_state={prev_action_state} base={base_action_state} "
            f"sexual_ready={state.get('sexual_ready', False)} "
            f"sexual_turns={state.get('sexual_turns', 0)} "
            f"aftermath_turns={state.get('aftermath_turns', 0)} "
            f"sexual_lock={state.get('sexual_lock', 0)} "
            f"crisis_lock={crisis_lock} stall_level={stall_level}",
        )

        sexual_turns = int(state.get("sexual_turns", 0))
        aftermath_turns = int(state.get("aftermath_turns", 0))
        sexual_lock = int(state.get("sexual_lock", 0))
        crisis_turns = int(state.get("crisis_turns", 0))
        if prev_action_state in SEXUAL_STATES or base_action_state in SEXUAL_STATES:
            sexual_turns = min(4, sexual_turns + 1)
            state["action_state"] = f"SEXUAL_{sexual_turns}"
            if sexual_lock <= 0:
                sexual_lock = MIN_SEXUAL_TURNS
            sexual_lock = max(sexual_lock - 1, 0)
            if sexual_turns >= MIN_SEXUAL_TURNS:
                state["action_state"] = "AFTERMATH_SEX_1"
                sexual_turns = 0
                aftermath_turns = 1
                sexual_lock = 0
            crisis_turns = 0
        elif prev_action_state in AFTERMATH_SEX_STATES or base_action_state in AFTERMATH_SEX_STATES:
            if prev_action_state == "AFTERMATH_SEX_1":
                aftermath_turns = 2
                state["action_state"] = "AFTERMATH_SEX_2"
            else:
                aftermath_turns = 0
                state["action_state"] = "IDLE"
            sexual_lock = 0
            crisis_turns = 0
        elif prev_action_state in CRISIS_STATES or base_action_state in CRISIS_STATES:
            # CRISIS는 action FSM의 현재 결과(base)를 우선한다.
            # base가 CRISIS가 아니면 이전 상태(prev)가 CRISIS여도 즉시 여파 단계로 이탈한다.
            if base_action_state not in CRISIS_STATES:
                state["action_state"] = "AFTERMATH_CRISIS_1"
                crisis_turns = 0
                crisis_lock = 0
            elif base_action_state == "CRISIS_2":
                state["action_state"] = "CRISIS_2"
                crisis_turns = max(crisis_turns, 2)
                if crisis_lock <= 0:
                    crisis_lock = MIN_CRISIS_TURNS
                crisis_lock = max(crisis_lock - 1, 0)
                if crisis_lock == 0:
                    state["action_state"] = "AFTERMATH_CRISIS_1"
                    crisis_turns = 0
            else:
                state["action_state"] = "CRISIS_1"
                crisis_turns = 1
                if crisis_lock <= 0:
                    crisis_lock = MIN_CRISIS_TURNS
                crisis_lock = max(crisis_lock - 1, 0)
            sexual_turns = 0
            aftermath_turns = 0
            sexual_lock = 0
        elif prev_action_state in AFTERMATH_CRISIS_STATES or base_action_state in AFTERMATH_CRISIS_STATES:
            state["action_state"] = "EVENT" if int(state["signals"].get("pressure", 0)) <= 1 else "CONFLICT"
            crisis_turns = 0
            crisis_lock = 0
            sexual_turns = 0
            aftermath_turns = 0
            sexual_lock = 0
        else:
            if base_action_state in SEXUAL_STATES and not state.get("sexual_ready", False):
                base_action_state = "EVENT"
            if base_action_state in CRISIS_STATES or base_action_state in AFTERMATH_CRISIS_STATES:
                base_action_state = "EVENT"
            genre = parse_genre(state["system_lore"])
            diversified = _pick_action_state(
                current=base_action_state,
                history=action_hist,
                genre=genre,
                sexual_ready=bool(state.get("sexual_ready", False)),
            )
            state["action_state"] = diversified
            sexual_turns = 0
            aftermath_turns = 0
            sexual_lock = 0
            crisis_turns = 0
            if state.get("sexual_ready", False) and state["action_state"] not in SEXUAL_STATES:
                sexual_turns = 1
                sexual_lock = MIN_SEXUAL_TURNS
                state["action_state"] = "SEXUAL_1"

        if state.get("user_idle_streak", 0) >= 2 and state["action_state"] == "IDLE":
            state["action_state"] = "EVENT"
        # 정체가 2턴 이상 누적되면 국면을 강제로 전진시켜 반복 루프를 끊는다.
        if (
            state.get("stall_count", 0) >= 1
            and state.get("action_state") not in SEXUAL_STATES
            and state.get("action_state") not in AFTERMATH_SEX_STATES
            and state.get("action_state") not in CRISIS_STATES
            and state.get("action_state") not in AFTERMATH_CRISIS_STATES
        ):
            cur = state.get("action_state", "IDLE")
            pressure = int(state["signals"].get("pressure", 0))
            if cur in ("IDLE", "AFTERMATH"):
                state["action_state"] = "CONFLICT" if state.get("stall_count", 0) >= 2 else "EVENT"
            elif cur == "EVENT":
                state["action_state"] = "RESOLUTION" if state.get("stall_count", 0) >= 2 else ("CONFLICT" if pressure >= 1 else "RESOLUTION")
            elif cur == "CONFLICT":
                state["action_state"] = "RESOLUTION"
            elif cur == "RESOLUTION" and state.get("stall_count", 0) >= 2:
                state["action_state"] = "AFTERMATH"
        if state.get("action_state") in SEXUAL_STATES and not state.get("sexual_ready", False):
            state["action_state"] = "EVENT"

        state["sexual_turns"] = sexual_turns
        state["aftermath_turns"] = aftermath_turns
        state["sexual_lock"] = sexual_lock
        state["crisis_turns"] = crisis_turns
        state["crisis_lock"] = crisis_lock

        action_hist.append(state["action_state"])
        state["action_state_hist"] = action_hist[-6:]
        log_step(
            "FSM_STEP",
            f"post action_state={state.get('action_state')} "
            f"sexual_turns={sexual_turns} aftermath_turns={aftermath_turns} "
            f"sexual_lock={sexual_lock} crisis_lock={crisis_lock}",
        )
        log_step("FSM_STEP", "done")
        return state

    def next_node(state: VNState) -> VNState:
        """
        턴 종료 후 다음 턴 준비

        turn_index를 증가시키고,
        텍스트 버퍼/재시도 카운터/일시 오류 상태를 다음 턴 기준으로 초기화한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: 다음 턴 실행 준비가 끝난 상태.
        """
        log_step("NEXT", "start")
        state["turn_index"] += 1
        # 직전 턴 대비 정체 여부를 LLM으로 판정해 stall_count를 누적한다.
        user_dia = _extract_dialogue_line(state.get("user_text", ""))
        asst_dia = _extract_dialogue_line(state.get("assistant_text", ""))
        user_intent = _intent_key(user_dia)
        asst_intent = _intent_key(asst_dia)
        user_sig = _content_signature(user_dia)
        asst_sig = _content_signature(asst_dia)
        llm_stalled = llm_stall_detect_local(
            model=model,
            tokenizer=tokenizer,
            system_lore=state.get("system_lore", ""),
            action_state=state.get("action_state", ""),
            prev_user_dialogue=state.get("last_user_dialogue", ""),
            prev_asst_dialogue=state.get("last_asst_dialogue", ""),
            user_dialogue=user_dia,
            asst_dialogue=asst_dia,
        )
        log_step(
            "NEXT",
            "stall_detect="
            f"llm:{'STALL' if llm_stalled else 'ADVANCE'} "
            "source:llm_only "
            f"prev_stall_count:{state.get('stall_count', 0)}",
        )
        if llm_stalled:
            state["stall_count"] = int(state.get("stall_count", 0)) + 1
        else:
            state["stall_count"] = 0
        state["user_idle_streak"] = state["stall_count"]
        state["last_user_intent"] = user_intent
        state["last_asst_intent"] = asst_intent
        state["last_user_signature"] = user_sig
        state["last_asst_signature"] = asst_sig
        state["last_user_dialogue"] = user_dia
        state["last_asst_dialogue"] = asst_dia
        state["user_text"] = ""
        state["assistant_text"] = ""
        state["narration_text"] = ""
        state["dialogue_text"] = ""
        state["sexual_request"] = False
        state["rewrite_mode"] = ""

        state["retry_user"] = 0
        state["retry_assistant"] = 0
        state["retry_quality"] = 0
        state["retry_sexual"] = 0
        state["retry_eval"] = 0
        state["retry_crisis"] = 0
        state["error"] = ""
        log_step("NEXT", f"done turn_index={state['turn_index']}")
        return state

    def done_node(state: VNState) -> VNState:
        """
        그래프 종료 정리

        마지막에 user만 남아 불완전한 턴이 된 경우 해당 메시지를 제거해
        출력 데이터를 user/assistant 쌍 기준으로 정리한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            VNState: 종료 정리 후 상태.
        """
        msgs = state.get("messages", [])
        if msgs and msgs[-1].get("role") == "user":
            msgs.pop()
        state["messages"] = msgs
        log_step("DONE", "cleanup_done")
        return state


    def after_gen_user(state: VNState) -> str:
        """
        user 생성 결과 분기

        user 생성 오류가 있으면 재시도 또는 종료를 선택하고,
        성공 시 detect 노드로 진행한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["error"]:
            if state["retry_user"] < 30:
                return "GEN_USER"
            return "DONE"
        return "DETECT"

    def after_detect(state: VNState) -> str:
        """
        detect 이후 국면 분기

        관계 FSM/액션 상태에 따라 crisis, sexual, normal assistant 노드 중 하나를 선택한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        action_state = state.get("action_state", "")
        if action_state.startswith("SEXUAL_") or action_state in AFTERMATH_SEX_STATES:
            return "GEN_SEXUAL"
        if (
            action_state in CRISIS_STATES
            or action_state in AFTERMATH_CRISIS_STATES
            or fsm_rel.get_state() == "CRISIS"
            or int(state.get("crisis_lock", 0)) > 0
        ):
            return "GEN_CRISIS"
        return "GEN_ASSISTANT"

    def after_gen_assistant(state: VNState) -> str:
        """
        assistant 생성 결과 분기

        오류 유형에 따라 rewrite 또는 재생성을 선택하고,
        성공 시 품질 평가 노드로 이동한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["error"]:
            if state["error"] in (
                "asst_meta",
                "asst_role_conflict",
                "asst_requires_two_lines",
                "asst_dia_requires_single_quote",
                "asst_unbalanced_quotes",
                "asst_narr_formal",
                "asst_narr_style_mismatch_llm",
                "asst_narr_has_quote",
                "asst_dia_silent",
                "asst_dialogue_narration",
                "asst_repeat_turn",
                "asst_player_third_person",
            ):
                if state.get("retry_rewrite", 0) < REWRITE_MAX_ATTEMPTS:
                    return "GEN_ASSISTANT_REWRITE"
            if state["retry_assistant"] < 30:
                return "GEN_ASSISTANT"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_assistant_rewrite(state: VNState) -> str:
        """
        assistant rewrite 결과 분기

        rewrite 실패 시 제한 횟수 내 재시도하고,
        성공하면 품질 평가 단계로 진행한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        crisis_errors = (
            "crisis_empty",
            "crisis_forbidden_chars",
            "crisis_colon_char",
            "crisis_unbalanced_quotes",
            "crisis_requires_two_lines",
            "crisis_no_narr_line",
            "crisis_requires_single_quote",
            "crisis_player_third_person",
        )
        if state["error"]:
            if state.get("rewrite_mode") == "crisis" or state["error"] in crisis_errors:
                if state.get("retry_rewrite", 0) < REWRITE_MAX_ATTEMPTS:
                    state["rewrite_mode"] = "crisis"
                    return "GEN_ASSISTANT_REWRITE"
                if state["retry_crisis"] < 12:
                    return "GEN_CRISIS"
                return "DONE"
            if state["error"] in (
                "asst_meta",
                "asst_role_conflict",
                "asst_requires_two_lines",
                "asst_dia_requires_single_quote",
                "asst_unbalanced_quotes",
                "asst_narr_formal",
                "asst_narr_style_mismatch_llm",
                "asst_narr_has_quote",
                "asst_dia_silent",
                "asst_dialogue_narration",
                "asst_player_third_person",
            ):
                if state.get("retry_rewrite", 0) < REWRITE_MAX_ATTEMPTS:
                    return "GEN_ASSISTANT_REWRITE"
            if state["retry_assistant"] < 30:
                return "GEN_ASSISTANT"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_sexual(state: VNState) -> str:
        """
        sexual 생성 결과 분기

        sexual 출력 실패 유형에 따라 rewrite 또는 재생성을 선택하고,
        성공 시 품질 평가 노드로 이동한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["error"]:
            if state["error"] in (
                "asst_role_conflict",
                "asst_requires_two_lines",
                "asst_dia_requires_single_quote",
                "asst_unbalanced_quotes",
                "asst_narr_formal",
                "asst_narr_style_mismatch_llm",
                "asst_narr_has_quote",
                "asst_dia_silent",
                "asst_dialogue_narration",
                "asst_player_third_person",
            ):
                if state.get("retry_rewrite", 0) < REWRITE_MAX_ATTEMPTS:
                    # sexual 노드 실패는 항상 sexual 전용 rewrite로 보정한다.
                    state["rewrite_mode"] = "sexual"
                    return "GEN_ASSISTANT_REWRITE"
            if state["retry_sexual"] < 30:
                return "GEN_SEXUAL"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_gen_crisis(state: VNState) -> str:
        """
        crisis 생성 결과 분기

        crisis 출력 오류는 한도 내 재시도하고,
        성공하면 품질 평가 단계로 진행한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["error"]:
            if state["error"] in (
                "crisis_empty",
                "crisis_forbidden_chars",
                "crisis_colon_char",
                "crisis_unbalanced_quotes",
                "crisis_requires_two_lines",
                "crisis_no_narr_line",
                "crisis_requires_single_quote",
                "crisis_player_third_person",
            ):
                if state.get("retry_rewrite", 0) < REWRITE_MAX_ATTEMPTS:
                    state["rewrite_mode"] = "crisis"
                    return "GEN_ASSISTANT_REWRITE"
            if state["retry_crisis"] < 12:
                return "GEN_CRISIS"
            return "DONE"
        return "EVAL_DIALOGUE_QUALITY"

    def after_eval_json(state: VNState) -> str:
        """
        JSON 평가 결과 분기

        JSON 파싱/평가 실패는 재시도 한도까지 반복하고,
        성공하면 FSM 전이 노드로 진행한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["error"]:
            if state["retry_eval"] < 40:
                return "EVAL_JSON"
            return "DONE"
        return "FSM_STEP"

    def after_next(state: VNState) -> str:
        """
        턴 진행 여부 분기

        목표 턴 수에 도달하면 종료하고,
        그렇지 않으면 sexual 준비 여부에 따라 다음 user 노드를 선택한다.

        Args:
            state: 현재 VN 실행 상태.

        Returns:
            str: 다음 노드 키.
        """
        if state["turn_index"] >= state["turns_target"]:
            return "DONE"
        action_state = state.get("action_state", "")
        if action_state.startswith("SEXUAL_") or action_state in AFTERMATH_SEX_STATES:
            return "GEN_USER_SEXUAL"
        return "GEN_USER"


    g = StateGraph(VNState)
    g.add_node("INIT", init_node)
    g.add_node("GEN_USER", gen_user_node)
    g.add_node("GEN_USER_SEXUAL", gen_user_sexual_node)
    g.add_node("DETECT", detect_node)
    g.add_node("GEN_ASSISTANT", gen_assistant_node)  # 통합 생성 노드
    g.add_node("GEN_ASSISTANT_REWRITE", gen_assistant_rewrite_node)
    g.add_node("EVAL_DIALOGUE_QUALITY", eval_dialogue_quality_node)
    g.add_node("GEN_CRISIS", gen_crisis_node)
    g.add_node("GEN_SEXUAL", gen_sexual_node)
    g.add_node("EVAL_INTERNAL", eval_internal_node)
    g.add_node("EVAL_JSON", eval_json_node)
    g.add_node("FSM_STEP", fsm_step_node)
    g.add_node("NEXT", next_node)
    g.add_node("DONE", done_node)

    g.set_entry_point("INIT")

    g.add_edge("INIT", "GEN_USER")

    g.add_conditional_edges("GEN_USER", after_gen_user, {
        "GEN_USER": "GEN_USER",
        "DETECT": "DETECT",
        "DONE": "DONE",
    })
    g.add_conditional_edges("GEN_USER_SEXUAL", after_gen_user, {
        "GEN_USER": "GEN_USER_SEXUAL",
        "DETECT": "DETECT",
        "DONE": "DONE",
    })

    g.add_conditional_edges("DETECT", after_detect, {
        "GEN_CRISIS": "GEN_CRISIS",
        "GEN_ASSISTANT": "GEN_ASSISTANT",
        "GEN_SEXUAL": "GEN_SEXUAL",
    })

    g.add_conditional_edges("GEN_ASSISTANT", after_gen_assistant, {
        "GEN_ASSISTANT": "GEN_ASSISTANT",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_ASSISTANT_REWRITE", after_gen_assistant_rewrite, {
        "GEN_ASSISTANT": "GEN_ASSISTANT",
        "GEN_CRISIS": "GEN_CRISIS",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_SEXUAL", after_gen_sexual, {
        "GEN_SEXUAL": "GEN_SEXUAL",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_conditional_edges("GEN_CRISIS", after_gen_crisis, {
        "GEN_CRISIS": "GEN_CRISIS",
        "GEN_ASSISTANT_REWRITE": "GEN_ASSISTANT_REWRITE",
        "EVAL_DIALOGUE_QUALITY": "EVAL_DIALOGUE_QUALITY",
        "DONE": "DONE",
    })

    g.add_edge("EVAL_DIALOGUE_QUALITY", "EVAL_INTERNAL")
    g.add_edge("EVAL_INTERNAL", "EVAL_JSON")
    g.add_conditional_edges("EVAL_JSON", after_eval_json, {
        "EVAL_JSON": "EVAL_JSON",
        "FSM_STEP": "FSM_STEP",
        "DONE": "DONE",
    })
    g.add_edge("FSM_STEP", "NEXT")
    g.add_conditional_edges("NEXT", after_next, {
        "GEN_USER": "GEN_USER",
        "GEN_USER_SEXUAL": "GEN_USER_SEXUAL",
        "DONE": "DONE",
    })

    g.add_edge("DONE", END)

    return g.compile()


def run_scenario(
    system_lore: str,
    model,
    tokenizer,
    turns: int,
    fsm_path: str,
    action_fsm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """입력 설정으로 `run_scenario` 실행 단위를 수행하고 결과를 반환한다."""
    fsm_rel = QwenFSMEngine(fsm_path, system_lore)
    act_path = action_fsm_path or "data/version_2/v6_qwen/action_fsm.yaml"
    fsm_act = QwenFSMEngine(act_path, system_lore)

    # MOD: device 자동 선택(모델 device_map="auto" 고려)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_memory = EmbeddingMemory("data/embedding/BGE-m3-ko", device=device)

    graph = build_graph(
        model=model,
        tokenizer=tokenizer,
        fsm_rel=fsm_rel,
        fsm_act=fsm_act,
        embed_memory=embed_memory,
    )

    init_state: VNState = {
        "system_lore": system_lore,
        "protagonist": "",
        "player_name": "",
        "relation_status": "",
        "protagonist_speech_formal": None,
        "sexual_condition_rel": "",
        "relation_intent": "",        "action_state": "",
        "last_user_question": False,
        "last_asst_question": False,
        "last_asst_sexual": False,
        "action_state_hist": [],
        "crisis_lock": 0,
        "crisis_turns": 0,
        "sexual_turns": 0,
        "sexual_lock": 0,
        "aftermath_turns": 0,
        "user_idle_streak": 0,
        "stall_count": 0,
        "last_user_intent": "",
        "last_asst_intent": "",
        "last_user_signature": "",
        "last_asst_signature": "",
        "last_user_dialogue": "",
        "last_asst_dialogue": "",

        "turns_target": int(turns),
        "turn_index": 0,

        "messages": [],

        "user_text": "",
        "assistant_text": "",
        "raw_assistant": "",
        "narration_text": "",
        "dialogue_text": "",

        "sexual_request": False,
        "allow_sexual": False,
        "sexual_ready": False,

        "signals": {
            "mental_instability": 0,
            "intimacy": 0,
            "threat": 0,
            "pressure": 0,
            "probe": 0,
            "resolve": 0,
            "event": 0,
        },

        "retry_user": 0,
        "retry_assistant": 0,
        "retry_rewrite": 0,
        "retry_quality": 0,
        "retry_sexual": 0,
        "retry_eval": 0,
        "retry_crisis": 0,
        "eval_internal_text": "",
        "rewrite_mode": "",

        "error": "",
    }

    def _has_min_turn(msgs: List[Dict[str, str]]) -> bool:
        """내부 헬퍼로 `_has_min_turn` 계산 절차를 수행한다."""
        has_user = False
        has_asst = False
        for m in msgs:
            if m.get("role") == "user":
                has_user = True
            elif m.get("role") == "assistant":
                has_asst = True
            if has_user and has_asst:
                return True
        return False

    out = None
    for attempt in range(4):
        out = graph.invoke(init_state)
        if _has_min_turn(out.get("messages", [])):
            break
        print(f"[WARN] no_turn_generated retry={attempt+1}/4", flush=True)

    return {"messages": out["messages"] if out else []}


def main():
    """CLI 인자를 파싱하고 파일 단위 멀티턴 생성 배치를 실행한다.

    Args:
        없음

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scenario_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--fsm_path", required=True)
    parser.add_argument("--action_fsm_path", default="data/version_2/v6_qwen/action_fsm.yaml")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=bnb,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    model.eval()

    def count_lines(path: str) -> int:
        """
        비어있지 않은 라인 수 계산

        재시작(resume) 시 scenario/multiturn 파일의 진행 개수를 비교하기 위해
        파일의 유효 라인 수를 센다.

        Args:
            path: 집계할 파일 경로.

        Returns:
            int: 공백 라인을 제외한 라인 개수.
        """
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as rf:
            return sum(1 for _ in rf if _.strip())

    out_done = count_lines(args.out_path)
    if out_done > 0:
        print(f"[MULTITURN] resume_from={out_done}", flush=True)

    total_scen = count_lines(args.scenario_path)
    if out_done >= total_scen:
        print("[MULTITURN] nothing to do (out >= scenarios)", flush=True)
        return

    with open(args.scenario_path, "r", encoding="utf-8") as f, open(
        args.out_path, "a", encoding="utf-8"
    ) as out:
        idx = 0
        for line in f:
            if not line.strip():
                continue
            if idx < out_done:
                idx += 1
                continue
            system_lore = json.loads(line)["messages"][0]["content"]
            data = run_scenario(
                system_lore=system_lore,
                model=model,
                tokenizer=tokenizer,
                turns=args.turns,
                fsm_path=args.fsm_path,
                action_fsm_path=args.action_fsm_path,
            )
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            out.flush()
            idx += 1

    print("DONE")
