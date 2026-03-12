"""Tool-use 기반 LangGraph 턴 파이프라인."""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class ToolCall(TypedDict, total=False):
    """계획된 도구 호출."""

    name: str
    args: dict[str, Any]


class ToolTrace(TypedDict, total=False):
    """실행된 도구 로그."""

    name: str
    args: dict[str, Any]
    result_preview: str


class ToolTurnState(TypedDict, total=False):
    """툴-유즈 턴 그래프 상태."""

    text_ko: str
    image_path: str | None
    style_index: int
    style_weight: float
    speaker_name: str

    messages: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    tool_trace: list[ToolTrace]
    tool_context: str

    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None
    emotion: dict[str, int]


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """planner 출력에서 첫 JSON 객체를 꺼낸다."""
    if not raw_text.strip():
        return {}
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _planner_messages(text_ko: str, image_path: str | None, has_history: bool) -> list[dict[str, str]]:
    """도구 계획용 planner 메시지를 만든다."""
    image_available = "yes" if image_path else "no"
    history_available = "yes" if has_history else "no"
    return [
        {
            "role": "system",
            "content": """
너는 RP 대화용 도구 계획기다.
허용 도구는 아래 셋뿐이다.
- memory_lookup: 현재 입력과 관련된 장기기억/최근 대화 맥락이 필요할 때
- recent_history_lookup: 직전 턴들의 실제 문면을 다시 확인해야 할 때
- image_analyze: 현재 턴에 이미지가 있고, 현재 이미지의 시각 정보/OCR 확인이 필요할 때

규칙:
- 필요 없는 도구는 호출하지 마라.
- 최대 2개까지만 호출하라.
- image_analyze는 현재 턴에 이미지가 있을 때만 호출하라.
- 출력은 JSON 객체 하나만 사용하라.
- 스키마:
{"tool_calls":[{"name":"memory_lookup","args":{"query":"..."}}]}
""".strip(),
        },
        {
            "role": "user",
            "content": (
                f"현재 user 입력: {text_ko}\n"
                f"현재 턴 이미지 존재: {image_available}\n"
                f"최근 history 존재: {history_available}\n"
                "필요한 도구 호출만 JSON으로 출력하라."
            ),
        },
    ]


def _planner_gen_config(llm_engine: Any) -> Any | None:
    """planner용 저온 생성 설정을 만든다."""
    default_gen = getattr(llm_engine, "default_gen", None)
    if default_gen is None:
        return None
    gen_type = type(default_gen)
    try:
        return gen_type(
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            top_k=20,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            do_sample=False,
            use_cache=True,
        )
    except Exception:
        return None


def build_tool_use_turn_graph(
    *,
    llm_engine: Any,
    image_tool: Any | None,
    rp_parser: Any,
    translator: Any,
    tts_synth: Callable[..., str],
    prompt_compiler: Any,
    memory_chain: Any,
    history: deque[dict[str, Any]],
    normalize_dialogue: Callable[[str], str],
    infer_emotion_fallback: Callable[[str, str, str], dict[str, int]],
):
    """tool-use 기반 턴 파이프라인 그래프를 생성한다."""

    def node_plan_tools(state: ToolTurnState) -> ToolTurnState:
        text_ko = (state.get("text_ko") or "").strip()
        image_path = state.get("image_path")
        planner_raw = llm_engine.generate_from_messages(
            _planner_messages(text_ko, image_path, bool(history)),
            gen_config=_planner_gen_config(llm_engine),
        )
        parsed = _extract_json_object(planner_raw)
        raw_calls = parsed.get("tool_calls", []) if isinstance(parsed, dict) else []
        tool_calls: list[ToolCall] = []
        for item in raw_calls[:2]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            args = item.get("args", {})
            if name not in {"memory_lookup", "recent_history_lookup", "image_analyze"}:
                continue
            if name == "image_analyze" and not image_path:
                continue
            if not isinstance(args, dict):
                args = {}
            tool_calls.append({"name": name, "args": args})
        return {"text_ko": text_ko, "tool_calls": tool_calls, "tool_trace": []}

    def route_after_plan(state: ToolTurnState) -> str:
        return "execute_tools" if state.get("tool_calls") else "compose"

    def node_execute_tools(state: ToolTurnState) -> ToolTurnState:
        text_ko = state.get("text_ko", "")
        image_path = state.get("image_path")
        tool_calls = state.get("tool_calls", [])
        traces: list[ToolTrace] = []
        context_chunks: list[str] = []

        for call in tool_calls:
            name = call.get("name", "")
            args = dict(call.get("args", {}))
            result_text = ""

            if name == "memory_lookup":
                query = str(args.get("query") or text_ko).strip()
                result = memory_chain.build_memory_system_message(current_user_text=query)
                result_text = str(result.get("content", "")).strip() if isinstance(result, dict) else ""
                if not result_text:
                    result_text = "관련 메모리 없음."
            elif name == "recent_history_lookup":
                history_lines = []
                for item in history:
                    role = "user" if item.get("role") == "user" else "assistant"
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    history_lines.append(f"- {role}: {content[:220]}")
                result_text = "\n".join(history_lines) if history_lines else "최근 history 없음."
            elif name == "image_analyze":
                if image_tool is None:
                    result_text = "이미지 분석 도구를 사용할 수 없음."
                elif not image_path:
                    result_text = "현재 턴에 제공된 이미지 없음."
                else:
                    analysis_messages = [
                        {
                            "role": "system",
                            "content": """
너는 이미지 분석 도구다.
현재 이미지에서 직접 확인되는 사실만 짧게 정리하라.
보이는 글자가 있으면 OCR 결과를 적고, 없으면 '보이는 글자 없음'이라고 적어라.
추측하지 말고 현재 이미지 근거만 사용하라.
""".strip(),
                        },
                        {
                            "role": "user",
                            "content": "현재 이미지에서 확인되는 글자, 장면, 인물 상태를 짧게 요약해라.",
                        },
                    ]
                    result_text = image_tool.generate_from_messages(
                        analysis_messages,
                        image_path=image_path,
                    ).strip()
                    if not result_text:
                        result_text = "이미지 분석 결과 없음."

            traces.append(
                {
                    "name": name,
                    "args": args,
                    "result_preview": result_text.replace("\n", " ")[:240],
                }
            )
            context_chunks.append(f"[tool:{name}]\n{result_text}")

        return {
            "tool_trace": traces,
            "tool_context": "\n\n".join(context_chunks).strip(),
        }

    def node_compose(state: ToolTurnState) -> ToolTurnState:
        text_ko = state.get("text_ko", "")
        messages: list[dict[str, Any]] = []
        messages.extend(prompt_compiler.compile())

        tool_context = (state.get("tool_context") or "").strip()
        if tool_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "도구 실행 결과:\n"
                        f"{tool_context}\n\n"
                        "규칙: 도구 결과는 참고용이다. 현재 user의 마지막 발화에 직접 반응하라. "
                        "최종 출력은 반드시 사야의 서술 1블록과 대사 1블록으로 작성하라."
                    ),
                }
            )

        messages.extend(history)
        messages.append({"role": "user", "content": text_ko})
        rp_text = llm_engine.generate_from_messages(messages)
        rp = rp_parser.parse(rp_text)
        dialogue_ko = normalize_dialogue(rp.dialogue_en or "")
        return {
            "messages": messages,
            "rp_text": rp_text,
            "narration": rp.narration,
            "dialogue_ko": dialogue_ko,
        }

    def node_translate(state: ToolTurnState) -> ToolTurnState:
        dialogue_ko = state.get("dialogue_ko", "")
        if not dialogue_ko:
            return {"dialogue_ja": ""}
        return {"dialogue_ja": translator.translate(dialogue_ko)}

    def node_emotion(state: ToolTurnState) -> ToolTurnState:
        text_ko = state.get("text_ko", "")
        narration = state.get("narration", "")
        dialogue_ko = state.get("dialogue_ko", "")
        emotion = llm_engine.infer_emotion_json(narration, dialogue_ko)
        if emotion is None:
            emotion = infer_emotion_fallback(text_ko, narration, dialogue_ko)
        return {"emotion": emotion}

    def node_tts(state: ToolTurnState) -> ToolTurnState:
        dialogue_ja = state.get("dialogue_ja", "")
        if not dialogue_ja:
            return {"wav_path": None}
        wav_path = tts_synth(
            dialogue_ja,
            style_index=int(state.get("style_index", 0)),
            style_weight=float(state.get("style_weight", 1.0)),
            speaker_name=(state.get("speaker_name") or "saya"),
        )
        return {"wav_path": wav_path}

    def node_commit(state: ToolTurnState) -> ToolTurnState:
        text_ko = state.get("text_ko", "")
        rp_text = state.get("rp_text", "")
        narration = (state.get("narration") or "").strip()
        dialogue_ko = state.get("dialogue_ko", "")

        history.append({"role": "user", "content": text_ko})
        history.append({"role": "assistant", "content": rp_text})

        mem_assistant_text = "\n".join(p for p in [narration, dialogue_ko] if p).strip() or rp_text
        memory_chain.update(user_text=text_ko, assistant_text=mem_assistant_text)
        return {}

    graph = StateGraph(ToolTurnState)
    graph.add_node("plan_tools", node_plan_tools)
    graph.add_node("execute_tools", node_execute_tools)
    graph.add_node("compose", node_compose)
    graph.add_node("translate", node_translate)
    graph.add_node("emotion", node_emotion)
    graph.add_node("tts", node_tts)
    graph.add_node("commit", node_commit)

    graph.add_edge(START, "plan_tools")
    graph.add_conditional_edges(
        "plan_tools",
        route_after_plan,
        {
            "execute_tools": "execute_tools",
            "compose": "compose",
        },
    )
    graph.add_edge("execute_tools", "compose")
    graph.add_edge("compose", "translate")
    graph.add_edge("translate", "emotion")
    graph.add_edge("emotion", "tts")
    graph.add_edge("tts", "commit")
    graph.add_edge("commit", END)

    return graph.compile()
