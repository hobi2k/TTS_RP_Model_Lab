"""네이티브 function calling 기반 LangGraph 턴 파이프라인."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class NativeFCTurnState(TypedDict, total=False):
    """네이티브 function calling 그래프 상태."""

    text_ko: str
    image_path: str | None
    style_index: int
    style_weight: float
    speaker_name: str

    messages: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None
    emotion: dict[str, int]


def build_native_fc_turn_graph(
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
    """네이티브 function calling 기반 메인 턴 그래프를 생성한다."""

    def available_tools(image_path: str | None) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "memory_lookup",
                "description": "현재 user 입력과 관련된 장기기억과 최근 대화 스냅샷을 조회한다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "조회 기준이 되는 현재 user 입력 또는 핵심 질문",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "recent_history_lookup",
                "description": "직전 user/assistant 대화 문면을 다시 확인한다.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]
        if image_path:
            tools.append(
                {
                    "name": "image_analyze",
                    "description": "현재 턴 이미지에서 보이는 장면과 글자(OCR)를 확인한다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "focus": {
                                "type": "string",
                                "description": "이미지에서 집중해서 볼 항목",
                            }
                        },
                        "required": [],
                    },
                }
            )
        return tools

    def node_prepare_messages(state: NativeFCTurnState) -> NativeFCTurnState:
        text_ko = (state.get("text_ko") or "").strip()
        messages: list[dict[str, Any]] = []
        messages.extend(prompt_compiler.compile())
        messages.append(
            {
                "role": "system",
                "content": (
                    "도구가 필요할 때만 function call을 사용하라. "
                    "도구 결과를 받으면 그 근거를 반영해 사야의 RP 형식으로 응답하라. "
                    "최종 출력은 반드시 서술 1블록과 대사 1블록으로 작성하라."
                ),
            }
        )
        messages.extend(history)
        messages.append({"role": "user", "content": text_ko})
        return {"text_ko": text_ko, "messages": messages}

    def node_native_fc_generate(state: NativeFCTurnState) -> NativeFCTurnState:
        text_ko = state.get("text_ko", "")
        image_path = state.get("image_path")

        def tool_executor(name: str, args: dict[str, Any]) -> str:
            if name == "memory_lookup":
                query = str(args.get("query") or text_ko).strip()
                result = memory_chain.build_memory_system_message(current_user_text=query)
                if isinstance(result, dict):
                    content = str(result.get("content", "")).strip()
                    return content or "관련 메모리 없음."
                return "관련 메모리 없음."
            if name == "recent_history_lookup":
                lines = []
                for item in history:
                    role = "user" if item.get("role") == "user" else "assistant"
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    lines.append(f"- {role}: {content[:220]}")
                return "\n".join(lines) if lines else "최근 history 없음."
            if name == "image_analyze":
                if image_tool is None or not image_path:
                    return "현재 분석 가능한 이미지 없음."
                focus = str(args.get("focus") or "").strip()
                prompt = "현재 이미지에서 보이는 장면과 글자를 짧게 정리하라."
                if focus:
                    prompt += f" 특히 다음 항목에 집중하라: {focus}"
                return image_tool.generate_from_messages(
                    [
                        {
                            "role": "system",
                            "content": (
                                "너는 이미지 분석 도구다. 현재 이미지에서 직접 확인되는 사실만 짧게 정리하라. "
                                "읽을 수 있는 글자가 있으면 OCR 결과를 적고, 없으면 없다고 적어라."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    image_path=image_path,
                ).strip() or "이미지 분석 결과 없음."
            return f"지원하지 않는 도구: {name}"

        rp_text, tool_trace, _ = llm_engine.run_tool_loop(
            messages=state["messages"],
            tools=available_tools(image_path),
            tool_executor=tool_executor,
        )
        rp = rp_parser.parse(rp_text)
        dialogue_ko = normalize_dialogue(rp.dialogue_en or "")
        return {
            "tool_trace": tool_trace,
            "rp_text": rp_text,
            "narration": rp.narration,
            "dialogue_ko": dialogue_ko,
        }

    def node_translate(state: NativeFCTurnState) -> NativeFCTurnState:
        dialogue_ko = state.get("dialogue_ko", "")
        if not dialogue_ko:
            return {"dialogue_ja": ""}
        return {"dialogue_ja": translator.translate(dialogue_ko)}

    def node_emotion(state: NativeFCTurnState) -> NativeFCTurnState:
        text_ko = state.get("text_ko", "")
        narration = state.get("narration", "")
        dialogue_ko = state.get("dialogue_ko", "")
        emotion = llm_engine.infer_emotion_json(narration, dialogue_ko)
        if emotion is None:
            emotion = infer_emotion_fallback(text_ko, narration, dialogue_ko)
        return {"emotion": emotion}

    def node_tts(state: NativeFCTurnState) -> NativeFCTurnState:
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

    def node_commit(state: NativeFCTurnState) -> NativeFCTurnState:
        text_ko = state.get("text_ko", "")
        rp_text = state.get("rp_text", "")
        narration = (state.get("narration") or "").strip()
        dialogue_ko = state.get("dialogue_ko", "")
        history.append({"role": "user", "content": text_ko})
        history.append({"role": "assistant", "content": rp_text})
        mem_assistant_text = "\n".join(p for p in [narration, dialogue_ko] if p).strip() or rp_text
        memory_chain.update(user_text=text_ko, assistant_text=mem_assistant_text)
        return {}

    graph = StateGraph(NativeFCTurnState)
    graph.add_node("prepare_messages", node_prepare_messages)
    graph.add_node("native_fc_generate", node_native_fc_generate)
    graph.add_node("translate", node_translate)
    graph.add_node("emotion", node_emotion)
    graph.add_node("tts", node_tts)
    graph.add_node("commit", node_commit)

    graph.add_edge(START, "prepare_messages")
    graph.add_edge("prepare_messages", "native_fc_generate")
    graph.add_edge("native_fc_generate", "translate")
    graph.add_edge("translate", "emotion")
    graph.add_edge("emotion", "tts")
    graph.add_edge("tts", "commit")
    graph.add_edge("commit", END)
    return graph.compile()
