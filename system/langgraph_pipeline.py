"""LangGraph 기반 턴 파이프라인.

이 모듈은 기존 순차 코드(LLM -> parse -> translate -> emotion -> TTS + memory)를
LangGraph StateGraph로 구성해 FastAPI/CLI에서 공통으로 재사용할 수 있게 한다.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class TurnState(TypedDict, total=False):
    """턴 그래프 상태."""

    text_ko: str
    style_index: int
    style_weight: float
    speaker_name: str

    messages: list[dict[str, Any]]
    prompt: str
    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None
    emotion: dict[str, int]


def build_turn_graph(
    *,
    llm_engine: Any,
    rp_parser: Any,
    translator: Any,
    tts_synth: Callable[..., str],
    prompt_compiler: Any,
    memory_chain: Any,
    history: deque[dict[str, Any]],
    normalize_dialogue: Callable[[str], str],
    infer_emotion_fallback: Callable[[str, str, str], dict[str, int]],
):
    """턴 파이프라인 그래프를 생성/컴파일한다.

    Notes:
        - history/memory는 외부(RuntimeServices) 소유 객체를 그대로 참조한다.
        - 동시성 제어는 상위 호출자에서 수행해야 한다.
    """

    def node_prepare_prompt(state: TurnState) -> TurnState:
        text_ko = (state.get("text_ko") or "").strip()
        system_msgs = prompt_compiler.compile()
        messages: list[dict[str, Any]] = []
        messages.extend(system_msgs)

        memory_msg = memory_chain.build_memory_system_message(current_user_text=text_ko)
        if memory_msg is not None:
            messages.append(memory_msg)

        messages.extend(history)
        messages.append({"role": "user", "content": text_ko})
        return {"text_ko": text_ko, "messages": messages}

    def node_generate_parse(state: TurnState) -> TurnState:
        prompt = llm_engine.build_prompt(state["messages"])
        rp_text = llm_engine.generate(prompt)
        rp = rp_parser.parse(rp_text)
        dialogue_ko = normalize_dialogue(rp.dialogue_en or "")
        return {
            "prompt": prompt,
            "rp_text": rp_text,
            "narration": rp.narration,
            "dialogue_ko": dialogue_ko,
        }

    def node_translate(state: TurnState) -> TurnState:
        dialogue_ko = state.get("dialogue_ko", "")
        if not dialogue_ko:
            return {"dialogue_ja": ""}
        return {"dialogue_ja": translator.translate(dialogue_ko)}

    def node_emotion(state: TurnState) -> TurnState:
        text_ko = state.get("text_ko", "")
        narration = state.get("narration", "")
        dialogue_ko = state.get("dialogue_ko", "")
        emotion = llm_engine.infer_emotion_json(narration, dialogue_ko)
        if emotion is None:
            emotion = infer_emotion_fallback(text_ko, narration, dialogue_ko)
        return {"emotion": emotion}

    def node_tts(state: TurnState) -> TurnState:
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

    def node_commit(state: TurnState) -> TurnState:
        text_ko = state.get("text_ko", "")
        rp_text = state.get("rp_text", "")
        narration = (state.get("narration") or "").strip()
        dialogue_ko = state.get("dialogue_ko", "")

        history.append({"role": "user", "content": text_ko})
        history.append({"role": "assistant", "content": rp_text})

        mem_assistant_text = "\n".join(p for p in [narration, dialogue_ko] if p).strip() or rp_text
        memory_chain.update(user_text=text_ko, assistant_text=mem_assistant_text)
        return {}

    graph = StateGraph(TurnState)
    graph.add_node("prepare_prompt", node_prepare_prompt)
    graph.add_node("generate_parse", node_generate_parse)
    graph.add_node("translate", node_translate)
    graph.add_node("emotion", node_emotion)
    graph.add_node("tts", node_tts)
    graph.add_node("commit", node_commit)

    graph.add_edge(START, "prepare_prompt")
    graph.add_edge("prepare_prompt", "generate_parse")
    graph.add_edge("generate_parse", "translate")
    graph.add_edge("translate", "emotion")
    graph.add_edge("emotion", "tts")
    graph.add_edge("tts", "commit")
    graph.add_edge("commit", END)

    return graph.compile()

