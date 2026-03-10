"""VLM 전용 LangGraph 턴 파이프라인."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class VLMTurnState(TypedDict, total=False):
    """VLM 턴 그래프 상태."""

    text_ko: str
    image_path: str | None
    style_index: int
    style_weight: float
    speaker_name: str

    messages: list[dict[str, Any]]
    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None
    emotion: dict[str, int]


def build_vlm_turn_graph(
    *,
    llm_engine: Any,
    rp_parser: Any,
    translator: Any,
    tts_synth: Callable[..., str],
    prompt_compiler: Any,
    history: deque[dict[str, Any]],
    normalize_dialogue: Callable[[str], str],
    infer_emotion_fallback: Callable[[str, str, str], dict[str, int]],
):
    """VLM 전용 턴 파이프라인 그래프를 생성/컴파일한다."""

    def node_prepare_prompt(state: VLMTurnState) -> VLMTurnState:
        text_ko = (state.get("text_ko") or "").strip()
        messages: list[dict[str, Any]] = []
        messages.extend(prompt_compiler.compile())
        messages.extend(history)
        messages.append({"role": "user", "content": text_ko})
        return {
            "text_ko": text_ko,
            "image_path": state.get("image_path"),
            "messages": messages,
        }

    def node_generate_parse(state: VLMTurnState) -> VLMTurnState:
        rp_text = llm_engine.generate_from_messages(
            state["messages"],
            image_path=state.get("image_path"),
        )
        rp = rp_parser.parse(rp_text)
        dialogue_ko = normalize_dialogue(rp.dialogue_en or "")
        return {
            "rp_text": rp_text,
            "narration": rp.narration,
            "dialogue_ko": dialogue_ko,
        }

    def node_translate(state: VLMTurnState) -> VLMTurnState:
        dialogue_ko = state.get("dialogue_ko", "")
        if not dialogue_ko:
            return {"dialogue_ja": ""}
        return {"dialogue_ja": translator.translate(dialogue_ko)}

    def node_emotion(state: VLMTurnState) -> VLMTurnState:
        narration = state.get("narration", "")
        dialogue_ko = state.get("dialogue_ko", "")
        emotion = llm_engine.infer_emotion_json(narration, dialogue_ko)
        if emotion is None:
            emotion = infer_emotion_fallback(state.get("text_ko", ""), narration, dialogue_ko)
        return {"emotion": emotion}

    def node_tts(state: VLMTurnState) -> VLMTurnState:
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

    def node_commit(state: VLMTurnState) -> VLMTurnState:
        history.append({"role": "user", "content": state.get("text_ko", "")})
        history.append({"role": "assistant", "content": state.get("rp_text", "")})
        return {}

    graph = StateGraph(VLMTurnState)
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
