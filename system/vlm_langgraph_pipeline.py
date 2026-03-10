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
    visual_facts: str
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
    memory_chain: Any,
    debug_memory: bool,
    history: deque[dict[str, Any]],
    normalize_dialogue: Callable[[str], str],
    infer_emotion_fallback: Callable[[str, str, str], dict[str, int]],
):
    """VLM 전용 턴 파이프라인 그래프를 생성/컴파일한다."""

    image_analysis_guidance = {
        "role": "system",
        "content": """
너의 일은 현재 이미지에서 보이는 사실만 추출하는 것이다.
이전 턴 이미지, 이전 OCR 결과, 장기 기억, 추측을 섞지 마라.
현재 이미지에서 직접 다시 확인되지 않은 글자, 사물, 배경, 장면 단서는 쓰지 마라.
이미지 안에 읽을 수 있는 글자가 보이면 가능한 한 정확히 읽어라.
글자가 선명하지 않아 확신이 낮으면 추측이라고 분명히 밝혀라.

아래 형식으로만 답하라.
VISIBLE_TEXT: ...
SCENE: ...
SUBJECT_STATE: ...
OBJECTS: ...

보이지 않거나 읽히지 않으면 none 이라고 써라.
""".strip(),
    }

    image_guidance = {
        "role": "system",
        "content": """
현재 장면의 1차 근거는 방금 추출된 visual facts다.
장기 기억과 최근 대화는 관계, 배경, 누적 단서, 말투를 보강하는 데만 사용하라.
visual facts와 기억이 충돌하면 visual facts를 우선하라.
이전 턴 이미지의 결과를 자동으로 이어서 쓰지 마라.

중요:
- user의 마지막 발화에 직접 반응해야 한다.
- 최종 출력은 반드시 사야의 RP 형식이어야 한다.
- 반드시 서술 1블록 + 대사 1블록만 출력하라.
- 서술은 3인칭 평어체로 쓰고, 대사는 큰따옴표로 감싸라.
- 사야의 대사만 작성하라. 카즈키의 대사나 행동을 대신 쓰지 마라.
- visual facts에 포함된 글자가 있으면 그 내용을 기반으로 대사를 작성하라.
""".strip(),
    }

    def node_prepare_prompt(state: VLMTurnState) -> VLMTurnState:
        text_ko = (state.get("text_ko") or "").strip()
        base_messages = list(prompt_compiler.compile())
        memory_msg = memory_chain.build_memory_system_message(current_user_text=text_ko)
        history_messages = list(history)
        user_message = {"role": "user", "content": text_ko}
        if memory_msg is not None:
            if debug_memory:
                preview = str(memory_msg.get("content", "")).replace("\n", " ")[:240]
                print(f"[vlm-memory] turn inject: {preview}")
            base_messages.append(memory_msg)
        messages = [*base_messages, *history_messages, user_message]
        return {
            "text_ko": text_ko,
            "image_path": state.get("image_path"),
            "messages": messages,
        }

    def _parse_generated_rp(rp_text: str) -> VLMTurnState:
        """생성된 RP 원문을 파싱/정규화해 공통 상태로 변환한다."""
        rp = rp_parser.parse(rp_text)
        dialogue_ko = normalize_dialogue(rp.dialogue_en or "")
        return {
            "rp_text": rp_text,
            "narration": rp.narration,
            "dialogue_ko": dialogue_ko,
        }

    def route_has_image(state: VLMTurnState) -> str:
        """이미지 입력 유무에 따라 생성 경로를 분기한다."""
        image_path = (state.get("image_path") or "").strip()
        return "extract_visual_facts" if image_path else "generate_text_only"

    def node_extract_visual_facts(state: VLMTurnState) -> VLMTurnState:
        visual_facts = llm_engine.generate_from_messages(
            [
                image_analysis_guidance,
                {"role": "user", "content": state.get("text_ko", "") or "현재 이미지를 분석해줘."},
            ],
            image_path=state.get("image_path"),
        ).strip()
        return {"visual_facts": visual_facts}

    def node_generate_text_only(state: VLMTurnState) -> VLMTurnState:
        rp_text = llm_engine.generate_from_messages(state["messages"])
        return _parse_generated_rp(rp_text)

    def node_generate_with_image(state: VLMTurnState) -> VLMTurnState:
        messages = [
            *state["messages"],
            image_guidance,
            {
                "role": "system",
                "content": f"현재 이미지에서 추출된 visual facts:\n{state.get('visual_facts', '').strip()}",
            },
        ]
        rp_text = llm_engine.generate_from_messages(
            messages,
        )
        return _parse_generated_rp(rp_text)

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
        text_ko = state.get("text_ko", "")
        rp_text = state.get("rp_text", "")
        narration = (state.get("narration") or "").strip()
        dialogue_ko = state.get("dialogue_ko", "")

        history.append({"role": "user", "content": text_ko})
        history.append({"role": "assistant", "content": rp_text})

        mem_assistant_text = "\n".join(p for p in [narration, dialogue_ko] if p).strip() or rp_text
        memory_chain.update(user_text=text_ko, assistant_text=mem_assistant_text)
        return {}

    graph = StateGraph(VLMTurnState)
    graph.add_node("prepare_prompt", node_prepare_prompt)
    graph.add_node("extract_visual_facts", node_extract_visual_facts)
    graph.add_node("generate_text_only", node_generate_text_only)
    graph.add_node("generate_with_image", node_generate_with_image)
    graph.add_node("translate", node_translate)
    graph.add_node("emotion", node_emotion)
    graph.add_node("tts", node_tts)
    graph.add_node("commit", node_commit)

    graph.add_edge(START, "prepare_prompt")
    graph.add_conditional_edges(
        "prepare_prompt",
        route_has_image,
        {
            "generate_text_only": "generate_text_only",
            "extract_visual_facts": "extract_visual_facts",
        },
    )
    graph.add_edge("extract_visual_facts", "generate_with_image")
    graph.add_edge("generate_text_only", "translate")
    graph.add_edge("generate_with_image", "translate")
    graph.add_edge("translate", "emotion")
    graph.add_edge("emotion", "tts")
    graph.add_edge("tts", "commit")
    graph.add_edge("commit", END)
    return graph.compile()
