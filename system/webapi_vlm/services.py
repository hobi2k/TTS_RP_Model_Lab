from __future__ import annotations

"""VLM 전용 런타임 서비스 컨테이너.

qwen/vLLM 메모리 체인과 분리된 단순 순차 파이프라인을 제공한다.
"""

import os
from collections import deque
from pathlib import Path
from threading import Lock, RLock

from system.memory_chain import SummaryMemoryChain, SummaryMemoryConfig
from system.prompt_compiler import CharacterProfile, PromptCompiler
from system.rp_parser import RPParser
from system.vlm_langgraph_pipeline import build_vlm_turn_graph
from system.vlm_engine import KananaVLMEngine, VLMGenerationConfig


class RuntimeServices:
    """VLM 전용 REST 서비스 모음."""

    def __init__(self) -> None:
        self._load_lock = RLock()
        self._turn_lock = Lock()
        self._parser = RPParser()
        self._history: deque[dict] = deque(maxlen=6)
        self._vlm = None
        self._prompt_compiler = PromptCompiler(CharacterProfile(name="사야"))
        self._memory_chain = None
        self._turn_graph = None
        self._translator = None
        self._tts = None
        self._memory_debug = os.getenv("VLM_MEMORY_DEBUG", "0") == "1"

        project_root = Path(__file__).resolve().parents[2]
        self.vlm_model = os.getenv(
            "KANANA_VLM_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "saya_vlm_3b"),
        )
        self.trans_base = os.getenv(
            "TRANS_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "qtranslator_1.7b_v2"),
        )

    def _ensure_vlm(self) -> KananaVLMEngine:
        """VLM 엔진을 필요 시 지연 로딩한다."""
        if self._vlm is not None:
            return self._vlm
        with self._load_lock:
            if self._vlm is None:
                self._vlm = KananaVLMEngine(
                    model_dir=self.vlm_model,
                    load_in_4bit=os.getenv("KANANA_VLM_LOAD_IN_4BIT", "0") == "1",
                    trust_remote_code=os.getenv("KANANA_VLM_TRUST_REMOTE_CODE", "1") == "1",
                    attn_implementation=os.getenv("KANANA_VLM_ATTN_IMPL", "flash_attention_2"),
                    default_gen=VLMGenerationConfig(
                        max_length=int(os.getenv("KANANA_VLM_MAX_LENGTH", "4096")),
                        max_new_tokens=220,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.05,
                        dummy_image_size=int(os.getenv("KANANA_VLM_DUMMY_IMAGE_SIZE", "224")),
                    ),
                )
        return self._vlm

    def _ensure_translator(self):
        """번역기 인스턴스를 필요 시 생성하고 반환한다."""
        if self._translator is not None:
            return self._translator
        with self._load_lock:
            if self._translator is None:
                from system.translator import KoJaTranslator

                self._translator = KoJaTranslator(model_dir=self.trans_base)
        return self._translator

    def _ensure_tts(self):
        """SBV2 워커 클라이언트를 필요 시 생성하고 반환한다."""
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is None:
                from system.tts_worker_client import SBV2WorkerClient

                self._tts = SBV2WorkerClient()
        return self._tts

    @staticmethod
    def _normalize_single_line_dialogue(text: str) -> str:
        """멀티라인 대사를 번역기 입력 규칙(한 줄)으로 정규화한다."""
        if not text:
            return ""
        return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    @staticmethod
    def _infer_emotion_keyword(user_text: str, narration: str, dialogue_ko: str) -> dict[str, int]:
        """키워드 기반 감정 one-hot 폴백."""
        text = f"{user_text} {narration} {dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

        sad_kw = ("슬퍼", "울", "눈물", "외로", "힘들", "아파", "불안", "우울", "미안", "실망")
        happy_kw = ("좋아", "행복", "웃", "기뻐", "반가", "신나", "즐거", "설레")
        angry_kw = ("화나", "짜증", "분노", "빡", "싫어", "미워", "그만", "닥쳐", "거짓말")

        score_sad = sum(1 for k in sad_kw if k in text)
        score_happy = sum(1 for k in happy_kw if k in text)
        score_angry = sum(1 for k in angry_kw if k in text)

        if score_angry > 0 and score_angry >= max(score_sad, score_happy):
            return {"neutral": 0, "sad": 0, "happy": 0, "angry": 1}
        if score_sad > 0 and score_sad >= score_happy:
            return {"neutral": 0, "sad": 1, "happy": 0, "angry": 0}
        if score_happy > 0:
            return {"neutral": 0, "sad": 0, "happy": 1, "angry": 0}
        return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

    def _build_messages(self, text: str) -> list[dict]:
        """현재 user 입력에 대한 VLM 입력 메시지 배열을 만든다."""
        messages: list[dict] = []
        messages.extend(self._prompt_compiler.compile())
        memory_msg = self._memory_chain.build_memory_system_message(current_user_text=text)
        if memory_msg is not None:
            if self._memory_debug:
                preview = str(memory_msg.get("content", "")).replace("\n", " ")[:240]
                print(f"[vlm-memory] chat inject: {preview}")
            messages.append(memory_msg)
        messages.extend(self._history)
        messages.append({"role": "user", "content": text})
        return messages

    def _ensure_memory_chain(self):
        """장기 기억 체인을 필요 시 생성하고 반환한다."""
        if self._memory_chain is not None:
            return self._memory_chain
        with self._load_lock:
            if self._memory_chain is None:
                self._memory_chain = SummaryMemoryChain(
                    self._ensure_vlm(),
                    SummaryMemoryConfig(enabled=True, update_every_turns=1, max_summary_chars=900),
                )
        return self._memory_chain

    def _ensure_turn_graph(self):
        """LangGraph 기반 VLM 턴 파이프라인을 필요 시 생성하고 반환한다."""
        if self._turn_graph is not None:
            return self._turn_graph
        with self._load_lock:
            if self._turn_graph is None:
                self._ensure_memory_chain()
                self._turn_graph = build_vlm_turn_graph(
                    llm_engine=self._ensure_vlm(),
                    rp_parser=self._parser,
                    translator=self._ensure_translator(),
                    tts_synth=lambda text_ja, style_index, style_weight, speaker_name: self.tts(
                        text_ja,
                        style_index=style_index,
                        style_weight=style_weight,
                        speaker_name=speaker_name,
                    ),
                    prompt_compiler=self._prompt_compiler,
                    memory_chain=self._memory_chain,
                    debug_memory=self._memory_debug,
                    history=self._history,
                    normalize_dialogue=self._normalize_single_line_dialogue,
                    infer_emotion_fallback=self._infer_emotion_keyword,
                )
        return self._turn_graph

    def chat(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        image_path: str | None = None,
    ) -> str:
        """VLM 단일 응답을 생성하고 히스토리를 갱신한다."""
        with self._turn_lock:
            vlm = self._ensure_vlm()
            self._ensure_memory_chain()
            messages = self._build_messages(text)
            raw_text = vlm.generate_from_messages(
                messages,
                gen_config=VLMGenerationConfig(
                    max_length=vlm.default_gen.max_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.05,
                    dummy_image_size=vlm.default_gen.dummy_image_size,
                ),
                image_path=image_path,
            )
            rp = self._parser.parse(raw_text)
            mem_assistant_text = "\n".join(
                p for p in [rp.narration.strip(), self._normalize_single_line_dialogue(rp.dialogue_en or "")]
                if p
            ).strip() or raw_text
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": raw_text})
            self._memory_chain.update(user_text=text, assistant_text=mem_assistant_text)
            return raw_text

    def translate(self, text_ko: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        """한 줄 한국어 대사를 일본어로 번역한다."""
        translator = self._ensure_translator()
        return translator.translate(
            text_ko,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def parse(self, text: str):
        """RP 원문 텍스트를 구조화 블록으로 파싱한다."""
        return self._parser.parse(text)

    def tts(self, text_ja: str, style_index: int, style_weight: float, speaker_name: str = "saya") -> str:
        """일본어 텍스트를 SBV2 워커로 합성한다."""
        client = self._ensure_tts()
        return client.speak(
            text_ja,
            style_index=style_index,
            style_weight=style_weight,
            speaker_name=speaker_name,
        )

    def turn(
        self,
        text_ko: str,
        style_index: int,
        style_weight: float,
        speaker_name: str = "saya",
        image_path: str | None = None,
    ):
        """LangGraph 기반 VLM 메인 턴 파이프라인을 실행한다."""
        with self._turn_lock:
            graph = self._ensure_turn_graph()
            result = graph.invoke(
                {
                    "text_ko": text_ko,
                    "image_path": image_path,
                    "style_index": style_index,
                    "style_weight": style_weight,
                    "speaker_name": speaker_name,
                }
            )
            return {
                "rp_text": result.get("rp_text", ""),
                "narration": result.get("narration", ""),
                "dialogue_ko": result.get("dialogue_ko", ""),
                "dialogue_ja": result.get("dialogue_ja", ""),
                "wav_path": result.get("wav_path"),
                "emotion": result.get("emotion") or {"neutral": 1, "sad": 0, "happy": 0, "angry": 0},
            }
