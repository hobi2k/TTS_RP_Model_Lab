from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from threading import Lock, RLock

from system.memory_chain import SummaryMemoryChain, SummaryMemoryConfig
from system.prompt_compiler import CharacterProfile, PromptCompiler
from system.rp_parser import RPParser
from system.tool_use_pipeline import (
    _extract_json_object,
    _planner_gen_config,
    _planner_messages,
    build_tool_use_turn_graph,
)


class RuntimeServices:
    """tool-use 전용 런타임 서비스 컨테이너."""

    def __init__(self) -> None:
        self._load_lock = RLock()
        self._turn_lock = Lock()
        self._history: deque[dict] = deque(maxlen=6)
        self._parser = RPParser()
        self._llm = None
        self._vlm = None
        self._translator = None
        self._tts = None
        self._memory_chain = None
        self._turn_graph = None
        self._prompt_compiler = PromptCompiler(CharacterProfile(name="사야"))

        project_root = Path(__file__).resolve().parents[2]
        self.qwen_model = os.getenv(
            "TOOLUSE_QWEN_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_4b_v3"),
        )
        self.vlm_model = os.getenv(
            "TOOLUSE_VLM_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "saya_vlm_3b"),
        )
        self.trans_base = os.getenv(
            "TRANS_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "qtranslator_1.7b_v2"),
        )

    def _ensure_llm(self):
        """planner/composer용 Qwen 엔진을 로드한다."""
        if self._llm is not None:
            return self._llm
        with self._load_lock:
            if self._llm is None:
                os.environ.setdefault("LLM_BACKEND", os.getenv("TOOLUSE_LLM_BACKEND", "hf"))
                os.environ.setdefault("LLM_STRICT_VLLM", "0")
                from system.llm_engine import GenerationConfig, QwenEngine

                self._llm = QwenEngine(
                    base_model_id=self.qwen_model,
                    default_gen=GenerationConfig(
                        max_new_tokens=200,
                        temperature=0.75,
                        top_p=0.85,
                        top_k=50,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=0,
                        do_sample=True,
                        use_cache=True,
                    ),
                )
        return self._llm

    def _ensure_vlm(self):
        """image_analyze tool용 VLM 엔진을 로드한다."""
        if self._vlm is not None:
            return self._vlm
        with self._load_lock:
            if self._vlm is None:
                from system.vlm_engine import KananaVLMEngine, VLMGenerationConfig

                self._vlm = KananaVLMEngine(
                    model_dir=self.vlm_model,
                    load_in_4bit=os.getenv("KANANA_VLM_LOAD_IN_4BIT", "0") == "1",
                    trust_remote_code=os.getenv("KANANA_VLM_TRUST_REMOTE_CODE", "1") == "1",
                    attn_implementation=os.getenv("KANANA_VLM_ATTN_IMPL", "flash_attention_2"),
                    default_gen=VLMGenerationConfig(
                        max_length=int(os.getenv("KANANA_VLM_MAX_LENGTH", "4096")),
                        max_new_tokens=180,
                        do_sample=False,
                        temperature=0.2,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.0,
                        dummy_image_size=int(os.getenv("KANANA_VLM_DUMMY_IMAGE_SIZE", "224")),
                    ),
                )
        return self._vlm

    def _ensure_translator(self):
        """번역기를 로드한다."""
        if self._translator is not None:
            return self._translator
        with self._load_lock:
            if self._translator is None:
                from system.translator import KoJaTranslator

                self._translator = KoJaTranslator(model_dir=self.trans_base)
        return self._translator

    def _ensure_tts(self):
        """TTS 클라이언트를 로드한다."""
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is None:
                from system.tts_worker_client import SBV2WorkerClient

                self._tts = SBV2WorkerClient()
        return self._tts

    def _ensure_memory_chain(self):
        """tool-use 경로 전용 memory chain을 로드한다."""
        if self._memory_chain is not None:
            return self._memory_chain
        with self._load_lock:
            if self._memory_chain is None:
                self._memory_chain = SummaryMemoryChain(
                    self._ensure_llm(),
                    SummaryMemoryConfig(enabled=True, update_every_turns=1, max_summary_chars=900),
                )
        return self._memory_chain

    def _ensure_turn_graph(self):
        """tool-use LangGraph를 생성한다."""
        if self._turn_graph is not None:
            return self._turn_graph
        with self._load_lock:
            if self._turn_graph is None:
                self._ensure_memory_chain()
                self._turn_graph = build_tool_use_turn_graph(
                    llm_engine=self._ensure_llm(),
                    image_tool=self._ensure_vlm(),
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
                    history=self._history,
                    normalize_dialogue=self._normalize_single_line_dialogue,
                    infer_emotion_fallback=self._infer_emotion_keyword,
                )
        return self._turn_graph

    @staticmethod
    def _normalize_single_line_dialogue(text: str) -> str:
        """멀티라인 대사를 한 줄로 접는다."""
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

    def tool_plan(self, text_ko: str, image_path: str | None = None) -> list[dict]:
        """현재 입력에 대한 planner 결과만 반환한다."""
        llm = self._ensure_llm()
        raw = llm.generate_from_messages(
            _planner_messages(text_ko, image_path, bool(self._history)),
            gen_config=_planner_gen_config(llm),
        )
        parsed = _extract_json_object(raw)
        tool_calls = parsed.get("tool_calls", []) if isinstance(parsed, dict) else []
        return [call for call in tool_calls if isinstance(call, dict)]

    def tts(self, text_ja: str, style_index: int, style_weight: float, speaker_name: str = "saya") -> str:
        """일본어 텍스트를 TTS로 합성한다."""
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
        image_path: str | None,
        style_index: int,
        style_weight: float,
        speaker_name: str = "saya",
    ) -> dict:
        """tool-use 메인 턴을 실행한다."""
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
                "emotion": result.get("emotion", {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}),
                "tool_trace": result.get("tool_trace", []),
            }
