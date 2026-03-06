from __future__ import annotations

import os
from pathlib import Path
from threading import Lock, RLock
from collections import deque

from system.langgraph_pipeline import build_turn_graph
from system.rp_parser import RPParser
from system.prompt_compiler import PromptCompiler, CharacterProfile
from system.memory_chain import SummaryMemoryChain, SummaryMemoryConfig
from system.llm_engine import GenerationConfig


class RuntimeServices:
    """
    Lazy-loading service container for REST handlers.

    - LLM, Translator, TTS 등 무거운 리소스는 실제 사용 시점까지 로딩을 지연한다.
    - main_loop.py와 동일한 LLM/Translator/TTS 인스턴스를 공유하여 일관된 상태(history/memory) 유지한다.
    - chat/translate/parse/tts/turn 등의 메서드를 통해 REST 핸들러에서 필요한 기능을 제공한다.
    """

    def __init__(self) -> None:
        """서비스 컨테이너를 초기화한다.

        실제 모델 로딩은 `_ensure_*` 메서드에서 지연(lazy) 수행된다.
        """
        self._load_lock = RLock()
        self._turn_lock = Lock()
        self._parser = RPParser()
        # user+assistant 2개 메시지가 1턴이므로 3턴 유지=6개 메시지
        self._history: deque[dict] = deque(maxlen=6)
        self._llm = None
        self._prompt_compiler = None
        self._memory_chain = None
        self._turn_graph = None
        self._translator = None
        self._tts = None

        project_root = Path(__file__).resolve().parents[2]
        self.qwen_model = os.getenv(
            "QWEN_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_4b_v3"),
        )
        self.trans_base = os.getenv(
            "TRANS_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "qtranslator_1.7b_v2"),
        )

    def _ensure_llm(self):
        """LLM 엔진 인스턴스를 필요 시 생성하고 반환한다."""
        if self._llm is not None:
            return self._llm
        with self._load_lock:
            if self._llm is None:
                from system.llm_engine import QwenEngine

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

    def _ensure_translator(self):
        """번역기 인스턴스를 필요 시 생성하고 반환한다."""
        if self._translator is not None:
            return self._translator
        with self._load_lock:
            if self._translator is None:
                from system.translator import KoJaTranslator

                self._translator = KoJaTranslator(
                    model_dir=self.trans_base,
                )
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

    def _ensure_mainloop_components(self):
        """main_loop 파이프라인의 부가 컴포넌트를 초기화한다."""
        if self._prompt_compiler is None:
            profile = CharacterProfile(
                name="사야",
            )
            self._prompt_compiler = PromptCompiler(profile)

        if self._memory_chain is None:
            self._memory_chain = SummaryMemoryChain(
                self._ensure_llm(),
                SummaryMemoryConfig(enabled=True, update_every_turns=1, max_summary_chars=900),
            )

    def _ensure_turn_graph(self):
        """LangGraph 기반 턴 파이프라인을 필요 시 생성하고 반환한다."""
        if self._turn_graph is not None:
            return self._turn_graph
        with self._load_lock:
            if self._turn_graph is None:
                self._ensure_mainloop_components()
                self._turn_graph = build_turn_graph(
                    llm_engine=self._ensure_llm(),
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

    def _infer_emotion_keyword(self, user_text: str, narration: str, dialogue_ko: str) -> dict[str, int]:
        """키워드 기반 감정 one-hot을 추정한다.

        LLM 감정 분류(JSON)가 실패했을 때 사용할 폴백 로직이다.
        """
        text = f"{user_text} {narration} {dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

        sad_kw = ("슬퍼", "울", "눈물", "외로", "힘들", "아파", "불안", "우울", "미안", "실망", "기대했")
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

    @staticmethod
    def _normalize_single_line_dialogue(text: str) -> str:
        """멀티라인 대사를 번역기 입력 규칙(한 줄)으로 정규화한다."""
        if not text:
            return ""
        # 번역기 입력 정책(한 줄)과 맞추기 위해 줄바꿈을 공백으로 접는다.
        return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    def chat(self, text: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        """LLM RP 원문을 생성하고 history/memory 상태를 갱신한다."""
        # main_loop와 동일한 prompt/history/memory 경로를 사용한다.
        with self._turn_lock:
            llm = self._ensure_llm()
            self._ensure_mainloop_components()

            system_msgs = self._prompt_compiler.compile()
            messages: list[dict] = []
            messages.extend(system_msgs)

            memory_msg = self._memory_chain.build_memory_system_message(current_user_text=text)
            if memory_msg is not None:
                messages.append(memory_msg)

            messages.extend(self._history)
            messages.append({"role": "user", "content": text})

            prompt = llm.build_prompt(messages)

            # 기본값은 main_loop GenerationConfig와 동일
            cfg = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.1,
                no_repeat_ngram_size=0,
                do_sample=True,
                use_cache=True,
            )
            raw_text = llm.generate(prompt, cfg)

            rp = self._parser.parse(raw_text)
            mem_assistant_text = "\n".join(
                p for p in [rp.narration.strip(), rp.dialogue_en.strip()] if p
            ).strip() or raw_text

            # main_loop와 동일하게 history/memory 갱신
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

    def turn(self, text_ko: str, style_index: int, style_weight: float, speaker_name: str = "saya"):
        """LangGraph 기반 메인 턴 파이프라인을 실행한다.

        순서:
        1) LLM RP 생성
        2) RP 파싱
        3) 한국어 대사 추출/정규화
        4) 일본어 번역
        5) TTS 합성
        6) 감정 추정 + history/memory 저장
        """
        # history/memory 공유 상태를 직렬 처리한다.
        with self._turn_lock:
            graph = self._ensure_turn_graph()
            result = graph.invoke(
                {
                    "text_ko": text_ko,
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
