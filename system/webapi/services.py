from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from collections import deque

from system.rp_parser import RPParser
from system.prompt_compiler import PromptCompiler, CharacterProfile
from system.memory_chain import SummaryMemoryChain, SummaryMemoryConfig
from system.llm_engine import GenerationConfig


class RuntimeServices:
    """Lazy-loading service container for REST handlers.

    NOTE:
    - /api/turn path intentionally mirrors main_loop.py execution order.
    """

    def __init__(self) -> None:
        self._load_lock = Lock()
        self._turn_lock = Lock()
        self._parser = RPParser()
        self._history: deque[dict] = deque(maxlen=6)
        self._llm = None
        self._prompt_compiler = None
        self._memory_chain = None
        self._translator = None
        self._tts = None

        project_root = Path(__file__).resolve().parents[2]
        self.qwen_model = os.getenv(
            "QWEN_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_4b_v2"),
        )
        self.trans_base = os.getenv(
            "TRANS_MODEL_DIR",
            str(project_root / "models" / "qwen3_core" / "model_assets" / "qtranslator_1.7b"),
        )

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        with self._load_lock:
            if self._llm is None:
                from system.llm_engine import QwenEngine

                self._llm = QwenEngine(
                    base_model_id=self.qwen_model,
                    default_gen=GenerationConfig(
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        top_k=40,
                        repetition_penalty=1.12,
                        no_repeat_ngram_size=4,
                        do_sample=True,
                        use_cache=True,
                    ),
                )
        return self._llm

    def _ensure_translator(self):
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
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is None:
                from system.tts_worker_client import SBV2WorkerClient

                self._tts = SBV2WorkerClient()
        return self._tts

    def _ensure_mainloop_components(self):
        if self._prompt_compiler is None:
            profile = CharacterProfile(
                name="사야",
                persona="조용하고 내향적이다. 부드러운 말투를 사용하며 일상적인 대화를 선호한다.",
                speaking_style="짧고 자연스러운 문장으로 차분하게 말한다.",
            )
            self._prompt_compiler = PromptCompiler(profile)

        if self._memory_chain is None:
            self._memory_chain = SummaryMemoryChain(
                self._ensure_llm(),
                SummaryMemoryConfig(enabled=True, update_every_turns=1, max_summary_chars=900),
            )

    def _infer_emotion_keyword(self, user_text: str, narration: str, dialogue_ko: str) -> dict[str, int]:
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
        if not text:
            return ""
        # 번역기 입력 정책(한 줄)과 맞추기 위해 줄바꿈을 공백으로 접는다.
        return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    def chat(self, text: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
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
                repetition_penalty=1.12,
                no_repeat_ngram_size=4,
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
        translator = self._ensure_translator()
        return translator.translate(
            text_ko,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def parse(self, text: str):
        return self._parser.parse(text)

    def tts(self, text_ja: str, style_index: int, style_weight: float) -> str:
        client = self._ensure_tts()
        return client.speak(text_ja, style_index=style_index, style_weight=style_weight)

    def turn(self, text_ko: str, style_index: int, style_weight: float):
        # main_loop.py 흐름과 동일하게 상태(history/memory)를 직렬 처리
        with self._turn_lock:
            llm = self._ensure_llm()
            self._ensure_mainloop_components()
            parser = self._parser
            translator = self._ensure_translator()

            system_msgs = self._prompt_compiler.compile()
            messages: list[dict] = []
            messages.extend(system_msgs)

            memory_msg = self._memory_chain.build_memory_system_message(current_user_text=text_ko)
            if memory_msg is not None:
                messages.append(memory_msg)

            messages.extend(self._history)
            messages.append({"role": "user", "content": text_ko})

            prompt = llm.build_prompt(messages)
            raw_text = llm.generate(prompt)

            rp = parser.parse(raw_text)
            dialogue_ko = self._normalize_single_line_dialogue(rp.dialogue_en or "")
            dialogue_ja = ""
            wav_path = None

            if dialogue_ko:
                dialogue_ja = translator.translate(dialogue_ko)
                wav_path = self.tts(dialogue_ja, style_index=style_index, style_weight=style_weight)
            emotion = llm.infer_emotion_json(rp.narration, dialogue_ko)
            if emotion is None:
                emotion = self._infer_emotion_keyword(text_ko, rp.narration, dialogue_ko)

            # main_loop와 같은 history/memory 업데이트
            self._history.append({"role": "user", "content": text_ko})
            self._history.append({"role": "assistant", "content": raw_text})

            mem_assistant_text = "\n".join(
                p for p in [rp.narration.strip(), dialogue_ko] if p
            ).strip() or raw_text
            self._memory_chain.update(user_text=text_ko, assistant_text=mem_assistant_text)

            return {
                "rp_text": raw_text,
                "narration": rp.narration,
                "dialogue_ko": dialogue_ko,
                "dialogue_ja": dialogue_ja,
                "wav_path": wav_path,
                "emotion": emotion,
            }
