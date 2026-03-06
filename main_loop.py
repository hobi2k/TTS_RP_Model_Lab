"""메인 대화 루프."""

from __future__ import annotations

import subprocess
import os
from collections import deque

from system.langgraph_pipeline import build_turn_graph
from system.llm_engine import QwenEngine, GenerationConfig
from system.memory_chain import SummaryMemoryChain, SummaryMemoryConfig
from system.prompt_compiler import PromptCompiler, CharacterProfile
from system.translator import KoJaTranslator
from system.tts_worker_client import SBV2WorkerClient
from system.rp_parser import RPParser


class MainLoop:
    def __init__(self) -> None:
        # 캐릭터 기본 설정
        profile = CharacterProfile(
            name="사야",
        )

        # 핵심 컴포넌트 초기화
        self.prompt_compiler = PromptCompiler(profile)
        self.llm_engine = QwenEngine(
            default_gen=GenerationConfig(
                max_new_tokens=200,
                temperature=0.75,
                top_p=0.85,
                top_k=50,
                repetition_penalty=1.1,
                no_repeat_ngram_size=0,
                do_sample=True,
                use_cache=True,
            )
        )
        self.rp_parser = RPParser()
        self.memory_chain = SummaryMemoryChain(
            self.llm_engine,
            SummaryMemoryConfig(enabled=True, update_every_turns=1, max_summary_chars=900),
        )
        self.translator = KoJaTranslator()
        self.tts_client = SBV2WorkerClient()
        self.speaker_name = os.getenv("TTS_SPEAKER_NAME", "saya").strip().lower()
        self.style_index = int(os.getenv("TTS_STYLE_INDEX", "0"))
        self.style_weight = float(os.getenv("TTS_STYLE_WEIGHT", "1.0"))

        # 대화 이력 저장
        # user+assistant 2개 메시지가 1턴이므로 3턴 유지=6개 메시지
        self.history: deque[dict] = deque(maxlen=6)
        # role: "user" / "assistant"
        # content: LLM 원문 출력

        self.turn_graph = build_turn_graph(
            llm_engine=self.llm_engine,
            rp_parser=self.rp_parser,
            translator=self.translator,
            tts_synth=lambda text_ja, style_index, style_weight, speaker_name: self.tts_client.speak(
                text_ja,
                style_index=style_index,
                style_weight=style_weight,
                speaker_name=speaker_name,
            ),
            prompt_compiler=self.prompt_compiler,
            memory_chain=self.memory_chain,
            history=self.history,
            normalize_dialogue=self._normalize_single_line_dialogue,
            infer_emotion_fallback=self._infer_emotion_keyword,
        )

    @staticmethod
    def _normalize_single_line_dialogue(text: str) -> str:
        if not text:
            return ""
        return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    @staticmethod
    def _infer_emotion_keyword(user_text: str, narration: str, dialogue_ko: str) -> dict[str, int]:
        """키워드 기반 감정 one-hot 폴백."""
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

    def step(self, user_text: str) -> None:
        result = self.turn_graph.invoke(
            {
                "text_ko": user_text,
                "style_index": self.style_index,
                "style_weight": self.style_weight,
                "speaker_name": self.speaker_name,
            }
        )

        # UI 출력: 서술 + 대사 2줄 형식
        narration = result.get("narration", "")
        dialogue_ko = result.get("dialogue_ko", "")
        emotion = result.get("emotion") or {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
        wav_path = result.get("wav_path")

        if narration:
            print(narration)

        if dialogue_ko:
            print(f"\"{dialogue_ko}\"")

        print(f"[emotion] {emotion}")

        if wav_path:
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", str(wav_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[TTS ERROR] {e}")


if __name__ == "__main__":
    loop = MainLoop()
    print("사야 준비 완료. 종료하려면 'exit'를 입력하세요.\n")

    try:
        while True:
            user_input = input("나: ").strip()
            if user_input.lower() == "exit":
                break

            loop.step(user_input)

    finally:
        loop.tts_client.close()
