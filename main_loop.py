"""메인 대화 루프."""

from __future__ import annotations

import subprocess
import os
from collections import deque

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
            persona=(
                "조용하고 내향적이다. 부드러운 말투를 사용하며 일상적인 대화를 선호한다."
            ),
            speaking_style="짧고 자연스러운 문장으로 차분하게 말한다.",
        )

        # 핵심 컴포넌트 초기화
        self.prompt_compiler = PromptCompiler(profile)
        self.llm_engine = QwenEngine(
            default_gen=GenerationConfig(
                max_new_tokens=200,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.12,
                no_repeat_ngram_size=4,
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

        # 대화 이력 저장
        # user+assistant 2개 메시지가 1턴이므로 3턴 유지=6개 메시지
        self.history: deque[dict] = deque(maxlen=6)
        # role: "user" / "assistant"
        # content: LLM 원문 출력

    @staticmethod
    def _normalize_single_line_dialogue(text: str) -> str:
        if not text:
            return ""
        return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()

    def step(self, user_text: str) -> None:
        # 1) 시스템 프롬프트 구성
        system_msgs = self.prompt_compiler.compile()

        # 2) 메시지 묶기
        messages: list[dict] = []
        messages.extend(system_msgs)
        memory_msg = self.memory_chain.build_memory_system_message(current_user_text=user_text)
        if memory_msg is not None:
            messages.append(memory_msg)
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_text})

        # 3) 프롬프트 생성 및 응답 생성
        prompt = self.llm_engine.build_prompt(messages)
        raw_text = self.llm_engine.generate(prompt)

        # 4) RP 블록 파싱
        rp = self.rp_parser.parse(raw_text)

        # UI 출력: 서술 + 대사 2줄 형식
        if rp.narration:
            print(rp.narration)

        dialogue_ko = self._normalize_single_line_dialogue(rp.dialogue_en or "")
        if dialogue_ko:
            print(f"\"{dialogue_ko}\"")

        # 공용 LLM 감정 판정(JSON one-hot)
        emotion = self.llm_engine.infer_emotion_json(rp.narration, dialogue_ko)
        if emotion is None:
            emotion = {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
        print(f"[emotion] {emotion}")

        # 대사 번역 및 TTS
        if dialogue_ko:
            try:
                dialogue_ja = self.translator.translate(dialogue_ko)

                wav_path = self.tts_client.speak(
                    dialogue_ja,
                    speaker_name=self.speaker_name,
                )

                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", str(wav_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"[TTS ERROR] {e}")

        # 이력 업데이트
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": raw_text})

        # 요약 메모리 업데이트
        mem_assistant_text = "\n".join(
            p for p in [rp.narration.strip(), dialogue_ko] if p
        ).strip() or raw_text
        self.memory_chain.update(user_text=user_text, assistant_text=mem_assistant_text)


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
