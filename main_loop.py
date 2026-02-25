"""메인 대화 루프."""

from __future__ import annotations

import subprocess
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
                max_new_tokens=256,
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

        # 대화 이력 저장
        self.history: deque[dict] = deque(maxlen=6)
        # role: "user" / "assistant"
        # content: LLM 원문 출력

    def step(self, user_text: str) -> None:
        # 1) 시스템 프롬프트 구성
        system_msgs = self.prompt_compiler.compile()

        # 2) 메시지 묶기
        messages: list[dict] = []
        messages.extend(system_msgs)
        memory_msg = self.memory_chain.build_memory_system_message()
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

        if rp.dialogue_en:
            print(f"\"{rp.dialogue_en}\"")

        # 대사 번역 및 TTS
        if rp.dialogue_en:
            try:
                dialogue_ja = self.translator.translate(rp.dialogue_en)

                wav_path = self.tts_client.speak(dialogue_ja)

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
            p for p in [rp.narration.strip(), rp.dialogue_en.strip()] if p
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
