"""LangChain 기반 요약 메모리 체인."""

from __future__ import annotations

from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate

from system.llm_engine import GenerationConfig, QwenEngine


@dataclass
class SummaryMemoryConfig:
    """요약 메모리 동작 설정."""
    enabled: bool = True
    update_every_turns: int = 1
    max_summary_chars: int = 900


class SummaryMemoryChain:
    """대화 요약을 누적하고 system 메모리로 제공한다."""

    def __init__(self, llm_engine: QwenEngine, config: SummaryMemoryConfig | None = None) -> None:
        self.llm_engine = llm_engine
        self.config = config or SummaryMemoryConfig()
        self.summary_text: str = ""
        self.turn_count: int = 0

        self.prompt_tmpl = PromptTemplate.from_template(
            "너는 대화 메모리 요약기다.\n"
            "아래 정보를 바탕으로 요약 메모리를 갱신하라.\n"
            "- 사실과 관계 변화만 남겨라.\n"
            "- 말투 규칙, 메타 설명, 시스템 문구는 쓰지 마라.\n"
            "- 6문장 이내 한국어 평문으로 써라.\n\n"
            "[기존 요약]\n{prev_summary}\n\n"
            "[이번 user 발화]\n{user_text}\n\n"
            "[이번 assistant 응답]\n{assistant_text}\n\n"
            "[출력]\n"
        )

        self.summary_gen = GenerationConfig(
            max_new_tokens=180,
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            do_sample=False,
            use_cache=True,
        )

    def build_memory_system_message(self) -> dict | None:
        """요약 메모리를 system 메시지로 반환한다."""
        if not self.config.enabled:
            return None
        if not self.summary_text.strip():
            return None
        return {
            "role": "system",
            "content": (
                "대화 요약 메모리:\n"
                f"{self.summary_text.strip()}\n\n"
                "위 요약을 참고하되, 반드시 user의 마지막 발화에 직접 반응해라."
            ),
        }

    def update(self, user_text: str, assistant_text: str) -> None:
        """새 턴 정보를 반영해 요약 메모리를 갱신한다."""
        if not self.config.enabled:
            return
        self.turn_count += 1
        if self.turn_count % max(1, self.config.update_every_turns) != 0:
            return

        payload = self.prompt_tmpl.format(
            prev_summary=self.summary_text.strip() or "(없음)",
            user_text=user_text.strip(),
            assistant_text=assistant_text.strip(),
        )

        msgs = [
            {"role": "system", "content": "너는 요약 전용 보조 모델이다. 출력에는 요약 본문만 작성한다."},
            {"role": "user", "content": payload},
        ]
        prompt = self.llm_engine.build_prompt(msgs)
        out = self.llm_engine.generate(prompt, gen_config=self.summary_gen)

        cleaned = (out or "").strip()
        if not cleaned:
            return
        if len(cleaned) > self.config.max_summary_chars:
            cleaned = cleaned[: self.config.max_summary_chars].rstrip()
        self.summary_text = cleaned

