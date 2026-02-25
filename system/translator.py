"""한국어 대사를 일본어 대사로 변환하는 번역기."""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class KoJaTranslator:
    """대사 전용 한국어-일본어 번역기."""

    DEFAULT_INSTRUCTION = "다음 한국어 문장을 자연스러운 일본어로 번역하시오."

    def __init__(
        self,
        base_model_dir: str = "/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/qwen3_core/model_assets/qwen3-1.7b-base",
        lora_dir: str = "/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/qwen3_core/model_assets/qwen3_1.7_ko2ja_lora/lora_adapter",
        device: str | None = None,
        instruction: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_dir,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 베이스 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        # LoRA 어댑터 결합
        self.model = PeftModel.from_pretrained(
            base_model,
            lora_dir,
        )

        self.model.eval()

    def _build_prompt(self, text_ko: str) -> str:
        return (
            "### 지시문:\n"
            f"{self.instruction}\n\n"
            "### 입력:\n"
            f"{text_ko}\n\n"
            "### 출력:\n"
        )

    @torch.no_grad()
    def translate(
        self,
        dialogue_ko: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        """한 줄 한국어 대사를 일본어로 번역한다."""

        if not dialogue_ko:
            return ""

        dialogue_ko = dialogue_ko.strip()

        # 대사 한 줄만 허용
        if "\n" in dialogue_ko:
            raise ValueError(
                "번역기는 여러 줄 입력을 받지 않는다. "
                "한 줄 대사만 전달해야 한다."
            )

        prompt = self._build_prompt(dialogue_ko)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # 출력 블록만 추출
        if "### 출력:" in decoded:
            decoded = decoded.split("### 출력:", 1)[1]

        return decoded.strip()
