"""LLM 추론 엔진.

역할:
- 채팅 메시지를 모델 입력 프롬프트로 변환한다.
- 모델 추론을 실행하고 assistant 텍스트를 반환한다.
"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


@dataclass
class GenerationConfig:
    """생성 파라미터 설정값."""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 4
    do_sample: bool = True
    use_cache: bool = True


class QwenEngine:
    """Qwen 기반 추론 엔진."""

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        lora_model_id: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        default_gen: Optional[GenerationConfig] = None,
    ) -> None:
        self.default_gen = default_gen or GenerationConfig()
        project_root = Path(__file__).resolve().parents[1]
        resolved_base_model_id = str(
            Path(base_model_id)
            if base_model_id is not None
            else project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_4b"
        )

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_base_model_id,
            use_fast=False,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 베이스 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # 필요 시 LoRA 어댑터 결합
        if lora_model_id:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_model_id,
                subfolder="lora",
            )

        self.model.eval()

    def build_prompt(self, messages: List[dict]) -> str:
        """채팅 메시지를 모델 프롬프트 문자열로 변환한다."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> str:
        """프롬프트를 입력받아 assistant 생성 텍스트를 반환한다."""
        cfg = gen_config or self.default_gen
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[-1]

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            do_sample=cfg.do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=cfg.use_cache,
        )

        gen_ids = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 최소 정제만 수행
        return text.strip()
