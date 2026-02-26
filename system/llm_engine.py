"""LLM 추론 엔진.

역할:
- 채팅 메시지를 모델 입력 프롬프트로 변환한다.
- 모델 추론을 실행하고 assistant 텍스트를 반환한다.
"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


@dataclass
class GenerationConfig:
    """생성 파라미터 설정값."""
    max_new_tokens: int = 180
    temperature: float = 0.6
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
            dtype=dtype,
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

    @staticmethod
    def _normalize_emotion(raw: dict | None) -> dict[str, int]:
        out = {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
        if not isinstance(raw, dict):
            return out

        vals: dict[str, int] = {}
        for key in out:
            try:
                vals[key] = 1 if int(raw.get(key, 0)) > 0 else 0
            except Exception:
                vals[key] = 0

        # one-hot 보장
        if sum(vals.values()) == 0:
            return out
        if vals["angry"]:
            return {"neutral": 0, "sad": 0, "happy": 0, "angry": 1}
        if vals["sad"]:
            return {"neutral": 0, "sad": 1, "happy": 0, "angry": 0}
        if vals["happy"]:
            return {"neutral": 0, "sad": 0, "happy": 1, "angry": 0}
        return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

    @torch.no_grad()
    def infer_emotion_json(self, narration: str, dialogue_ko: str) -> dict[str, int] | None:
        text = f"{narration}\n{dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

        messages = [
            {
                "role": "system",
                "content": """너는 감정 분류기다. 입력된 사야의 서술/대사를 보고 감정을 one-hot JSON으로만 출력하라.
반드시 키는 neutral,sad,happy,angry 4개만 사용하고 값은 0 또는 1.
정확히 하나만 1이어야 한다.
출력은 JSON 객체 단 하나만 출력하라. 다른 문장 금지.""",
            },
            {
                "role": "user",
                "content": f"사야 출력:\n{text}",
            },
        ]
        prompt = self.build_prompt(messages)
        cfg = GenerationConfig(
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            do_sample=False,
            use_cache=True,
        )
        raw = self.generate(prompt, cfg)
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return self._normalize_emotion(parsed)
        except Exception:
            return None

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
