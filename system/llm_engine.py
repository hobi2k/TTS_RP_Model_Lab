"""LLM 추론 엔진.

역할:
- 채팅 메시지를 모델 입력 프롬프트로 변환한다.
- 모델 추론을 실행하고 assistant 텍스트를 반환한다.

백엔드 우선순위:
1) vLLM
2) transformers + bitsandbytes(4bit)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


@dataclass
class GenerationConfig:
    """생성 파라미터 설정값."""

    max_new_tokens: int = 180
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
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
        self.resolved_base_model_id = str(
            Path(base_model_id)
            if base_model_id is not None
            else project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_7b"
        )

        self.backend = "hf"
        self._vllm_engine = None
        self.model = None

        # 토크나이저는 공통으로 사용
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.resolved_base_model_id,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prefer_vllm = os.getenv("LLM_BACKEND", "vllm").lower() == "vllm"
        strict_vllm = os.getenv("LLM_STRICT_VLLM", "1") == "1"

        if prefer_vllm:
            try:
                self._init_vllm()
                self.backend = "vllm"
                return
            except Exception as e:
                if strict_vllm:
                    raise RuntimeError(
                        "vLLM init failed and strict mode is enabled. "
                        "Tune VLLM_MAX_MODEL_LEN/VLLM_GPU_MEMORY_UTILIZATION and retry. "
                        f"Root cause: {e}"
                    ) from e
                print(f"[llm_engine] vLLM init failed, fallback to transformers+bnb: {e}")

        self._init_hf_with_bnb(dtype=dtype, device_map=device_map, lora_model_id=lora_model_id)
        self.backend = "hf"

    def _init_vllm(self) -> None:
        from vllm import LLM

        gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
        # 7B + 단일 GPU 환경에서 32768 기본값은 KV cache OOM을 유발하기 쉬워
        # 보수적인 기본값(1536)으로 시작한다.
        max_model_len = int(max_model_len_env) if max_model_len_env else 1536
        max_num_seqs_env = os.getenv("VLLM_MAX_NUM_SEQS")
        max_num_seqs = int(max_num_seqs_env) if max_num_seqs_env else 1

        llm_kwargs = {
            "model": self.resolved_base_model_id,
            "dtype": os.getenv("VLLM_DTYPE", "auto"),
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_mem_util,
            "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
            "max_num_seqs": max_num_seqs,
            "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "0") == "1",
        }
        llm_kwargs["max_model_len"] = max_model_len

        # 요청사항: vLLM + bitsandbytes를 기본으로 사용.
        # 비활성화하려면 VLLM_QUANTIZATION=none
        quant_mode = os.getenv("VLLM_QUANTIZATION", "bitsandbytes").strip().lower()
        if quant_mode in {"bnb", "bitsandbytes"}:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"

        self._vllm_engine = LLM(**llm_kwargs)

    def _init_hf_with_bnb(
        self,
        *,
        dtype: torch.dtype,
        device_map: str,
        lora_model_id: Optional[str],
    ) -> None:
        # fallback은 bitsandbytes 4bit를 기본으로 사용
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=os.getenv("BNB_4BIT_QUANT_TYPE", "nf4"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_kwargs = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "quantization_config": quant_cfg,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.resolved_base_model_id,
            **model_kwargs,
        )

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

        if sum(vals.values()) == 0:
            return out
        if vals["angry"]:
            return {"neutral": 0, "sad": 0, "happy": 0, "angry": 1}
        if vals["sad"]:
            return {"neutral": 0, "sad": 1, "happy": 0, "angry": 0}
        if vals["happy"]:
            return {"neutral": 0, "sad": 0, "happy": 1, "angry": 0}
        return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

    def infer_emotion_json(self, narration: str, dialogue_ko: str) -> dict[str, int] | None:
        text = f"{narration}\n{dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

        messages = [
            {
                "role": "system",
                "content": """
                    너는 감정 분류기다. 입력된 사야의 서술/대사를 보고 감정을 one-hot JSON으로만 출력하라.
                    반드시 키는 neutral,sad,happy,angry 4개만 사용하고 값은 0 또는 1.
                    정확히 하나만 1이어야 한다.
                    출력은 JSON 객체 단 하나만 출력하라. 다른 문장 금지.
                    """,
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
    def _generate_hf(self, prompt: str, cfg: GenerationConfig) -> str:
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
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _generate_vllm(self, prompt: str, cfg: GenerationConfig) -> str:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repetition_penalty=cfg.repetition_penalty,
        )
        outputs = self._vllm_engine.generate([prompt], sampling_params)
        if not outputs:
            return ""
        if not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def generate(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> str:
        """프롬프트를 입력받아 assistant 생성 텍스트를 반환한다."""
        cfg = gen_config or self.default_gen
        if self.backend == "vllm" and self._vllm_engine is not None:
            return self._generate_vllm(prompt, cfg)
        return self._generate_hf(prompt, cfg)
