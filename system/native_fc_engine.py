"""saya_rp_4b_v3용 네이티브 function calling 엔진."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class NativeFCGenerationConfig:
    """네이티브 function calling 엔진 생성 설정."""

    max_new_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.08
    no_repeat_ngram_size: int = 0
    do_sample: bool = True
    use_cache: bool = True


class NativeFunctionCallingEngine:
    """saya_rp_4b_v3의 chat template를 직접 쓰는 HF 전용 엔진."""

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        lora_model_id: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        default_gen: Optional[NativeFCGenerationConfig] = None,
    ) -> None:
        self.default_gen = default_gen or NativeFCGenerationConfig()
        project_root = Path(__file__).resolve().parents[1]
        self.resolved_base_model_id = str(
            Path(base_model_id)
            if base_model_id is not None
            else project_root / "models" / "qwen3_core" / "model_assets" / "saya_rp_4b_v3"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.resolved_base_model_id,
            use_fast=False,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
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
            self.model = PeftModel.from_pretrained(self.model, lora_model_id, subfolder="lora")
        self.model.eval()

    def build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> str:
        """메시지와 도구 목록을 chat template에 맞춰 prompt로 렌더한다."""
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        gen_config: Optional[NativeFCGenerationConfig] = None,
    ) -> str:
        """assistant 생성 문자열을 반환한다."""
        cfg = gen_config or self.default_gen
        prompt = self.build_prompt(messages, tools=tools)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs.pop("token_type_ids", None)
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
        return self.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        gen_config: Optional[NativeFCGenerationConfig] = None,
    ) -> str:
        """memory_chain 호환용 단순 assistant 생성 경로."""
        return self.generate(messages, tools=None, gen_config=gen_config)

    @staticmethod
    def parse_tool_output(raw_text: str) -> tuple[str, list[dict[str, Any]]]:
        """assistant 출력에서 일반 텍스트와 tool call 목록을 분리한다."""
        tool_calls: list[dict[str, Any]] = []
        pattern = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
        for match in pattern.finditer(raw_text):
            try:
                payload = json.loads(match.group(1))
            except Exception:
                continue
            name = str(payload.get("name", "")).strip()
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            if name:
                tool_calls.append({"name": name, "arguments": arguments})

        cleaned = pattern.sub("", raw_text)
        cleaned = cleaned.replace("<|im_end|>", "").strip()
        return cleaned, tool_calls

    def infer_emotion_json(self, narration: str, dialogue_ko: str) -> dict[str, int] | None:
        """감정 one-hot JSON을 추론한다."""
        text = f"{narration}\n{dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 감정 분류기다. 입력된 사야의 서술/대사를 보고 "
                    "neutral,sad,happy,angry 키만 가진 one-hot JSON 객체만 출력하라."
                ),
            },
            {"role": "user", "content": f"사야 출력:\n{text}"},
        ]
        raw = self.generate(
            messages,
            gen_config=NativeFCGenerationConfig(
                max_new_tokens=96,
                temperature=0.0,
                top_p=1.0,
                top_k=20,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                do_sample=False,
                use_cache=True,
            ),
        )
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
        out = {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}
        vals: dict[str, int] = {}
        for key in out:
            try:
                vals[key] = 1 if int(parsed.get(key, 0)) > 0 else 0
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

    def run_tool_loop(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict[str, Any]], str],
        planning_gen: Optional[NativeFCGenerationConfig] = None,
        final_gen: Optional[NativeFCGenerationConfig] = None,
        max_rounds: int = 3,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """네이티브 tool-call 루프를 돌린 뒤 최종 assistant 텍스트를 반환한다."""
        conversation = list(messages)
        tool_trace: list[dict[str, Any]] = []
        planning_cfg = planning_gen or NativeFCGenerationConfig(
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
            top_k=20,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            do_sample=False,
            use_cache=True,
        )
        final_cfg = final_gen or self.default_gen

        for _ in range(max_rounds):
            raw = self.generate(conversation, tools=tools, gen_config=planning_cfg)
            assistant_text, tool_calls = self.parse_tool_output(raw)
            if not tool_calls:
                if assistant_text:
                    return assistant_text, tool_trace, conversation
                break
            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "tool_calls": tool_calls,
                }
            )
            for tool_call in tool_calls:
                name = str(tool_call.get("name", "")).strip()
                arguments = tool_call.get("arguments", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                result = tool_executor(name, arguments)
                tool_trace.append(
                    {
                        "name": name,
                        "args": arguments,
                        "result_preview": result.replace("\n", " ")[:240],
                    }
                )
                conversation.append({"role": "tool", "content": result})

        raw = self.generate(conversation, tools=tools, gen_config=final_cfg)
        assistant_text, _ = self.parse_tool_output(raw)
        return assistant_text, tool_trace, conversation
