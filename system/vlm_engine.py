"""Kanana 기반 VLM 추론 엔진.

이 모듈은 qwen/vLLM 경로와 분리된 VLM 전용 추론 래퍼를 제공한다.
현재 파이프라인은 text-only RP 데모를 우선 목표로 하므로,
모델 입력 형식은 Kanana 권장 conv + dummy image 포맷을 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


@dataclass
class VLMGenerationConfig:
    """VLM 텍스트 생성 파라미터 묶음."""

    max_length: int = 4096
    max_new_tokens: int = 220
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    dummy_image_size: int = 224


class KananaVLMEngine:
    """Kanana VLM 기반 text-only RP 추론 엔진."""

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
        default_gen: Optional[VLMGenerationConfig] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.model_dir = str(
            Path(model_dir)
            if model_dir is not None
            else project_root / "models" / "qwen3_core" / "model_assets" / "saya_vlm_3b"
        )
        self.default_gen = default_gen or VLMGenerationConfig()
        self._trust_remote_code = trust_remote_code

        dtype = torch.bfloat16
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quant_config,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    @staticmethod
    def _build_conv(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """OpenAI-style messages를 Kanana 학습 포맷과 맞는 conv로 변환한다."""
        conv: list[dict[str, str]] = []
        image_inserted = False
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                conv.append({"role": "system", "content": content})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": content})
            else:
                if not image_inserted:
                    content = f"<image>\n{content}"
                    image_inserted = True
                conv.append({"role": "user", "content": content})
        return conv

    def generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        gen_config: Optional[VLMGenerationConfig] = None,
        image_path: str | None = None,
    ) -> str:
        """메시지 배열을 바로 생성에 연결하는 헬퍼."""
        return self.generate(
            self._build_conv(messages),
            gen_config=gen_config,
            image_path=image_path,
        )

    def _extract_assistant_text(self, decoded: str) -> str:
        """전체 디코드 문자열에서 마지막 assistant 구간만 추출한다."""
        text = decoded.strip()
        if not text:
            return ""

        if "[어시스턴트]" in text:
            text = text.rsplit("[어시스턴트]", 1)[-1].strip()

        cut_markers = ["[사용자]", "[시스템]", "<|im_end|>", "<end_of_turn>"]
        for marker in cut_markers:
            if marker in text:
                text = text.split(marker, 1)[0].strip()

        # conv 포맷 잔여 태그를 걷어낸다.
        text = re.sub(r"^<image>\s*", "", text).strip()
        return text

    @torch.inference_mode()
    def generate(
        self,
        prompt: list[dict[str, str]],
        gen_config: Optional[VLMGenerationConfig] = None,
        image_path: str | None = None,
    ) -> str:
        """Kanana conv 입력으로 assistant 응답 텍스트를 생성한다."""
        cfg = gen_config or self.default_gen
        max_length = int(getattr(cfg, "max_length", self.default_gen.max_length))
        max_new_tokens = int(getattr(cfg, "max_new_tokens", self.default_gen.max_new_tokens))
        do_sample = bool(getattr(cfg, "do_sample", self.default_gen.do_sample))
        temperature = float(getattr(cfg, "temperature", self.default_gen.temperature))
        top_p = float(getattr(cfg, "top_p", self.default_gen.top_p))
        top_k = int(getattr(cfg, "top_k", self.default_gen.top_k))
        repetition_penalty = float(
            getattr(cfg, "repetition_penalty", self.default_gen.repetition_penalty)
        )
        dummy_image_size = int(
            getattr(cfg, "dummy_image_size", self.default_gen.dummy_image_size)
        )

        image = self._load_image(image_path, dummy_image_size)

        batch = self.processor.batch_encode_collate(
            data_list=[{"conv": prompt, "image": [image]}],
            padding="longest",
            padding_side="right",
            max_length=max_length,
            add_generation_prompt=True,
        )

        model_device = next(self.model.parameters()).device
        inputs: dict[str, Any] = {}
        for key, value in batch.items():
            inputs[key] = value.to(model_device) if isinstance(value, torch.Tensor) else value

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
        }
        if self.tokenizer is not None:
            if self.tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            if self.tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if do_sample:
            gen_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        output = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.tokenizer.decode(
            output[0],
            skip_special_tokens=False,
        ).strip()
        text = self._extract_assistant_text(decoded)
        if text:
            return text

        if self.tokenizer is not None:
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return decoded

    @staticmethod
    def _load_image(image_path: str | None, dummy_image_size: int) -> Image.Image:
        """입력 이미지가 있으면 로드하고, 없으면 더미 이미지를 사용한다."""
        if image_path:
            path = Path(image_path)
            if path.exists():
                return Image.open(path).convert("RGB")
        return Image.new(
            "RGB",
            (dummy_image_size, dummy_image_size),
            color=(255, 255, 255),
        )

    @staticmethod
    def _normalize_emotion(raw: dict | None) -> dict[str, int]:
        """임의 JSON을 감정 one-hot으로 정규화한다."""
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
        """감정 JSON 분류를 VLM 자체로 시도한다."""
        text = f"{narration}\n{dialogue_ko}".strip()
        if not text:
            return {"neutral": 1, "sad": 0, "happy": 0, "angry": 0}

        raw = self.generate_from_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 감정 분류기다. 입력된 사야의 서술/대사를 보고 "
                        "neutral,sad,happy,angry 4개 키만 가진 one-hot JSON만 출력하라."
                    ),
                },
                {
                    "role": "user",
                    "content": f"사야 출력:\n{text}",
                },
            ],
            gen_config=VLMGenerationConfig(
                max_length=1024,
                max_new_tokens=96,
                do_sample=False,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.0,
                dummy_image_size=self.default_gen.dummy_image_size,
            ),
        )
        match = re.search(r"\{[\s\S]*?\}", raw)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return self._normalize_emotion(parsed)
        except Exception:
            return None
