"""Qwen 계열 VLM RP GRPO 학습 스크립트.

기준:
- Hugging Face Qwen3.5-4B 카드의 VLM 로더 패턴을 따른다.
- qwen3_core의 기존 GRPO 보상 함수를 재사용한다.
- 텍스트 RP 데이터로도 VLM 경로를 실험할 수 있게 더미 이미지를 기본 입력으로 사용한다.
"""

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, ProcessorMixin
from trl import GRPOConfig, GRPOTrainer

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel


PLAYER_NAMES = ("카즈키", "하야토", "소마", "유저", "플레이어")
ROLE_MARKERS = ("SYSTEM:", "USER:", "ASSISTANT:", "role:", "<|im_start|", "<|assistant|")


def _as_text(x: Any) -> str:
    """보상 함수 입력을 문자열로 정규화한다."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if isinstance(x.get("content"), str):
            return x["content"]
        return str(x)
    if isinstance(x, list):
        parts: List[str] = []
        for item in x:
            t = _as_text(item)
            if t:
                parts.append(t)
        return "\n".join(parts)
    return str(x)


def _normalize_role(role: Any) -> Optional[str]:
    """role 표기를 표준 role로 정규화한다."""
    if not isinstance(role, str):
        return None
    role_map = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
    }
    return role_map.get(role.strip().lower())


def _prompt_to_role_text(prompt: Any) -> str:
    """프롬프트를 ROLE prefix 평탄 텍스트로 변환한다."""
    if isinstance(prompt, list):
        out: List[str] = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            role = _as_text(m.get("role")).strip().upper()
            content = _as_text(m.get("content")).strip()
            if role and content:
                out.append(f"{role}: {content}")
        return "\n".join(out).strip()
    return _as_text(prompt)


def _extract_last_user(prompt: Any) -> str:
    """평탄 prompt에서 마지막 USER 블록을 추출한다."""
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if not isinstance(msg, dict):
                continue
            role = _normalize_role(msg.get("role", msg.get("from", msg.get("speaker"))))
            if role == "user":
                text = _as_text(msg.get("content", msg.get("value", msg.get("text", "")))).strip()
                if text:
                    return text
        return ""
    prompt_text = _as_text(prompt)
    matches = re.findall(r"USER:\s*(.*?)(?=\n(?:SYSTEM|USER|ASSISTANT):|\Z)", prompt_text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def _normalize_completion_for_scoring(raw: str) -> tuple[str, str, bool]:
    """completion을 채점용 2블록으로 정규화한다."""
    txt = (raw or "").strip()
    if not txt:
        return "", "", False
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "", "", False
    narration = lines[0]
    quote = ""
    for ln in lines[1:]:
        if re.fullmatch(r'"[^"\n]{2,300}"', ln):
            quote = ln
            break
    if not quote:
        return "", "", False
    return narration, quote, len(lines) == 2


def _rough_token_len(text: str) -> int:
    return len(re.findall(r"[가-힣A-Za-z0-9]+|[^\s]", text or ""))


def _ngram_overlap_ratio(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    ta = re.findall(r"[가-힣A-Za-z0-9]+", a.lower())
    tb = re.findall(r"[가-힣A-Za-z0-9]+", b.lower())
    if len(ta) < n or len(tb) < n:
        return 0.0
    a_ngrams = {" ".join(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    b_ngrams = {" ".join(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    return len(a_ngrams & b_ngrams) / float(len(a_ngrams) or 1)


def reward_format(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """형식 보상."""
    scores: List[float] = []
    for comp in completions:
        raw = _as_text(comp)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        _, quote, valid = _normalize_completion_for_scoring(raw)
        has_role_marker = any(m in raw.upper() for m in ROLE_MARKERS)
        score = 0.0
        if lines and not lines[0].startswith('"'):
            score += 0.30
        if quote:
            score += 0.40
        if valid and not has_role_marker:
            score += 0.30
        scores.append(max(0.0, min(1.0, score)))
    return scores


def reward_role_split(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """assistant가 player 화법을 대필하지 않도록 보상한다."""
    scores: List[float] = []
    player_re = "|".join(PLAYER_NAMES)
    for comp in completions:
        raw = _as_text(comp)
        penalty = 0.0
        if re.search(rf"(?:^|\n)\s*(?:{player_re})\s*[:：]", raw):
            penalty += 0.6
        if re.search(rf"(?:^|\n)\s*(?:{player_re}).{{0,16}}(?:말했|말한다|묻는다|대답했)", raw):
            penalty += 0.5
        scores.append(max(0.0, 1.0 - penalty))
    return scores


def reward_grounded(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """직전 user 발화와의 표면 정합 보상."""
    scores: List[float] = []
    for prompt, comp in zip(prompts, completions):
        user_text = _extract_last_user(prompt)
        raw = _as_text(comp)
        narr, quote, valid = _normalize_completion_for_scoring(raw)
        if not valid or not user_text:
            scores.append(0.0)
            continue
        overlap = _ngram_overlap_ratio(user_text, f"{narr} {quote}", n=2)
        if overlap >= 0.25:
            scores.append(1.0)
        elif overlap >= 0.12:
            scores.append(0.7)
        elif overlap >= 0.05:
            scores.append(0.4)
        else:
            scores.append(0.1)
    return scores


def reward_length(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
    """길이 안정성 보상."""
    scores: List[float] = []
    for comp in completions:
        n = _rough_token_len(_as_text(comp))
        if 35 <= n <= 140:
            scores.append(1.0)
        elif 20 <= n < 35 or 140 < n <= 200:
            scores.append(0.7)
        elif 200 < n <= 260:
            scores.append(0.35)
        else:
            scores.append(0.1)
    return scores


class QwenVLMGRPOProcessor(ProcessorMixin):
    """TRL GRPO에서 Qwen 계열 VLM을 안정적으로 호출하기 위한 어댑터."""

    attributes = ["tokenizer"]

    def __init__(self, base_processor: Any, max_length: int, dummy_image_size: int = 224) -> None:
        self.base_processor = base_processor
        self.tokenizer = getattr(base_processor, "tokenizer", None)
        self.chat_template = getattr(base_processor, "chat_template", None)
        if self.chat_template is None and self.tokenizer is not None:
            self.chat_template = getattr(self.tokenizer, "chat_template", None)
        self.max_length = max_length
        self.dummy_image_size = dummy_image_size
        if self.tokenizer is None:
            raise ValueError("processor에 tokenizer가 없습니다.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _dummy_image(self) -> Image.Image:
        return Image.new("RGB", (self.dummy_image_size, self.dummy_image_size), color=(255, 255, 255))

    def _to_conv(self, text: str, has_image: bool = True) -> list[dict[str, Any]]:
        text = (text or "").strip()
        if has_image:
            content: list[dict[str, str]] = [{"type": "image"}, {"type": "text", "text": text}]
        else:
            content = [{"type": "text", "text": text}]
        return [{"role": "user", "content": content}]

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        texts = kwargs.get("text")
        images = kwargs.get("images")
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", self.max_length)
        if texts is None:
            return self.base_processor(*args, **kwargs)
        if isinstance(texts, str):
            texts = [texts]

        prompts: list[str] = []
        image_batch: list[Image.Image] = []
        has_any_image = True
        if images is None:
            images = [None] * len(texts)
        elif not isinstance(images, list):
            images = [images]

        while len(images) < len(texts):
            images.append(None)

        for text, image in zip(texts, images, strict=True):
            pil_image = image if isinstance(image, Image.Image) else self._dummy_image()
            prompt = self.base_processor.apply_chat_template(
                self._to_conv(str(text), has_image=True),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            image_batch.append(pil_image)

        return self.base_processor(
            text=prompts,
            images=image_batch if has_any_image else None,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        """TRL이 호출하는 chat template 경로를 호환시킨다."""
        tokenize = kwargs.get("tokenize", False)
        if not tokenize:
            if isinstance(conversation, list) and conversation and isinstance(conversation[0], dict):
                return _prompt_to_role_text(conversation)
            return _as_text(conversation)

        conv_batch = conversation
        if isinstance(conv_batch, list) and conv_batch and isinstance(conv_batch[0], dict):
            conv_batch = [conv_batch]
        if not isinstance(conv_batch, list):
            conv_batch = [[{"role": "user", "content": str(conv_batch)}]]
        texts = [_prompt_to_role_text(conv) for conv in conv_batch]
        return self(text=texts, images=None, max_length=kwargs.get("max_length", self.max_length))

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> Any:
        return self.base_processor.save_pretrained(save_directory, **kwargs)

    @property
    def pad_token(self) -> str | None:
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self) -> int | None:
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self) -> str | None:
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id


def load_grpo_dataset(path: str) -> Dataset:
    """GRPO 데이터셋을 로드하고 prompt/reference/image 컬럼으로 정규화한다."""
    ds = load_dataset("json", data_files=path)["train"]

    def _trim_to_last_user_turn(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not messages:
            return []
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user" and _as_text(messages[i].get("content")).strip():
                last_user_idx = i
                break
        if last_user_idx < 0:
            return []
        return [m for m in messages[: last_user_idx + 1] if _as_text(m.get("content", "")).strip()]

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt: list[dict[str, str]] = []
        raw_prompt = ex.get("prompt")
        if isinstance(raw_prompt, list):
            for m in raw_prompt:
                if not isinstance(m, dict):
                    continue
                role = _normalize_role(m.get("role", m.get("from", m.get("speaker"))))
                if role is None:
                    continue
                content = _as_text(m.get("content", m.get("value", m.get("text", "")))).strip()
                if content:
                    prompt.append({"role": role, "content": content})

        if not prompt and isinstance(ex.get("messages"), list):
            for m in ex["messages"]:
                if not isinstance(m, dict):
                    continue
                role = _normalize_role(m.get("role", m.get("from", m.get("speaker"))))
                if role is None:
                    continue
                content = _as_text(m.get("content", m.get("value", m.get("text", "")))).strip()
                if content:
                    prompt.append({"role": role, "content": content})

        if not prompt:
            plain = _as_text(ex.get("prompt", "")).strip()
            if plain:
                prompt = [{"role": "user", "content": plain}]

        prompt = _trim_to_last_user_turn(prompt)
        reference = _as_text(ex.get("reference", ex.get("output", ""))).strip()
        return {"prompt": prompt, "reference": reference, "image": "__dummy__"}

    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(
        lambda ex: (
            isinstance(ex.get("prompt"), list)
            and len(ex["prompt"]) > 0
            and ex["prompt"][-1].get("role") == "user"
            and bool(_as_text(ex["prompt"][-1].get("content", "")).strip())
        )
    )
    if len(ds) == 0:
        raise ValueError(f"No valid GRPO samples from: {path}")
    return ds


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """LoRA 설정을 만든다."""
    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("--lora_target_modules is empty.")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="Qwen 계열 VLM RP GRPO trainer")
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--eval_data", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--save_total_limit", type=int, default=6)
    p.add_argument("--eval_strategy", choices=["no", "steps", "epoch"], default="steps")
    p.add_argument("--eval_steps", type=int, default=25)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=200)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dummy_image_size", type=int, default=224)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return p.parse_args()


def main() -> None:
    """학습 엔트리 포인트."""
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 or not args.fp16 else torch.float16
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    wrapped_processor = QwenVLMGRPOProcessor(
        base_processor=processor,
        max_length=args.max_prompt_length,
        dummy_image_size=args.dummy_image_size,
    )

    model = AutoVLMModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    peft_config = build_lora_config(args) if args.use_lora else None
    train_ds = load_grpo_dataset(args.train_data)
    eval_ds = load_grpo_dataset(args.eval_data) if args.eval_data else None
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16 or not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
    )

    valid_grpo = inspect.signature(GRPOConfig.__init__).parameters
    grpo_config = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in valid_grpo and v is not None})

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=[
            reward_format,
            reward_role_split,
            reward_grounded,
            reward_length,
        ],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=wrapped_processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    wrapped_processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
