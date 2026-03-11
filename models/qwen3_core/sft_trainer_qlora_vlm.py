"""Qwen 계열 VLM QLoRA SFT 학습 스크립트.

설계 기준:
1. Hugging Face Qwen3.5-4B 모델 카드의 `AutoProcessor` + 이미지-텍스트 생성 로더 패턴을 따른다.
2. 기존 qwen3_core 학습 스크립트의 CLI 습관을 유지한다.
3. 텍스트 전용 데이터와 이미지 포함 데이터를 모두 처리한다.
4. assistant-only loss를 지원한다.

예시:
uv run models/qwen3_core/sft_trainer_qlora_vlm.py \
  --model_name models/qwen3_core/model_assets/qwen3.5-4b \
  --data_path /mnt/d/rp_data/rewrite/singleturn_rewrite.jsonl \
  --output_dir models/qwen3_core/model_assets/qwen3.5-4b_stage1 \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --save_steps 25 \
  --save_total_limit 6 \
  --eval_split 0.02 \
  --eval_strategy steps \
  --eval_steps 25 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --prediction_loss_only \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --assistant_only_loss \
  --trust_remote_code
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    p = argparse.ArgumentParser(description="Qwen 계열 VLM QLoRA SFT trainer")
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--init_adapter_path", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--messages_key", type=str, default="messages")
    p.add_argument("--conversations_key", type=str, default="conversations")
    p.add_argument("--image_keys", type=str, default="image,image_path,images")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--dummy_image_size", type=int, default=224)
    p.add_argument("--text_only", action="store_true")
    p.add_argument("--assistant_only_loss", action="store_true")
    p.add_argument("--no_assistant_only_loss", dest="assistant_only_loss", action="store_false")

    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--eval_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_split", type=float, default=0.0)
    p.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--prediction_loss_only", action="store_true")
    p.add_argument("--load_best_model_at_end", action="store_true")
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
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
    p.set_defaults(assistant_only_loss=True)
    return p.parse_args()


def _normalize_role(role: Any) -> str | None:
    """원본 role 표기를 표준 role로 정규화한다."""
    if not isinstance(role, str):
        return None
    mapping = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
    }
    return mapping.get(role.strip().lower())


def _extract_messages(example: dict[str, Any], messages_key: str, conversations_key: str) -> list[dict[str, str]]:
    """데이터 샘플에서 표준 메시지 배열을 뽑는다."""
    raw = example.get(messages_key)
    if raw is None:
        raw = example.get(conversations_key)
    if not isinstance(raw, list):
        return []

    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _normalize_role(item.get("role", item.get("from", item.get("speaker"))))
        if role is None:
            continue
        content = item.get("content", item.get("value", item.get("text", "")))
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _extract_image_paths(example: dict[str, Any], image_keys: list[str], data_root: Path | None) -> list[str]:
    """샘플에서 이미지 경로 목록을 뽑아 절대 경로로 정규화한다."""
    paths: list[str] = []
    for key in image_keys:
        value = example.get(key)
        if value is None:
            continue
        raw_items: list[Any]
        if isinstance(value, list):
            raw_items = value
        else:
            raw_items = [value]
        for item in raw_items:
            if not isinstance(item, str):
                continue
            p = Path(item)
            if not p.is_absolute() and data_root is not None:
                p = data_root / p
            paths.append(str(p))
        if paths:
            break
    return paths


def _make_text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def build_example(
    example: dict[str, Any],
    messages_key: str,
    conversations_key: str,
    image_keys: list[str],
    data_root: Path | None,
    text_only: bool,
) -> dict[str, Any]:
    """원본 샘플을 VLM 학습용 예제로 변환한다."""
    messages = _extract_messages(example, messages_key, conversations_key)
    if not messages:
        return {"messages": [], "image_paths": []}

    image_paths = [] if text_only else _extract_image_paths(example, image_keys, data_root)
    out: list[dict[str, Any]] = []
    image_inserted = False
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            blocks: list[dict[str, str]] = []
            if image_paths and not image_inserted:
                blocks.append({"type": "image"})
                image_inserted = True
            blocks.append(_make_text_block(content))
            out.append({"role": role, "content": blocks})
        else:
            out.append({"role": role, "content": [_make_text_block(content)]})
    return {"messages": out, "image_paths": image_paths}


def is_valid_example(example: dict[str, Any]) -> bool:
    """학습 가능한 예제인지 검사한다."""
    messages = example.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return False
    has_user = any(m.get("role") == "user" for m in messages if isinstance(m, dict))
    has_assistant = any(m.get("role") == "assistant" for m in messages if isinstance(m, dict))
    return has_user and has_assistant


def _load_image(path: str, dummy_image_size: int) -> Image.Image:
    """이미지를 로드하고 실패 시 더미 이미지로 대체한다."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (dummy_image_size, dummy_image_size), color=(255, 255, 255))


def _inject_image_block(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """첫 user 메시지에 image block을 강제 주입한 복사본을 만든다."""
    out: list[dict[str, Any]] = []
    inserted = False
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", [])
        if role == "user" and not inserted and isinstance(content, list):
            has_image = any(isinstance(block, dict) and block.get("type") == "image" for block in content)
            if not has_image:
                content = [{"type": "image"}, *content]
            inserted = True
        out.append({"role": role, "content": content})
    return out


class QwenVLMCollator:
    """Qwen 계열 VLM 학습용 배치 콜레이터."""

    def __init__(self, processor: Any, max_length: int, dummy_image_size: int, assistant_only_loss: bool) -> None:
        self.processor = processor
        self.max_length = max_length
        self.dummy_image_size = dummy_image_size
        self.assistant_only_loss = assistant_only_loss
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _build_inputs(
        self,
        messages_batch: list[list[dict[str, Any]]],
        image_batch: list[Image.Image | None],
        add_generation_prompt: bool,
    ) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        has_any_image = any(img is not None for img in image_batch)
        normalized_messages: list[list[dict[str, Any]]] = []
        normalized_images: list[Image.Image] = []

        for messages, image in zip(messages_batch, image_batch, strict=True):
            cur_messages = messages
            cur_image = image
            if has_any_image and cur_image is None:
                cur_image = Image.new("RGB", (self.dummy_image_size, self.dummy_image_size), color=(255, 255, 255))
                cur_messages = _inject_image_block(cur_messages)
            normalized_messages.append(cur_messages)
            if cur_image is not None:
                normalized_images.append(cur_image)

        for messages in normalized_messages:
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )

        kwargs: dict[str, Any] = dict(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if has_any_image:
            kwargs["images"] = [img for img in normalized_images]
        return self.processor(**kwargs)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """features를 모델 입력 배치로 변환한다."""
        messages_batch = [ex["messages"] for ex in features]
        image_batch = [
            _load_image(ex["image_paths"][0], self.dummy_image_size) if ex.get("image_paths") else None
            for ex in features
        ]
        batch = self._build_inputs(messages_batch, image_batch, add_generation_prompt=False)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_id = tokenizer.pad_token_id if tokenizer is not None else None
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        if self.assistant_only_loss:
            for row_idx, (messages, image) in enumerate(zip(messages_batch, image_batch, strict=True)):
                last_assistant_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "assistant":
                        last_assistant_idx = i
                        break
                if last_assistant_idx is None:
                    continue
                prefix_messages = messages[:last_assistant_idx]
                if not prefix_messages:
                    continue
                prefix_batch = self._build_inputs([prefix_messages], [image], add_generation_prompt=True)
                prefix_len = int(prefix_batch["input_ids"].shape[1])
                if prefix_len > 0:
                    labels[row_idx, :prefix_len] = -100

        batch["labels"] = labels
        return batch


def load_dataset_for_train(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    """JSONL 데이터셋을 로드하고 전처리한다."""
    raw = load_dataset("json", data_files=args.data_path)["train"]
    if args.eval_split > 0:
        split = raw.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = raw
        eval_ds = None

    image_keys = [x.strip() for x in args.image_keys.split(",") if x.strip()]
    data_root = Path(args.data_root).resolve() if args.data_root else None

    fn = lambda ex: build_example(
        ex,
        args.messages_key,
        args.conversations_key,
        image_keys,
        data_root,
        args.text_only,
    )
    train_ds = train_ds.map(fn, remove_columns=train_ds.column_names)
    train_ds = train_ds.filter(is_valid_example)

    if eval_ds is not None:
        eval_ds = eval_ds.map(fn, remove_columns=eval_ds.column_names)
        eval_ds = eval_ds.filter(is_valid_example)

    if len(train_ds) == 0:
        raise ValueError("No valid training samples after preprocessing.")
    return train_ds, eval_ds


def build_lora(args: argparse.Namespace) -> LoraConfig:
    """LoRA 설정을 생성한다."""
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


def main() -> None:
    """학습 엔트리 포인트."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, build_lora(args))

    train_ds, eval_ds = load_dataset_for_train(args)
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    eval_steps = args.eval_steps if args.eval_steps > 0 else args.save_steps
    if args.load_best_model_at_end and eval_strategy == "no":
        raise ValueError("--load_best_model_at_end requires eval enabled.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        prediction_loss_only=args.prediction_loss_only,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        bf16=args.bf16 or not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=QwenVLMCollator(
            processor=processor,
            max_length=args.max_length,
            dummy_image_size=args.dummy_image_size,
            assistant_only_loss=args.assistant_only_loss,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(output_dir / "lora_adapter")
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir / "tokenizer")
    processor.save_pretrained(output_dir / "processor")

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2, ensure_ascii=False)

    print(f"[DONE] saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
