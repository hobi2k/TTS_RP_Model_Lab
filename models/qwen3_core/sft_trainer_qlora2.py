"""VLM QLoRA SFT 학습 스크립트.

사용 예시:
uv run models/qwen3_core/sft_trainer_qlora2.py \
  --model_name models/qwen3_core/model_assets/kanana_3b \
  --data_path /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --output_dir models/qwen3_core/model_assets/kanana_3b_stage1 \
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
  --text_only \
  --trust_remote_code

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def _normalize_role(role: Any) -> str | None:
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


def _extract_messages(example: dict[str, Any], messages_key: str, conversations_key: str) -> list[dict[str, Any]] | None:
    msgs = example.get(messages_key)
    if msgs is None:
        msgs = example.get(conversations_key)
    if not isinstance(msgs, list) or not msgs:
        return None

    out: list[dict[str, Any]] = []
    for item in msgs:
        if not isinstance(item, dict):
            continue
        role = _normalize_role(item.get("role", item.get("from", item.get("speaker"))))
        if role is None:
            continue
        content = item.get("content", item.get("value", item.get("text", "")))
        if isinstance(content, str):
            content = content.strip()
            if not content:
                continue
        elif isinstance(content, list):
            if not content:
                continue
        else:
            continue
        out.append({"role": role, "content": content})

    return out or None


def _normalize_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip().lower()
        if block_type in {"text"}:
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                out.append({"type": "text", "text": text.strip()})
        elif block_type in {"image", "image_url"}:
            # Training uses a separate `images=` argument. Placeholder keeps chat template shape.
            out.append({"type": "image"})
    return out


def _to_multimodal_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized: list[dict[str, Any]] = []
    has_image = False
    first_user_idx = None

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "user" and first_user_idx is None:
            first_user_idx = idx
        blocks = _normalize_content_blocks(msg.get("content"))
        if not blocks:
            continue
        if any(b.get("type") == "image" for b in blocks):
            has_image = True
        normalized.append({"role": role, "content": blocks})

    if not normalized:
        return None

    if not has_image:
        inject_idx = None
        for idx, msg in enumerate(normalized):
            if msg.get("role") == "user":
                inject_idx = idx
                break
        if inject_idx is None:
            return None
        normalized[inject_idx]["content"] = [{"type": "image"}, *normalized[inject_idx]["content"]]

    return normalized


def _ensure_image_placeholder(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    out = []
    inserted = False
    for msg in messages:
        role = msg.get("role")
        content = list(msg.get("content", []))
        if role == "user" and not inserted:
            has_image = any(b.get("type") == "image" for b in content if isinstance(b, dict))
            if not has_image:
                content = [{"type": "image"}, *content]
            inserted = True
        out.append({"role": role, "content": content})
    return out if inserted else None


def _resolve_image_path(
    example: dict[str, Any],
    *,
    image_keys: list[str],
    data_root: Path | None,
    default_base: Path,
) -> Path | None:
    image_value = None
    for key in image_keys:
        if key in example and example[key]:
            image_value = example[key]
            break

    if not isinstance(image_value, str):
        return None

    path = Path(image_value)
    if not path.is_absolute():
        if data_root is not None:
            path = data_root / path
        else:
            path = default_base / path
    return path if path.exists() else None


def build_training_example(
    example: dict[str, Any],
    *,
    messages_key: str,
    conversations_key: str,
    image_keys: list[str],
    data_root: Path | None,
    default_base: Path,
    text_only: bool,
) -> dict[str, Any]:
    messages = _extract_messages(example, messages_key=messages_key, conversations_key=conversations_key)
    if not messages:
        return {"messages": [], "image_path": ""}

    mm_messages = _to_multimodal_messages(messages)
    if not mm_messages:
        return {"messages": [], "image_path": ""}

    if text_only:
        mm_messages = _ensure_image_placeholder(mm_messages)
        if not mm_messages:
            return {"messages": [], "image_path": ""}
        return {"messages": mm_messages, "image_path": ""}

    image_path = _resolve_image_path(example, image_keys=image_keys, data_root=data_root, default_base=default_base)
    if image_path is None:
        return {"messages": [], "image_path": ""}

    return {"messages": mm_messages, "image_path": str(image_path)}


def is_valid_training_example(example: dict[str, Any], text_only: bool) -> bool:
    if not example.get("messages"):
        return False
    return True if text_only else bool(example.get("image_path"))


class VisionSFTCollator:
    def __init__(
        self,
        processor: Any,
        max_length: int,
        text_only: bool,
        dummy_image_size: int,
        assistant_only_loss: bool,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.text_only = text_only
        self.dummy_image_size = dummy_image_size
        self.assistant_only_loss = assistant_only_loss

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []

        for ex in features:
            messages = ex["messages"]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(prompt)
            if self.text_only:
                images.append(Image.new("RGB", (self.dummy_image_size, self.dummy_image_size), color=(255, 255, 255)))
            else:
                image_path = Path(ex["image_path"])
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))

        batch = self.processor(text=texts, images=images, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_id = tokenizer.pad_token_id if tokenizer is not None else None
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        if self.assistant_only_loss:
            for row_idx, ex in enumerate(features):
                msgs = ex.get("messages", [])
                last_assistant_idx = None
                for i in range(len(msgs) - 1, -1, -1):
                    if msgs[i].get("role") == "assistant":
                        last_assistant_idx = i
                        break
                if last_assistant_idx is None:
                    continue

                prefix_messages = msgs[:last_assistant_idx]
                if not prefix_messages:
                    continue

                prefix_text = self.processor.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # prefix 길이를 계산해 마지막 assistant 이전 토큰을 라벨에서 제외한다.
                prefix_batch = self.processor(
                    text=[prefix_text],
                    images=[images[row_idx]],
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                prefix_len = int(prefix_batch["input_ids"].shape[1])
                if prefix_len > 0:
                    labels[row_idx, :prefix_len] = -100

        batch["labels"] = labels
        return batch


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    task_type = TaskType.CAUSAL_LM if args.lora_task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules must contain at least one module name.")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-Language QLoRA SFT 학습기")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--messages_key", type=str, default="messages")
    parser.add_argument("--conversations_key", type=str, default="conversations")
    parser.add_argument("--image_keys", type=str, default="image,image_path,image_file,img,file_path")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--dummy_image_size", type=int, default=224)
    parser.add_argument("--assistant_only_loss", dest="assistant_only_loss", action="store_true")
    parser.add_argument("--no-assistant_only_loss", dest="assistant_only_loss", action="store_false")
    parser.add_argument("--max_length", type=int, default=4096)

    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_split", type=float, default=0.0)
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--prediction_loss_only", action="store_true")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--attn_implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM", choices=["CAUSAL_LM", "SEQ_2_SEQ_LM"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(assistant_only_loss=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.bfloat16
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

    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    if args.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.init_adapter_path,
            is_trainable=True,
        )
    else:
        model = get_peft_model(base_model, build_lora_config(args))

    raw = load_dataset("json", data_files=args.data_path)["train"]
    if args.eval_split > 0:
        split = raw.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = raw, None

    image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    data_root = Path(args.data_root).resolve() if args.data_root else None
    default_base = Path(args.data_path).resolve().parent

    train_ds = train_ds.map(
        lambda ex: build_training_example(
            ex,
            messages_key=args.messages_key,
            conversations_key=args.conversations_key,
            image_keys=image_keys,
            data_root=data_root,
            default_base=default_base,
            text_only=args.text_only,
        ),
        remove_columns=train_ds.column_names,
    )
    train_len_before = len(train_ds)
    train_ds = train_ds.filter(lambda ex: is_valid_training_example(ex, text_only=args.text_only))
    dropped_train = train_len_before - len(train_ds)
    if dropped_train > 0:
        print(f"[WARN] dropped invalid train samples: {dropped_train}")

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda ex: build_training_example(
                ex,
                messages_key=args.messages_key,
                conversations_key=args.conversations_key,
                image_keys=image_keys,
                data_root=data_root,
                default_base=default_base,
                text_only=args.text_only,
            ),
            remove_columns=eval_ds.column_names,
        )
        eval_len_before = len(eval_ds)
        eval_ds = eval_ds.filter(lambda ex: is_valid_training_example(ex, text_only=args.text_only))
        dropped_eval = eval_len_before - len(eval_ds)
        if dropped_eval > 0:
            print(f"[WARN] dropped invalid eval samples: {dropped_eval}")

    if len(train_ds) == 0:
        if args.text_only:
            raise ValueError("No valid text samples after preprocessing. Check messages/conversations format.")
        raise ValueError("No valid VLM samples after preprocessing. Check image path keys and messages format.")

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
        bf16=args.bf16,
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
        data_collator=VisionSFTCollator(
            processor=processor,
            max_length=args.max_length,
            text_only=args.text_only,
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

    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()
