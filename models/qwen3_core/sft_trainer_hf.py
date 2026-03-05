"""
Hugging Face Trainer 기반 QLoRA SFT 스크립트.

목적:
- qwen3_core의 대화형 JSONL 데이터셋을 QLoRA 방식으로 학습한다.

기본 데이터 포맷:
1) {"messages": [{"role": "...", "content": "..."}, ...]}
2) {"conversations": [{"from": "...", "value": "..."}, ...]}

예시:
uv run models/qwen3_core/sft_trainer_hf.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b \
  --dataset_name junidude14/korean_roleplay_dataset_for_chat_game_2 \
  --dataset_split train \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_lora1 \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --assistant_only_loss \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5

while kill -0 12345 2>/dev/null; do sleep 30; done
uv run models/qwen3_core/sft_trainer_hf.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b \
  --dataset_name junidude14/korean_roleplay_dataset_for_chat_game_2 \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_lora2 \
  --init_adapter_path models/qwen3_core/model_assets/qwen3_4b_lora1/lora_adapter \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --assistant_only_loss \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def build_instruction_prompt(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()

    if user_input:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{user_input}\n\n"
            "### Response:\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def _normalize_role(role: Any) -> Optional[str]:
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


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n".join(text_parts).strip()

    return ""


def extract_messages(
    example: Dict[str, Any],
    messages_key: str,
    conversations_key: str,
) -> Optional[List[Dict[str, str]]]:
    msgs = example.get(messages_key)
    if msgs is None:
        msgs = example.get(conversations_key)

    if not isinstance(msgs, list) or not msgs:
        return None

    out: List[Dict[str, str]] = []
    for item in msgs:
        if not isinstance(item, dict):
            continue

        role = _normalize_role(item.get("role", item.get("from", item.get("speaker"))))
        if role is None:
            continue

        content = _stringify_content(item.get("content", item.get("value", item.get("text", ""))))
        if not content:
            continue

        out.append({"role": role, "content": content})

    return out or None


def is_valid_conversation(messages: Optional[List[Dict[str, str]]]) -> bool:
    if not messages:
        return False
    return any(msg["role"] == "assistant" for msg in messages)


def extract_instruction_example(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    instruction = example.get("instruction")
    user_input = example.get("input", "")
    output = example.get("output")

    if not isinstance(instruction, str) or not instruction.strip():
        return None
    if not isinstance(output, str) or not output.strip():
        return None
    if not isinstance(user_input, str):
        user_input = ""

    return {
        "instruction": instruction.strip(),
        "input": user_input.strip(),
        "output": output.strip(),
    }


def tokenize_messages(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    max_length: int,
    assistant_only_loss: bool,
) -> Dict[str, List[int]]:
    assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
    if not assistant_indices:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    first_assistant_idx = assistant_indices[0]
    prompt_messages = messages[:first_assistant_idx]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    if not full_ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_ids = list(full_ids)
    attention_mask = [1] * len(input_ids)

    if assistant_only_loss:
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
    else:
        labels = list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_instruction_example(
    instruction_example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int,
    assistant_only_loss: bool,
) -> Dict[str, List[int]]:
    prompt_text = build_instruction_prompt(
        instruction_example["instruction"],
        instruction_example["input"],
    )
    full_text = prompt_text + instruction_example["output"]
    if tokenizer.eos_token:
        full_text += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    if not full_ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_ids = list(full_ids)
    attention_mask = [1] * len(input_ids)

    if assistant_only_loss:
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
    else:
        labels = list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    messages_key: str,
    conversations_key: str,
    assistant_only_loss: bool,
) -> Dict[str, Any]:
    messages = extract_messages(
        example=example,
        messages_key=messages_key,
        conversations_key=conversations_key,
    )
    if not is_valid_conversation(messages):
        instruction_example = extract_instruction_example(example)
        if instruction_example is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return tokenize_instruction_example(
            instruction_example=instruction_example,
            tokenizer=tokenizer,
            max_length=max_length,
            assistant_only_loss=assistant_only_loss,
        )

    return tokenize_messages(
        messages=messages,
        tokenizer=tokenizer,
        max_length=max_length,
        assistant_only_loss=assistant_only_loss,
    )


def has_tokens(example: Dict[str, Any]) -> bool:
    return bool(example["input_ids"])


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    ratio = 100 * trainable / total if total else 0.0
    print(
        f"trainable params: {trainable:,} | all params: {total:,} | trainable%: {ratio:.4f}",
        flush=True,
    )


@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="models/qwen3_core/model_assets/qwen3-4b-instruct",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--messages_key", type=str, default="messages")
    parser.add_argument("--conversations_key", type=str, default="conversations")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_split", type=float, default=0.0)
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--assistant_only_loss", dest="assistant_only_loss", action="store_true")
    parser.add_argument("--no-assistant_only_loss", dest="assistant_only_loss", action="store_false")
    parser.set_defaults(assistant_only_loss=True)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")

    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if not args.data_path and not args.dataset_name:
        raise ValueError("Provide either --data_path or --dataset_name.")
    if args.data_path and args.dataset_name:
        raise ValueError("Use only one of --data_path or --dataset_name.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.bfloat16
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
    )

    if args.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.init_adapter_path,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            base_model,
            build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout),
        )

    print_trainable_params(model)

    if args.dataset_name:
        raw_ds = load_dataset(args.dataset_name, split=args.dataset_split)
    else:
        data_path = Path(args.data_path)
        raw_ds = load_dataset("json", data_files=str(data_path))["train"]

    if args.eval_split > 0:
        split = raw_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = raw_ds
        eval_ds = None

    preprocess = lambda ex: preprocess_example(
        example=ex,
        tokenizer=tokenizer,
        max_length=args.max_length,
        messages_key=args.messages_key,
        conversations_key=args.conversations_key,
        assistant_only_loss=args.assistant_only_loss,
    )

    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train split",
    ).filter(has_tokens, desc="Filtering empty train rows")

    if len(train_ds) == 0:
        raise ValueError("No valid training samples after preprocessing.")

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            preprocess,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval split",
        ).filter(has_tokens, desc="Filtering empty eval rows")
        if len(eval_ds) == 0:
            eval_ds = None

    evaluation_strategy = args.evaluation_strategy if eval_ds is not None else "no"
    eval_steps = args.eval_steps if args.eval_steps > 0 else args.save_steps

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter and tokenizer to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
