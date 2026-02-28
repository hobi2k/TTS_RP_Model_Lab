"""
sft_trainer_translator.py

목적:
- Qwen/Qwen3-1.7B-Base 모델을 대상으로
- 번역 SFT 데이터(instruction/input/output JSONL)를 사용해
- LoRA 방식으로 미세조정(SFT)한다.

핵심 설계:
1) "번역기"로만 동작하도록, 데이터는 instruction+input -> output 형태로 학습한다.
2) LoRA를 사용해 VRAM 부담을 줄이고, 캐릭터/RP 모델과 역할을 분리한다.
3) __file__ 기준 경로를 사용해 uv run / 작업 디렉터리 변화에 안전하게 만든다.

uv run models/qwen3_core/sft_trainer_translator.py \
  --bf16 \
  --gradient_checkpointing \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 경로 / 기본값 설정

# 이 파일 위치: models/qwen3_core/sft_trainer_translator.py
QWEN3_CORE_DIR = Path(__file__).resolve().parent          # .../models/qwen3_core
MODEL_ASSETS_DIR = QWEN3_CORE_DIR / "model_assets"

# 번역용 Base 모델 (로컬 고정)
DEFAULT_BASE_MODEL_DIR = MODEL_ASSETS_DIR / "qwen3-1.7b-base"

# 학습 데이터 (WSL 기준 절대경로 권장)
DATA_DIR = Path("/mnt/d/rp_data/singleturn")
DEFAULT_DATA_PATH = DATA_DIR / "ko-ja_translation_sft.jsonl"

# LoRA 출력 디렉터리
DEFAULT_OUTPUT_DIR = MODEL_ASSETS_DIR / "qwen3_1.7_ko2ja_lora"


# 프롬프트 구성

def build_prompt(instruction: str, user_input: str) -> str:
    """
    번역 SFT용 프롬프트를 구성한다.
    (Chat template 사용 x)

    구조:
    ### Instruction:
    ...
    ### Input:
    ...
    ### Output:
    """
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Output:\n"
    )


def build_full_text(example: Dict[str, str], eos_token: str) -> str:
    """
    학습용 전체 텍스트 생성:
    [Instruction + Input + Output + EOS]
    """
    prompt = build_prompt(example.get("instruction", ""), example.get("input", ""))
    answer = (example.get("output", "") or "").strip()
    return prompt + answer + eos_token


# Tokenize

def tokenize_function(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:

    eos = tokenizer.eos_token or ""

    texts: List[str] = []
    output_start_positions: List[int] = []

    for inst, inp, out in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        prompt = build_prompt(inst, inp)
        full_text = prompt + (out or "").strip() + eos
        texts.append(full_text)

        # 🔥 Output 시작 위치 계산 (토큰 단위)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        output_start_positions.append(len(prompt_ids))

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=True,
    )

    labels = []

    for input_ids, output_start in zip(
        tokenized["input_ids"],
        output_start_positions,
    ):
        # Output 이전은 전부 -100
        label = [-100] * output_start + input_ids[output_start:]
        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


# LoRA 설정

def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """
    Qwen 계열 번역 SFT에 적합한 LoRA 설정
    """
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

@dataclass
class DataCollatorForCausalLM:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) labels를 분리 (tokenizer.pad에 넘기지 않음)
        labels = [f.pop("labels") for f in features]

        # 2) input_ids / attention_mask만 padding
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # 3) labels 수동 padding
        max_len = batch["input_ids"].size(1)
        padded_labels = []

        for label in labels:
            pad_len = max_len - len(label)
            padded = label + [self.tokenizer.pad_token_id] * pad_len
            padded_labels.append(padded)

        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

        # 4) pad token은 loss 계산에서 제외
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels_tensor
        return batch

# Main

def main() -> None:
    parser = argparse.ArgumentParser()

    # ----- 경로 -----
    parser.add_argument(
        "--model_name",
        type=str,
        default=str(DEFAULT_BASE_MODEL_DIR),
        help="Local path to base Qwen3 model (model_assets/qwen3-1.7b-base)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    # ----- 학습 파라미터 -----
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    # ----- LoRA -----
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ----- 기타 -----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_split", type=float, default=0.02)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer / Model 로드 (로컬)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA 적용

    lora_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset 로드

    raw = load_dataset("json", data_files=str(data_path))
    dataset = raw["train"]

    if args.eval_split > 0:
        split = dataset.train_test_split(
            test_size=args.eval_split,
            seed=args.seed,
        )
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = None

    train_ds = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train dataset",
    )

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda batch: tokenize_function(batch, tokenizer, args.max_length),
            batched=True,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval dataset",
        )

    # Trainer


    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 저장 (LoRA만)

    trainer.model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[DONE] LoRA adapter saved to: {output_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()