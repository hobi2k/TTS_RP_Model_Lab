"""
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b-instruct \
  --data_path /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --output_dir /tmp/sft_debug \
  --load_in_4bit \
  --bf16 \
  --assistant_only_loss \
  --debug_label_mask \
  --debug_label_rows 520 \
  --debug_only

# 1) 현재 돌아가는 학습 PID 확인
ps -ef | grep sft_trainer_qlora.py | grep -v grep
# 예: PID가 12345

# 2) 그 PID가 끝날 때까지 대기 후 다음 학습 실행
while kill -0 12345 2>/dev/null; do sleep 30; done
uv run models/qwen3_core/sft_trainer_qlora.py ...다음옵션...

# Recommended presets for ~843 samples (single+multi-turn mixed),
# where per-sample tokens are large and max_length 4096 is required.


# Stage 1) 싱글턴 데이터로 1차 LoRA 학습
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b-instruct \
  --data_path /mnt/d/rp_data/singleturn/rp_singleturn_cleaned.jsonl \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage1 \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 8 \
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
  --assistant_only_loss

uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/saya_rp_4b \
  --data_path /mnt/d/rp_data/rewrite/singleturn_rewrite.jsonl \
  --output_dir models/qwen3_core/model_assets/saya_rp_4b_lora_stage1 \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 8 \
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
  --assistant_only_loss
  
# Stage 2) Stage1 adapter를 로드해 멀티턴 데이터로 추가 학습
while kill -0 502791 2>/dev/null; do sleep 30; done
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b-instruct \
  --data_path /mnt/d/rp_data/v7/rp_datum_unite_cleaned.jsonl \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage2 \
  --init_adapter_path models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage1/lora_adapter \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --save_steps 25 \
  --save_total_limit 6 \
  --eval_split 0.1 \
  --eval_strategy steps \
  --eval_steps 25 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --prediction_loss_only \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --assistant_only_loss

while kill -0 115518 2>/dev/null; do sleep 30; done
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/saya_rp_4b \
  --data_path /mnt/d/rp_data/rewrite/multiturn_rewrite.jsonl \
  --output_dir models/qwen3_core/model_assets/saya_rp_4b_lora_stage2 \
  --init_adapter_path models/qwen3_core/model_assets/saya_rp_4b_lora_stage1/lora_adapter \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --save_steps 25 \
  --save_total_limit 6 \
  --eval_split 0.1 \
  --eval_strategy steps \
  --eval_steps 25 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --prediction_loss_only \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --assistant_only_loss
"""


from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


# =========================================================
# LoRA 설정
# =========================================================

def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# =========================================================
# Chat template
# =========================================================

def _normalize_role(role: Any) -> Optional[str]:
    if not isinstance(role, str):
        return None
    r = role.strip().lower()
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
    return role_map.get(r)


def _extract_messages(
    example: Dict[str, Any],
    messages_key: str,
    conversations_key: str,
) -> Optional[List[Dict[str, str]]]:
    # case 1) standard chat format: {"messages":[{"role":"...","content":"..."}]}
    msgs = example.get(messages_key)

    # case 2) common alt format: {"conversations":[{"from":"human","value":"..."}]}
    if msgs is None:
        msgs = example.get(conversations_key)

    if isinstance(msgs, list) and msgs:
        out: List[Dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue

            role = _normalize_role(m.get("role", m.get("from", m.get("speaker"))))
            if role is None:
                continue

            content = m.get("content", m.get("value", m.get("text", "")))
            if not isinstance(content, str):
                continue

            content = content.strip()
            if not content:
                continue

            out.append({"role": role, "content": content})

        if out:
            return out

    return None


def build_training_example(
    example,
    messages_key: str,
    conversations_key: str,
):
    messages = _extract_messages(
        example=example,
        messages_key=messages_key,
        conversations_key=conversations_key,
    )
    if not messages:
        return {"messages": []}

    return {"messages": messages}


def is_valid_training_example(example: Dict[str, Any]) -> bool:
    return is_conversational_example(example)


def is_conversational_example(example: Dict[str, Any]) -> bool:
    msgs = example.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    first = msgs[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def debug_print_label_mask(trainer: SFTTrainer, tokenizer, max_rows: int = 220) -> None:
    """
    학습 전에 라벨 마스킹이 정상인지 확인하기 위한 디버그 출력.

    출력 내용:
    - 첫 배치 1개 샘플의 토큰별 (token, label 상태)
    - label != -100 토큰 개수
    - 첫 라벨 시작/끝 인덱스
    """
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))
    input_ids = batch["input_ids"][0].detach().cpu().tolist()
    labels = batch["labels"][0].detach().cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    labeled_positions = [i for i, v in enumerate(labels) if v != -100]

    print("\n[DEBUG_LABEL_MASK] ===== first train sample =====", flush=True)
    print(f"[DEBUG_LABEL_MASK] total_tokens={len(input_ids)}", flush=True)
    print(f"[DEBUG_LABEL_MASK] labeled_tokens={len(labeled_positions)}", flush=True)
    if labeled_positions:
        print(
            f"[DEBUG_LABEL_MASK] first_labeled_idx={labeled_positions[0]} "
            f"last_labeled_idx={labeled_positions[-1]}",
            flush=True,
        )
    else:
        print("[DEBUG_LABEL_MASK] labeled span is empty", flush=True)

    rows = min(len(input_ids), max_rows)
    print("[DEBUG_LABEL_MASK] idx\tmask\ttoken\tlabel_token", flush=True)
    for i in range(rows):
        lbl = labels[i]
        masked = "L" if lbl != -100 else "-"
        tok = tokens[i].replace("\n", "\\n")
        if lbl != -100:
            lbl_tok = tokenizer.convert_ids_to_tokens([int(lbl)])[0]
            lbl_tok = lbl_tok.replace("\n", "\\n")
        else:
            lbl_tok = ""
        print(f"[DEBUG_LABEL_MASK] {i}\t{masked}\t{tok}\t{lbl_tok}", flush=True)
    print("[DEBUG_LABEL_MASK] ==================================\n", flush=True)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="models/qwen3_core/model_assets/qwen3-4b-instruct",
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_adapter_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--eval_split", type=float, default=0.0)
    parser.add_argument("--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--prediction_loss_only", action="store_true")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--messages_key", type=str, default="messages")
    parser.add_argument("--conversations_key", type=str, default="conversations")
    parser.add_argument("--assistant_only_loss", dest="assistant_only_loss", action="store_true")
    parser.add_argument("--no-assistant_only_loss", dest="assistant_only_loss", action="store_false")
    parser.set_defaults(assistant_only_loss=True)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa", choices=["sdpa", "eager"])

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug_label_mask", action="store_true")
    parser.add_argument("--debug_label_rows", type=int, default=220)
    parser.add_argument("--debug_only", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Force-load local chat template file when available.
    model_dir = Path(args.model_name)
    template_path = model_dir / "chat_template.jinja"
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    if args.assistant_only_loss:
        chat_template = getattr(tokenizer, "chat_template", "") or ""
        if "{% generation %}" not in chat_template:
            # Qwen3-8B template may not include generation markers required by TRL assistant_only_loss.
            # Fallback to sibling 4B-instruct template when available.
            fallback_template_path = model_dir.parent / "qwen3-4b-instruct" / "chat_template.jinja"
            if fallback_template_path.exists():
                fallback_template = fallback_template_path.read_text(encoding="utf-8")
                if "{% generation %}" in fallback_template:
                    tokenizer.chat_template = fallback_template
                    chat_template = fallback_template
                    print(
                        "[WARN] Current chat template has no `{% generation %}` block. "
                        f"Using fallback template: {fallback_template_path}",
                        flush=True,
                    )
            if "{% generation %}" not in chat_template:
                raise ValueError(
                    "assistant_only_loss=True requires a chat template with `{% generation %}` block. "
                    f"Please fix: {template_path if template_path.exists() else 'tokenizer.chat_template'}"
                )

    # dtype
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

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.init_adapter_path,
            is_trainable=True,
        )
        peft_config_for_trainer = None
    else:
        model = base_model
        peft_config_for_trainer = lora_config

    raw = load_dataset("json", data_files=args.data_path)["train"]

    if args.eval_split > 0:
        split = raw.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = raw, None

    train_ds = train_ds.map(
        lambda ex: build_training_example(
            ex,
            messages_key=args.messages_key,
            conversations_key=args.conversations_key,
        ),
        remove_columns=train_ds.column_names,
    )
    original_train_len = len(train_ds)
    train_ds = train_ds.filter(is_valid_training_example)
    dropped_train = original_train_len - len(train_ds)
    if dropped_train > 0:
        raise ValueError(
            f"Found {dropped_train} non-conversational samples in train split. "
            "This trainer now requires conversational format only: "
            "{'messages': [{'role': ..., 'content': ...}, ...]}."
        )

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda ex: build_training_example(
                ex,
                messages_key=args.messages_key,
                conversations_key=args.conversations_key,
            ),
            remove_columns=eval_ds.column_names,
        )
        original_eval_len = len(eval_ds)
        eval_ds = eval_ds.filter(is_valid_training_example)
        dropped_eval = original_eval_len - len(eval_ds)
        if dropped_eval > 0:
            raise ValueError(
                f"Found {dropped_eval} non-conversational samples in eval split. "
                "This trainer now requires conversational format only: "
                "{'messages': [{'role': ..., 'content': ...}, ...]}."
            )

    if len(train_ds) == 0:
        raise ValueError("No valid training samples after preprocessing.")

    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    eval_steps = args.eval_steps if args.eval_steps > 0 else args.save_steps
    if args.load_best_model_at_end and eval_strategy == "no":
        raise ValueError("`--load_best_model_at_end` requires eval enabled. Set `--eval_split > 0` and `--eval_strategy steps|epoch`.")

    sft_kwargs = dict(
        output_dir=str(output_dir),
        max_length=args.max_length,
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
        remove_unused_columns=True,
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )
    if "assistant_only_loss" in inspect.signature(SFTConfig).parameters:
        sft_kwargs["assistant_only_loss"] = args.assistant_only_loss
    elif args.assistant_only_loss:
        print("[WARN] This TRL version does not support `assistant_only_loss`; training will use full-sequence loss.")

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config_for_trainer,
        processing_class=tokenizer,
    )

    if args.debug_label_mask:
        debug_print_label_mask(
            trainer=trainer,
            tokenizer=tokenizer,
            max_rows=args.debug_label_rows,
        )
        if args.debug_only:
            print("[DEBUG_LABEL_MASK] debug_only=True, skip training.", flush=True)
            return

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()
