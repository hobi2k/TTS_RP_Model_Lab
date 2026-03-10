"""Kanana 전용 VLM QLoRA SFT 학습 스크립트.

핵심 정책:
1) 모델 로더는 AutoModelForVision2Seq 단일 경로만 사용
2) 4bit QLoRA + LoRA 학습
3) assistant-only loss 지원
4) text-only 대화 데이터 학습을 위해 더미 이미지 자동 주입
5) Kanana 전용 고정 경로

uv run models/qwen3_core/sft_trainer_qlora_kanana.py \
  --model_name models/qwen3_core/model_assets/kanana_3b \
  --data_path /mnt/d/rp_data/rewrite/singleturn_rewrite.jsonl \
  --output_dir models/qwen3_core/model_assets/kanana_3b_stage1 \
  --load_in_4bit \
  --bf16 \
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

PYTORCH_ALLOC_CONF=expandable_segments:True 
uv run models/qwen3_core/sft_trainer_qlora_kanana.py \
  --model_name models/qwen3_core/model_assets/kanana_3b \
  --data_path /mnt/d/rp_data/rewrite/multiturn_rewrite_vlm.jsonl \
  --output_dir models/qwen3_core/model_assets/kanana_3b_stage2 \
  --init_adapter_path models/qwen3_core/model_assets/kanana_3b_stage1/lora_adapter \
  --load_in_4bit \
  --bf16 \
  --max_length 2048 \
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
from datasets import Dataset, load_dataset
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    """학습 인자를 파싱한다.

    Args:
        없음
    Returns:
        argparse.Namespace: 파싱된 인자
    """
    p = argparse.ArgumentParser(description="Kanana-only VLM QLoRA trainer")
    p.add_argument("--model_name", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--init_adapter_path", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--messages_key", type=str, default="messages")
    p.add_argument("--conversations_key", type=str, default="conversations")
    p.add_argument("--max_length", type=int, default=2048)
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
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
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
    """대화 역할명을 표준 role로 정규화한다.

    Args:
        role: 원본 role 값
    Returns:
        str | None: 정규화된 role 또는 None
    """
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


def _extract_messages(example: dict[str, Any], messages_key: str, conversations_key: str) -> list[dict[str, Any]]:
    """원본 샘플에서 텍스트 메시지를 추출한다.

    Args:
        example: 원본 데이터 샘플
        messages_key: 우선 메시지 키
        conversations_key: 대체 메시지 키
    Returns:
        list[dict[str, Any]]: role/content 리스트
    """
    raw = example.get(messages_key)
    if raw is None:
        raw = example.get(conversations_key)
    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _normalize_role(item.get("role", item.get("from", item.get("speaker"))))
        if role is None:
            continue
        text = item.get("content", item.get("value", item.get("text", "")))
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        out.append({"role": role, "content": text})
    return out


def _to_kanana_vlm_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """텍스트 메시지를 Kanana model-card 스타일 conv 구조로 변환한다.

    Args:
        messages: 텍스트 메시지 리스트
    Returns:
        list[dict[str, Any]]: Kanana VLM 메시지
    """
    out: list[dict[str, Any]] = []
    image_inserted = False
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and not image_inserted:
            # Kanana 권장 포맷: 첫 user content 문자열에 <image> 토큰을 넣는다.
            content = f"<image>\n{content}"
            image_inserted = True
        out.append({"role": role, "content": content})
    return out if image_inserted else []


def build_example(example: dict[str, Any], messages_key: str, conversations_key: str) -> dict[str, Any]:
    """원본 샘플을 학습용 샘플로 변환한다.

    Args:
        example: 원본 샘플
        messages_key: 우선 메시지 키
        conversations_key: 대체 메시지 키
    Returns:
        dict[str, Any]: {"messages": ...}
    """
    messages = _extract_messages(example, messages_key, conversations_key)
    if not messages:
        return {"messages": []}
    return {"messages": _to_kanana_vlm_messages(messages)}


def is_valid_example(example: dict[str, Any]) -> bool:
    """학습 가능한 샘플인지 검사한다.

    Args:
        example: 학습용 샘플
    Returns:
        bool: 유효 여부
    """
    messages = example.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return False
    has_user = any(m.get("role") == "user" for m in messages if isinstance(m, dict))
    has_assistant = any(m.get("role") == "assistant" for m in messages if isinstance(m, dict))
    return has_user and has_assistant


class KananaCollator:
    """Kanana VLM 학습용 배치 콜레이터.

    Args:
        processor: Kanana processor
        max_length: 최대 토큰 길이
        dummy_image_size: 더미 이미지 크기
        assistant_only_loss: 마지막 assistant 턴만 loss 반영할지 여부
    Returns:
        없음
    """

    def __init__(self, processor: Any, max_length: int, dummy_image_size: int, assistant_only_loss: bool) -> None:
        self.processor = processor
        self.max_length = max_length
        self.dummy_image_size = dummy_image_size
        self.assistant_only_loss = assistant_only_loss

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """features를 모델 입력 배치로 변환한다.

        Args:
            features: 샘플 리스트
        Returns:
            dict[str, torch.Tensor]: model(**inputs)에 전달할 텐서 dict
        """
        data_list: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for ex in features:
            # text-only 데이터도 VLM 경로를 타도록 더미 이미지를 넣는다.
            image = Image.new("RGB", (self.dummy_image_size, self.dummy_image_size), color=(255, 255, 255))
            data_list.append({"conv": ex["messages"], "image": [image]})
            images.append(image)

        batch = self.processor.batch_encode_collate(
            data_list=data_list,
            padding="longest",
            padding_side="right",
            max_length=self.max_length,
            add_generation_prompt=False,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_id = tokenizer.pad_token_id if tokenizer is not None else None
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        # Kanana는 이미지 자리 토큰을 음수 id로 표기하므로 loss에서 제거한다.
        labels[input_ids < 0] = -100

        if self.assistant_only_loss:
            for row_idx, ex in enumerate(features):
                msgs = ex["messages"]
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

                # 마지막 assistant 이전까지 prefix 길이를 계산해 라벨 마스킹한다.
                prefix = self.processor(
                    data={"conv": prefix_messages, "image": [images[row_idx]]},
                    max_length=self.max_length,
                    add_generation_prompt=True,
                )
                prefix_len = int(prefix["text"]["input_ids"].shape[0])
                if prefix_len > 0:
                    labels[row_idx, :prefix_len] = -100

        batch["labels"] = labels
        return batch


def load_dataset_for_train(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    """JSONL 데이터셋을 로드하고 전처리한다.

    Args:
        args: CLI 인자
    Returns:
        tuple[Dataset, Dataset | None]: train/eval 데이터셋
    """
    raw = load_dataset("json", data_files=args.data_path)["train"]
    if args.eval_split > 0:
        split = raw.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = raw
        eval_ds = None

    train_ds = train_ds.map(
        lambda ex: build_example(ex, args.messages_key, args.conversations_key),
        remove_columns=train_ds.column_names,
    )
    train_ds = train_ds.filter(is_valid_example)

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda ex: build_example(ex, args.messages_key, args.conversations_key),
            remove_columns=eval_ds.column_names,
        )
        eval_ds = eval_ds.filter(is_valid_example)

    if len(train_ds) == 0:
        raise ValueError("No valid training samples after preprocessing.")
    return train_ds, eval_ds


def build_lora(args: argparse.Namespace) -> LoraConfig:
    """LoRA 설정을 생성한다.

    Args:
        args: CLI 인자
    Returns:
        LoraConfig: LoRA 설정
    """
    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("--lora_target_modules is empty.")
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,
        target_modules=targets,
    )


def main() -> None:
    """학습 엔트리 포인트.

    Args:
        없음
    Returns:
        없음
    """
    args = parse_args()
    if not args.trust_remote_code:
        raise ValueError("Kanana 전용 스크립트는 --trust_remote_code가 필요합니다.")

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
        trust_remote_code=True,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Kanana 전용: 단일 로더 경로만 사용한다.
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )

    # 학습 시 KV cache는 불필요하므로 비활성화해 메모리를 절약한다.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.load_in_4bit:
        # Kanana는 일부 입력-임베딩 경로 미구현 이슈가 있어 checkpointing hook은 사용하지 않는다.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )

    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter_path,
            is_trainable=True,
        )
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
        gradient_checkpointing=False,
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
        data_collator=KananaCollator(
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
