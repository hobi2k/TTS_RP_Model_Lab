"""
uv run models/qwen3_core/merge.py \
  --base_model models/qwen3_core/model_assets/qwen3-4b-instruct \
  --adapter_path models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage2/lora_adapter \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code

uv run models/qwen3_core/merge.py \
  --base_model models/qwen3_core/model_assets/qwen3_4b_rp \
  --adapter_path models/qwen3_core/model_assets/qwen3_4b_rp_grpo \
  --output_dir models/qwen3_core/model_assets/saya_rp_4b \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code

uv run models/qwen3_core/merge.py \
  --base_model models/qwen3_core/model_assets/qwen3-1.7b-base \
  --adapter_path models/qwen3_core/model_assets/qwen3_1.7_ko2ja_lora/lora_adapter \
  --output_dir models/qwen3_core/model_assets/qtranslator_1.7b \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code

while kill -0 1866426 2>/dev/null; do sleep 30; done
uv run models/qwen3_core/merge.py \
  --base_model models/qwen3_core/model_assets/YanoljaNEXT-EEVE-7B-v2_lora_sft1 \
  --adapter_path models/qwen3_core/model_assets/YanoljaNEXT-EEVE-7B-v2_lora_stage2/lora_adapter \
  --output_dir models/qwen3_core/model_assets/saya_rp_7b_v2_sft \
  --dtype bf16 \
  --device_map auto \
  --offload_dir /tmp/merge_offload \
  --safe_serialization \
  --trust_remote_code
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    table = {
        "auto": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return table[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge base instruct model + LoRA adapter into a standalone model."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="models/qwen3_core/model_assets/qwen3-4b-instruct",
        help="Base instruct model path or HF repo id.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="LoRA adapter directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where merged model/tokenizer will be saved.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="Load dtype for merge.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for loading base model. e.g. "auto", "cpu", "cuda:0".',
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--safe_serialization", action="store_true")
    parser.add_argument("--max_shard_size", type=str, default="10GB")
    parser.add_argument(
        "--offload_dir",
        type=str,
        default=None,
        help=(
            "Directory for accelerate offloading when --device_map auto triggers "
            "CPU/disk offload. Defaults to <output_dir>/offload."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = _resolve_dtype(args.dtype)
    torch_dtype = None if args.dtype == "auto" else dtype

    print(f"[MERGE] base_model={args.base_model}")
    print(f"[MERGE] adapter_path={args.adapter_path}")
    print(f"[MERGE] output_dir={out_dir}")
    print(f"[MERGE] dtype={args.dtype} device_map={args.device_map}")
    offload_dir = Path(args.offload_dir) if args.offload_dir else (out_dir / "offload")
    if args.device_map == "auto":
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MERGE] offload_dir={offload_dir}")

    # sft의 경우, tokenizer는 LoRA 모델 폴더의 tokenizer/ 하위 디렉토리를 강제로 사용
    # 예) adapter_path=.../lora_adapter 이면 tokenizer_dir=.../tokenizer
    # grpo의 경우 같은 폴더 사용
    adapter_dir = Path(args.adapter_path)
    tokenizer_dir = adapter_dir.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=args.trust_remote_code,
    )

    base_model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map == "auto":
        base_model_kwargs["offload_folder"] = str(offload_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **base_model_kwargs,
    )

    peft_kwargs = {
        "is_trainable": False,
    }
    if args.device_map == "auto":
        peft_kwargs["offload_dir"] = str(offload_dir)

    lora_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        **peft_kwargs,
    )
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(
        str(out_dir),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(out_dir))

    print("[MERGE] done")


if __name__ == "__main__":
    main()
