"""Kanana VLM 전용 LoRA 병합 스크립트.

예시:
uv run models/qwen3_core/merge_kanana.py \
  --base_model models/qwen3_core/model_assets/saya_vlm_3b_sft \
  --adapter_path models/qwen3_core/model_assets/kanana_3b_grpo \
  --output_dir models/qwen3_core/model_assets/saya_vlm_3b \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def _resolve_dtype(dtype_name: str) -> torch.dtype | None:
    """문자열 dtype 인자를 torch dtype으로 변환한다."""
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key == "auto":
        return None
    if key not in table:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return table[key]


def _pick_src(candidates: list[Path]) -> Path:
    """존재하는 첫 번째 경로를 반환한다."""
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Merge Kanana VLM base + LoRA adapter.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--safe_serialization", action="store_true")
    parser.add_argument("--max_shard_size", type=str, default="10GB")
    parser.add_argument("--offload_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Base Kanana VLM + LoRA adapter를 병합해 단일 모델로 저장한다."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_dtype(args.dtype)
    adapter_dir = Path(args.adapter_path)
    stage_dir = adapter_dir.parent if adapter_dir.name == "lora_adapter" else adapter_dir

    processor_src = _pick_src([
        stage_dir / "processor",
        stage_dir,
        Path(args.base_model),
    ])
    print(f"[MERGE] base_model={args.base_model}")
    print(f"[MERGE] adapter_path={args.adapter_path}")
    print(f"[MERGE] processor_src={processor_src}")
    print(f"[MERGE] output_dir={out_dir}")
    print(f"[MERGE] dtype={args.dtype} device_map={args.device_map}")

    offload_dir = Path(args.offload_dir) if args.offload_dir else (out_dir / "offload")
    if args.device_map == "auto":
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MERGE] offload_dir={offload_dir}")

    processor = AutoProcessor.from_pretrained(
        str(processor_src),
        trust_remote_code=args.trust_remote_code,
    )

    base_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map == "auto":
        base_kwargs["offload_folder"] = str(offload_dir)

    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        **base_kwargs,
    )

    peft_kwargs = {"is_trainable": False}
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
    processor.save_pretrained(str(out_dir))

    print("[MERGE] done")


if __name__ == "__main__":
    main()
