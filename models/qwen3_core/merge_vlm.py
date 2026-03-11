"""Qwen 계열 VLM 전용 LoRA 병합 스크립트.

예시:
uv run models/qwen3_core/merge_vlm.py \
  --base_model models/qwen3_core/model_assets/qwen3.5-4b \
  --adapter_path models/qwen3_core/model_assets/qwen3.5-4b_stage1/lora_adapter \
  --output_dir models/qwen3_core/model_assets/qwen3.5-4b_merged \
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
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel


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


def _has_any_file(dir_path: Path, candidates: list[str]) -> bool:
    """디렉토리에 후보 파일 중 하나라도 존재하는지 확인한다."""
    return any((dir_path / name).exists() for name in candidates)


def _resolve_processor_source(base_model: Path, adapter_path: Path) -> Path:
    """병합 결과물에 저장할 processor/tokenizer 소스를 결정한다."""
    stage_dir = adapter_path.parent if adapter_path.name == "lora_adapter" else adapter_path

    processor_files = ["processor_config.json", "preprocessor_config.json"]
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "chat_template.jinja",
    ]

    candidates = [
        stage_dir / "processor",
        stage_dir / "tokenizer",
        stage_dir,
        base_model,
    ]
    for path in candidates:
        if _has_any_file(path, processor_files) or _has_any_file(path, tokenizer_files):
            return path
    return base_model


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Merge Qwen VLM base + LoRA adapter.")
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
    """Base Qwen VLM + LoRA adapter를 병합해 단일 모델로 저장한다."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_dtype(args.dtype)
    base_model_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter_path)
    processor_src = _resolve_processor_source(base_model_dir, adapter_dir)

    print(f"[MERGE_VLM] base_model={base_model_dir}")
    print(f"[MERGE_VLM] adapter_path={adapter_dir}")
    print(f"[MERGE_VLM] processor_src={processor_src}")
    print(f"[MERGE_VLM] output_dir={out_dir}")
    print(f"[MERGE_VLM] dtype={args.dtype} device_map={args.device_map}")

    offload_dir = Path(args.offload_dir) if args.offload_dir else (out_dir / "offload")
    if args.device_map == "auto":
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MERGE_VLM] offload_dir={offload_dir}")

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

    base_model = AutoVLMModel.from_pretrained(
        str(base_model_dir),
        **base_kwargs,
    )

    peft_kwargs = {"is_trainable": False}
    if args.device_map == "auto":
        peft_kwargs["offload_dir"] = str(offload_dir)

    lora_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        **peft_kwargs,
    )
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(
        str(out_dir),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    processor.save_pretrained(str(out_dir))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out_dir))

    print("[MERGE_VLM] done")


if __name__ == "__main__":
    main()
