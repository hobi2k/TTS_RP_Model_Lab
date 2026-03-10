"""Gemma/Rosetta 전용 LoRA 병합 스크립트.

핵심 정책:
- 가중치는 base model + LoRA adapter를 merge한다.
- tokenizer/chat template는 객체를 다시 save하지 않고 stage tokenizer 파일을 그대로 복사한다.

예시:
uv run models/qwen3_core/merge_gemma.py \
  --base_model models/qwen3_core/model_assets/rosetta_4b \
  --adapter_path models/qwen3_core/model_assets/rosetta_4b_stage1/lora_adapter \
  --output_dir models/qwen3_core/model_assets/rosetta_4b_merge1 \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


TOKENIZER_COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
]


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


def _has_tokenizer_files(model_dir: Path) -> bool:
    """주요 tokenizer 파일 존재 여부를 확인한다."""
    required = [
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    return all((model_dir / name).exists() for name in required)


def _resolve_tokenizer_dir(base_model: Path, adapter_path: Path, explicit: str | None) -> Path:
    """병합 결과물에 복사할 tokenizer 소스를 결정한다."""
    if explicit:
        path = Path(explicit)
        if not _has_tokenizer_files(path):
            raise FileNotFoundError(f"tokenizer_dir missing required files: {path}")
        return path

    stage_dir = adapter_path.parent if adapter_path.name == "lora_adapter" else adapter_path
    stage_tokenizer = stage_dir / "tokenizer"
    if _has_tokenizer_files(stage_tokenizer):
        return stage_tokenizer
    if _has_tokenizer_files(stage_dir):
        return stage_dir
    if _has_tokenizer_files(base_model):
        return base_model
    raise FileNotFoundError("Could not resolve tokenizer source.")


def _copy_tokenizer_files(tokenizer_dir: Path, out_dir: Path) -> None:
    """stage tokenizer 메타데이터를 결과물에 원본 그대로 복사한다."""
    copied = []
    for name in TOKENIZER_COPY_FILES:
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)

    if not copied:
        raise FileNotFoundError(f"No tokenizer files were copied from: {tokenizer_dir}")

    print(f"[MERGE_GEMMA] copied tokenizer files: {', '.join(copied)}")


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Merge Gemma/Rosetta base + LoRA adapter.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Optional explicit tokenizer/chat_template source directory.",
    )
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
    """Base Gemma/Rosetta + LoRA adapter를 병합해 단일 모델로 저장한다."""
    args = parse_args()

    base_model = Path(args.base_model)
    adapter_path = Path(args.adapter_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _resolve_dtype(args.dtype)
    tokenizer_dir = _resolve_tokenizer_dir(base_model, adapter_path, args.tokenizer_dir)

    print(f"[MERGE_GEMMA] base_model={base_model}")
    print(f"[MERGE_GEMMA] adapter_path={adapter_path}")
    print(f"[MERGE_GEMMA] tokenizer_dir={tokenizer_dir}")
    print(f"[MERGE_GEMMA] output_dir={out_dir}")
    print(f"[MERGE_GEMMA] dtype={args.dtype} device_map={args.device_map}")

    offload_dir = Path(args.offload_dir) if args.offload_dir else (out_dir / "offload")
    if args.device_map == "auto":
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MERGE_GEMMA] offload_dir={offload_dir}")

    base_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map == "auto":
        base_kwargs["offload_folder"] = str(offload_dir)

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        **base_kwargs,
    )

    peft_kwargs = {"is_trainable": False}
    if args.device_map == "auto":
        peft_kwargs["offload_dir"] = str(offload_dir)

    lora_model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        **peft_kwargs,
    )
    merged_model = lora_model.merge_and_unload()

    merged_model.save_pretrained(
        str(out_dir),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    _copy_tokenizer_files(tokenizer_dir, out_dir)

    print("[MERGE_GEMMA] done")


if __name__ == "__main__":
    main()
