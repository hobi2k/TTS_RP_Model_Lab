from __future__ import annotations

import argparse

from .engine import build_runtime


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("sbv_runtime")
    p.add_argument("--model_onnx", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--style_vectors", required=True)
    p.add_argument("--bert_onnx_dir", required=True)
    p.add_argument("--bert_tokenizer_dir", default=None)
    p.add_argument("--text", required=True)
    p.add_argument("--out_wav", required=True)
    p.add_argument("--speaker_name", required=True)
    p.add_argument("--style", default="Neutral")
    p.add_argument("--style_weight", type=float, default=1.0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    runtime = build_runtime(
        model_onnx=args.model_onnx,
        config=args.config,
        style_vectors=args.style_vectors,
        bert_onnx_dir=args.bert_onnx_dir,
        bert_tokenizer_dir=args.bert_tokenizer_dir,
        speaker_name=args.speaker_name,
        style=args.style,
        style_weight=args.style_weight,
        device=args.device,
    )
    out_path = runtime.synthesize_to_file(args.text, args.out_wav)
    print(out_path)
