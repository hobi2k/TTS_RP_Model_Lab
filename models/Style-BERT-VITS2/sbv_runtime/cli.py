from __future__ import annotations

"""sbv_runtime CLI 진입점.

이 모듈은 `build_runtime()`를 커맨드라인에서 호출할 수 있도록 연결한다.
성공 시 생성된 WAV 파일의 절대 경로를 stdout으로 출력한다.
"""

import argparse

from .engine import build_runtime


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 구성한다."""
    p = argparse.ArgumentParser("sbv_runtime")
    # 모델 및 런타임 필수 경로
    p.add_argument("--model_onnx", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--style_vectors", required=True)
    p.add_argument("--bert_onnx_dir", required=True)
    p.add_argument("--bert_tokenizer_dir", default=None)
    # 합성 요청 파라미터
    p.add_argument("--text", required=True)
    p.add_argument("--out_wav", required=True)
    p.add_argument("--speaker_name", required=True)
    p.add_argument("--style", default="Neutral")
    p.add_argument("--style_weight", type=float, default=1.0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p


def main() -> None:
    """CLI 입력을 받아 한 번 합성 후 파일 경로를 출력한다."""
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
    # 파이프라인/스크립트에서 소비하기 쉽도록 경로만 단일 라인 출력.
    print(out_path)
