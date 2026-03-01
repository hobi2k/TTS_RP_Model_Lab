# sbv_runtime (ONNX-only)

`sbv_runtime` is a portable runtime directory for ONNX inference without PyTorch.
This runtime is configured for **Japanese-only (JP)**.

## Run

```bash
python -m sbv_runtime \
  --model_onnx /path/to/model.onnx \
  --config /path/to/config.json \
  --style_vectors /path/to/style_vectors.npy \
  --bert_onnx_dir /path/to/bert_onnx_dir \
  --text "こんにちは。" \
  --out_wav /tmp/out.wav \
  --speaker_name saya \
  --device cpu
```

## Required assets

- TTS ONNX model (`model.onnx`)
- model config (`config.json`)
- style vectors (`style_vectors.npy`)
- BERT ONNX directory containing:
  - `model_fp16.onnx` or `model.onnx`
  - tokenizer files (`tokenizer.json`, etc.)

If tokenizer files are in a different directory, pass `--bert_tokenizer_dir`.

## Notes

- Runtime is ONNX-only. Do not pass `.safetensors` to `--model_onnx`.
- JP only.
- Provider fallback:
  - `--device cuda`: CUDA EP if available, else CPU
  - `--device cpu`: CPU only
