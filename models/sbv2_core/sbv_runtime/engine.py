from __future__ import annotations

"""
uv run python -m sbv_runtime \
  --model_onnx model_assets/tts/mai/mai_e281_s263000.onnx \
  --config model_assets/tts/mai/config.json \
  --style_vectors model_assets/tts/mai/style_vectors.npy \
  --bert_onnx_dir bert/deberta-v2-large-japanese-char-wwm-onnx \
  --text "今日はちょっと寒いな。今日、うちに来る？" \
  --out_wav outputs/sbv_runtime_mai.wav \
  --speaker_name mai \
  --device cpu
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import onnxruntime
import soundfile as sf

from . import style_bert_vits2 as _vendored_style_bert_vits2  # noqa: F401
from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, DEFAULT_ONNX_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models
from style_bert_vits2.tts_model import TTSModel


def _abs(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _providers_from_device(device: str) -> list[str | tuple[str, dict[str, str]]]:
    device = device.lower()
    if device == "cuda":
        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


@dataclass
class RuntimeConfig:
    model_onnx: Path
    config: Path
    style_vectors: Path
    bert_onnx_dir: Path
    bert_tokenizer_dir: Path
    speaker_name: str
    style: str
    style_weight: float
    device: str


class SBVRuntime:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        if cfg.model_onnx.suffix.lower() != ".onnx":
            raise ValueError(
                f"ONNX-only runtime: --model_onnx must point to a .onnx file (got: {cfg.model_onnx})"
            )
        self._patch_bert_paths()
        self.model = TTSModel(
            model_path=cfg.model_onnx,
            config_path=cfg.config,
            style_vec_path=cfg.style_vectors,
            device="cpu",
            onnx_providers=_providers_from_device(cfg.device),
        )

    def _patch_bert_paths(self) -> None:
        # g2p tokenizer path (used for word2ph) and ONNX BERT path must both be explicit.
        DEFAULT_BERT_MODEL_PATHS[Languages.JP] = self.cfg.bert_tokenizer_dir
        DEFAULT_ONNX_BERT_MODEL_PATHS[Languages.JP] = self.cfg.bert_onnx_dir
        bert_models.unload_all_models()
        bert_models.unload_all_tokenizers()
        onnx_bert_models.unload_all_models()
        onnx_bert_models.unload_all_tokenizers()

    def synthesize(self, text: str) -> tuple[int, object]:
        if self.cfg.speaker_name not in self.model.spk2id:
            raise KeyError(
                f"speaker_name '{self.cfg.speaker_name}' not in config spk2id. "
                f"available={list(self.model.spk2id.keys())}"
            )
        sr, audio = self.model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=self.model.spk2id[self.cfg.speaker_name],
            style=self.cfg.style,
            style_weight=self.cfg.style_weight,
            line_split=False,
        )
        return sr, audio

    def synthesize_to_file(self, text: str, out_wav: str | Path) -> Path:
        out = _abs(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        sr, audio = self.synthesize(text)
        sf.write(str(out), audio, sr)
        return out


def build_runtime(
    model_onnx: str | Path,
    config: str | Path,
    style_vectors: str | Path,
    bert_onnx_dir: str | Path,
    speaker_name: str,
    style: str = "Neutral",
    style_weight: float = 1.0,
    device: str = "cpu",
    bert_tokenizer_dir: str | Path | None = None,
) -> SBVRuntime:
    bert_onnx = _abs(bert_onnx_dir)
    tokenizer_dir = _abs(bert_tokenizer_dir) if bert_tokenizer_dir is not None else bert_onnx
    cfg = RuntimeConfig(
        model_onnx=_abs(model_onnx),
        config=_abs(config),
        style_vectors=_abs(style_vectors),
        bert_onnx_dir=bert_onnx,
        bert_tokenizer_dir=tokenizer_dir,
        speaker_name=speaker_name,
        style=style,
        style_weight=style_weight,
        device=device,
    )
    return SBVRuntime(cfg)
