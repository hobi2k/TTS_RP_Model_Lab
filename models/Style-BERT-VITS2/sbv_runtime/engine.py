from __future__ import annotations

"""Style-BERT-VITS2 ONNX 런타임 래퍼.

이 모듈의 목적:
1) ONNX TTS 모델 + BERT ONNX/토크나이저 경로를 명시적으로 고정한다.
2) `TTSModel` 추론을 래핑해 간단한 Python API(`build_runtime`, `synthesize`)를 제공한다.
3) CLI/워커에서 공통으로 사용할 수 있는 단일 진입점을 유지한다.

실행 예:
    uv run --active --no-sync python -m sbv_runtime \
      --model_onnx model_assets/saya/saya_e150_s57000.onnx \
      --config model_assets/saya/config.json \
      --style_vectors model_assets/saya/style_vectors.npy \
      --bert_onnx_dir bert/deberta-v2-large-japanese-char-wwm-onnx \
      --text "今日はちょっと寒いな。今日、うちに来る？" \
      --out_wav outputs/sbv_runtime_saya.wav \
      --speaker_name saya \
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
    """사용자 입력 경로를 절대 경로로 정규화한다."""
    return Path(path).expanduser().resolve()


def _providers_from_device(device: str) -> list[str | tuple[str, dict[str, str]]]:
    """요청 디바이스 문자열에 맞는 ONNX Runtime provider 순서를 만든다.

    - `cuda` 요청 시 CUDA provider가 실제로 사용 가능한 경우에만 CUDA -> CPU 순으로 반환한다.
    - 그 외에는 CPU provider만 반환한다.
    """
    device = device.lower()
    if device == "cuda":
        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return [
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            ]
    return [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


@dataclass
class RuntimeConfig:
    """SBVRuntime 초기화에 필요한 파일/추론 파라미터 묶음."""

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
    """Style-BERT-VITS2 ONNX 추론 런타임.

    인스턴스 생성 시점에:
    - BERT tokenizer/ONNX 모델 기본 경로를 현재 런타임 기준으로 패치하고
    - `TTSModel`을 한 번 로드한다.
    """

    def __init__(self, cfg: RuntimeConfig) -> None:
        """런타임을 초기화하고 ONNX 모델을 로드한다."""
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
        """BERT 관련 전역 경로를 현재 런타임 값으로 덮어쓴다.

        Style-BERT-VITS2 내부는 전역 기본 경로를 참조하므로, 워커/CLI 환경에서
        원하는 디렉터리를 강제하려면 이 단계가 필요하다.
        """
        # g2p tokenizer path(word2ph 계산)와 ONNX BERT 경로를 모두 명시적으로 고정한다.
        DEFAULT_BERT_MODEL_PATHS[Languages.JP] = self.cfg.bert_tokenizer_dir
        DEFAULT_ONNX_BERT_MODEL_PATHS[Languages.JP] = self.cfg.bert_onnx_dir
        # 기존 캐시를 비워 경로 패치가 즉시 반영되도록 한다.
        bert_models.unload_all_models()
        bert_models.unload_all_tokenizers()
        onnx_bert_models.unload_all_models()
        onnx_bert_models.unload_all_tokenizers()

    def synthesize(self, text: str) -> tuple[int, object]:
        """입력 텍스트를 합성해 (샘플레이트, 오디오배열)을 반환한다."""
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
        """입력 텍스트를 합성해 WAV 파일로 저장한다."""
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
    style_weight: float = 10.0,
    device: str = "cpu",
    bert_tokenizer_dir: str | Path | None = None,
) -> SBVRuntime:
    """SBVRuntime을 생성하는 팩토리 함수.

    Args:
        model_onnx: 화자 ONNX 모델 파일 경로.
        config: 해당 모델의 config.json 경로.
        style_vectors: style_vectors.npy 경로.
        bert_onnx_dir: 일본어 ONNX BERT 디렉터리.
        speaker_name: config의 `spk2id`에 존재하는 화자 이름.
        style: 기본 스타일 이름.
        style_weight: 스타일 가중치.
        device: `cpu` 또는 `cuda`.
        bert_tokenizer_dir: tokenizer 디렉터리. None이면 `bert_onnx_dir`를 재사용한다.

    Returns:
        SBVRuntime: 초기화된 런타임 인스턴스.
    """
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
