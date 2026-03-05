"""
SBV2 Worker Process (Style-BERT-VITS2 ONNX Inference Worker)

- 프로세스 단위로 ONNX TTS 모델을 1회 로드
- stdin JSON 요청을 받아 wav를 생성
- stdout JSON으로 결과 경로를 반환

입출력 규약:
- 입력(한 줄 JSON): {"text": "...", "style": 0, "style_weight": 1.0, "speaker_name": "saya"}
- 출력(성공): {"wav_path": "..."}
- 출력(실패): {"error": "..."}
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import onnxruntime
import soundfile as sf

from style_bert_vits2.constants import Languages

from sbv_runtime.engine import build_runtime


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SPEAKER = "saya"
DEFAULT_STYLE_FALLBACK = "Neutral"
DEVICE = (
    "cuda"
    if "CUDAExecutionProvider" in set(onnxruntime.get_available_providers())
    else "cpu"
)

BERT_TOKENIZER_DIR = BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm"
BERT_ONNX_DIR = BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm-onnx"

MODEL_SPECS = {
    "saya": {
        "model_onnx": BASE_DIR / "model_assets" / "saya" / "saya_e150_s57000.onnx",
        "config": BASE_DIR / "model_assets" / "saya" / "config.json",
        "style_vectors": BASE_DIR / "model_assets" / "saya" / "style_vectors.npy",
        "speaker_name": "saya",
    },
    "mai": {
        "model_onnx": BASE_DIR / "model_assets" / "mai" / "mai_e100_s38000.onnx",
        "config": BASE_DIR / "model_assets" / "mai" / "config.json",
        "style_vectors": BASE_DIR / "model_assets" / "mai" / "style_vectors.npy",
        "speaker_name": "mai",
    },
}


def _resolve_style_name(runtime, style_index: int) -> str:
    """정수 스타일 인덱스를 runtime의 실제 스타일 이름으로 변환한다.

    Raises:
        RuntimeError: 모델 style2id가 비어 있을 때.
        IndexError: 스타일 인덱스가 범위를 벗어날 때.
    """
    style_names = list(runtime.model.style2id.keys())
    if not style_names:
        raise RuntimeError("style2id가 비어 있어 스타일을 선택할 수 없습니다.")

    idx = int(style_index)
    if idx < 0 or idx >= len(style_names):
        raise IndexError(f"style index out of range: {idx} (0..{len(style_names)-1})")
    return style_names[idx]


def load_engine(speaker_name: str) -> dict:
    """화자별 런타임 엔진을 로드한다.

    검증 단계:
    - speaker_name이 지원 목록에 존재하는지 확인
    - ONNX/config/style_vectors 파일 존재 확인
    - JP tokenizer/ONNX BERT 디렉터리 존재 확인

    Returns:
        dict: {"speaker_name", "runtime", "style_names"} 형태 엔진 상태.
    """
    key = (speaker_name or DEFAULT_SPEAKER).strip().lower()
    if key not in MODEL_SPECS:
        raise ValueError(f"지원하지 않는 speaker_name: {speaker_name}. 사용 가능: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[key]
    for field, path in spec.items():
        if field == "speaker_name":
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"{field} 파일이 없습니다: {path}")

    if not BERT_TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"BERT tokenizer 디렉토리가 없습니다: {BERT_TOKENIZER_DIR}")
    if not BERT_ONNX_DIR.exists():
        raise FileNotFoundError(f"BERT ONNX 디렉토리가 없습니다: {BERT_ONNX_DIR}")

    print(
        f"[SBV2_WORKER] speaker={key} ONNX runtime 로드: {spec['model_onnx']} (device={DEVICE})",
        flush=True,
    )
    runtime = build_runtime(
        model_onnx=spec["model_onnx"],
        config=spec["config"],
        style_vectors=spec["style_vectors"],
        bert_onnx_dir=BERT_ONNX_DIR,
        bert_tokenizer_dir=BERT_TOKENIZER_DIR,
        speaker_name=spec["speaker_name"],
        style=DEFAULT_STYLE_FALLBACK,
        style_weight=1.0,
        device=DEVICE,
    )

    style_names = list(runtime.model.style2id.keys())
    print(
        f"[SBV2_WORKER] speaker={key} 사용 가능 스타일: {style_names}",
        flush=True,
    )

    return {
        "speaker_name": key,
        "runtime": runtime,
        "style_names": style_names,
    }


def swap_engine_if_needed(current_engine: dict, target_speaker: str) -> dict:
    """요청 화자가 현재 화자와 다를 때만 런타임을 교체한다.

    기존 runtime에 unload 메서드가 있으면 호출 후 GC를 수행한다.
    """
    target = (target_speaker or DEFAULT_SPEAKER).strip().lower()
    if target == current_engine.get("speaker_name"):
        return current_engine

    try:
        runtime = current_engine.get("runtime")
        if runtime is not None and hasattr(runtime.model, "unload"):
            runtime.model.unload()
    except Exception:
        pass

    current_engine.clear()
    gc.collect()
    print(
        f"[SBV2_WORKER] speaker 변경: {current_engine.get('speaker_name')} -> {target}",
        flush=True,
    )
    return load_engine(target)


# 워커 시작 시 기본 화자 엔진을 1회 로드한다.
ENGINE = load_engine(DEFAULT_SPEAKER)
# 클라이언트(`SBV2WorkerClient`)가 준비 완료를 감지하는 sentinel line.
print("__SBV2_READY__", flush=True)


for line in sys.stdin:
    # 한 줄 단위 JSON 프로토콜을 사용한다.
    line = line.strip()
    if not line:
        continue

    try:
        # 1) 요청 파싱
        payload = json.loads(line)

        text = payload["text"]
        style_index = payload.get("style", 0)
        style_weight = float(payload.get("style_weight", 1.0))
        speaker_name = payload.get("speaker_name", DEFAULT_SPEAKER)

        # 2) 필요 시 화자 엔진 교체
        previous_speaker = ENGINE.get("speaker_name")
        ENGINE = swap_engine_if_needed(ENGINE, speaker_name)
        if previous_speaker != ENGINE.get("speaker_name"):
            print(
                f"[SBV2_WORKER] speaker 변경 완료: {previous_speaker} -> {ENGINE.get('speaker_name')}",
                flush=True,
            )

        # 3) 스타일/화자 id 해석 후 추론
        runtime = ENGINE["runtime"]
        style_name = _resolve_style_name(runtime, style_index)
        speaker_key = ENGINE["speaker_name"]
        speaker_id = runtime.model.spk2id[speaker_key]

        sr, audio = runtime.model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=speaker_id,
            style=style_name,
            style_weight=style_weight,
            line_split=False,
        )

        # 4) 결과 WAV 저장 및 응답 반환
        out_path = OUT_DIR / f"tts_{abs(hash((speaker_key, style_name, text)))}.wav"
        sf.write(out_path, audio, sr)

        print(json.dumps({"wav_path": str(out_path)}), flush=True)

    except Exception as e:
        # 워커는 죽지 않고 에러를 JSON으로 반환한다.
        print(json.dumps({"error": str(e)}), flush=True)
