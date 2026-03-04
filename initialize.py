"""
프로젝트 전체 모델 초기화 스크립트.

현재 프로젝트 구조에 맞는 모델 자산을 내려받고,
베이스 모델과 파인튜닝된 런타임 체크포인트를 분리 저장한다.

사용 예시:
  uv run initialize.py
  uv run initialize.py --only runtime_llm_v3
  uv run initialize.py --only tts_saya_v2
  uv run initialize.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
QWEN_ASSETS_DIR = MODELS_DIR / "qwen3_core" / "model_assets"
SBV2_DIR = MODELS_DIR / "Style-BERT-VITS2"
SBV2_ASSETS_DIR = SBV2_DIR / "model_assets"
SBV2_BERT_MODELS_JSON = SBV2_DIR / "bert" / "bert_models.json"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_dir: Path
    description: str
    kind: str = "snapshot"


SPECS: dict[str, ModelSpec] = {
    # data 경로
    "tri7b": ModelSpec(
        key="tri7b",
        repo_id="trillionlabs/Tri-7B",
        local_dir=DATA_DIR / "generator" / "Tri-7B",
        description="로컬 RP 데이터 생성 모델",
    ),
    "bge_m3_ko": ModelSpec(
        key="bge_m3_ko",
        repo_id="BAAI/BGE-m3",
        local_dir=DATA_DIR / "embedding" / "BGE-m3-ko",
        description="메모리 / 반복 억제용 임베딩 모델",
    ),
    # Qwen 계열 베이스 모델(파인튜닝 체크포인트와 분리 저장)
    "base_llm_7b": ModelSpec(
        key="base_llm_7b",
        repo_id="yanolja/YanoljaNEXT-EEVE-Instruct-7B-v2-Preview",
        local_dir=QWEN_ASSETS_DIR / "YanoljaNEXT-EEVE-Instruct-7B",
        description="사야 RP 7B 계열 학습에 쓰인 공개 7B 베이스 모델",
    ),
    "base_translator_1_7b": ModelSpec(
        key="base_translator_1_7b",
        repo_id="Qwen/Qwen3-1.7B-Base",
        local_dir=QWEN_ASSETS_DIR / "qwen3-1.7b-base",
        description="번역기 계열 학습에 쓰인 공개 1.7B 베이스 모델",
    ),
    # 실제 런타임에서 바로 쓰는 파인튜닝 모델
    "runtime_llm_v3": ModelSpec(
        key="runtime_llm_v3",
        repo_id="ahnhs2k/saya_rp_7b_v3",
        local_dir=QWEN_ASSETS_DIR / "saya_rp_7b_v3",
        description="현재 7B RP 런타임 모델",
    ),
    "runtime_llm_v2": ModelSpec(
        key="runtime_llm_v2",
        repo_id="ahnhs2k/saya_rp_7b_v2",
        local_dir=QWEN_ASSETS_DIR / "saya_rp_7b_v2",
        description="이전 7B RP 런타임 모델",
    ),
    "runtime_llm_4b_v2": ModelSpec(
        key="runtime_llm_4b_v2",
        repo_id="ahnhs2k/saya_rp_4b_v2",
        local_dir=QWEN_ASSETS_DIR / "saya_rp_4b_v2",
        description="현재 4B RP 런타임 모델",
    ),
    "runtime_translator_v2": ModelSpec(
        key="runtime_translator_v2",
        repo_id="ahnhs2k/qtranslator_1.7b_v2",
        local_dir=QWEN_ASSETS_DIR / "qtranslator_1.7b_v2",
        description="현재 번역기 런타임 모델",
    ),
    "runtime_translator_v1": ModelSpec(
        key="runtime_translator_v1",
        repo_id="ahnhs2k/qtranslator_1.7b",
        local_dir=QWEN_ASSETS_DIR / "qtranslator_1.7b",
        description="이전 번역기 런타임 모델",
    ),
    # Style-BERT-VITS2 화자 모델
    "tts_saya_v2": ModelSpec(
        key="tts_saya_v2",
        repo_id="ahnhs2k/sbv2_saya_v2",
        local_dir=SBV2_ASSETS_DIR / "saya",
        description="현재 saya ONNX TTS 자산",
        kind="tts",
    ),
    "tts_mai_v2": ModelSpec(
        key="tts_mai_v2",
        repo_id="ahnhs2k/sbv2_mai_v2",
        local_dir=SBV2_ASSETS_DIR / "mai",
        description="현재 mai ONNX TTS 자산",
        kind="tts",
    ),
}


TTS_FILENAME_MAP = {
    "tts_saya_v2": "saya_e150_s57000.onnx",
    "tts_mai_v2": "mai_e100_s38000.onnx",
}


DEFAULT_TARGETS = [
    "tri7b",
    "bge_m3_ko",
    "base_llm_7b",
    "base_translator_1_7b",
    "runtime_llm_v3",
    "runtime_translator_v2",
    "tts_saya_v2",
    "tts_mai_v2",
]


def _looks_downloaded(target_dir: Path, kind: str) -> bool:
    if kind == "tts":
        return (target_dir / "config.json").exists() and (target_dir / "style_vectors.npy").exists()
    return (target_dir / "config.json").exists() and any(
        (target_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors", "pytorch_model.bin")
    )


def _download_snapshot(spec: ModelSpec, force: bool) -> None:
    target_dir = spec.local_dir

    if not force and _looks_downloaded(target_dir, spec.kind):
        print(f"[SKIP] {spec.key}: already exists at {target_dir}")
        return

    if force and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] {spec.key}")
    print(f"  repo: {spec.repo_id}")
    print(f"  dir : {target_dir}")
    print(f"  note: {spec.description}")

    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            revision="main",
        )
    except HfHubHTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in (401, 403):
            raise RuntimeError(
                f"{spec.repo_id} 다운로드에 실패했다. "
                "이 리포지토리는 접근 권한이 필요하다. "
                "승인된 계정으로 `huggingface-cli login` 후 다시 시도하거나, "
                "`--only`로 공개 모델만 선택해서 먼저 내려받아라."
            ) from exc
        raise

    print(f"[DONE] {spec.key}")


def _find_single_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        preview = ", ".join(str(path.relative_to(root)) for path in matches[:5])
        raise RuntimeError(f"{root} 아래에서 {pattern} 패턴에 여러 파일이 잡혔다: {preview}")
    return matches[0]


def _normalize_tts_files(spec: ModelSpec) -> None:
    target_dir = spec.local_dir

    config_file = _find_single_file(target_dir, "config.json")
    style_vectors_file = _find_single_file(target_dir, "style_vectors.npy")
    onnx_file = _find_single_file(target_dir, "*.onnx")

    if config_file is None:
        raise FileNotFoundError(f"{target_dir}에서 config.json을 찾지 못했다")
    if style_vectors_file is None:
        raise FileNotFoundError(f"{target_dir}에서 style_vectors.npy를 찾지 못했다")
    if onnx_file is None:
        raise FileNotFoundError(f"{target_dir}에서 .onnx 모델 파일을 찾지 못했다")

    top_config = target_dir / "config.json"
    top_style_vectors = target_dir / "style_vectors.npy"
    target_onnx = target_dir / TTS_FILENAME_MAP[spec.key]

    if config_file != top_config:
        shutil.move(str(config_file), str(top_config))
    if style_vectors_file != top_style_vectors:
        shutil.move(str(style_vectors_file), str(top_style_vectors))
    if onnx_file != target_onnx:
        shutil.move(str(onnx_file), str(target_onnx))


def _download_one(spec: ModelSpec, force: bool) -> None:
    _download_snapshot(spec, force=force)
    if spec.kind == "tts":
        _normalize_tts_files(spec)


def _ensure_sbv2_bert_runtime_assets(force: bool) -> None:
    if not SBV2_BERT_MODELS_JSON.exists():
        raise FileNotFoundError(f"SBV2 BERT 모델 목록 파일이 없습니다: {SBV2_BERT_MODELS_JSON}")

    with SBV2_BERT_MODELS_JSON.open(encoding="utf-8") as fp:
        models = json.load(fp)

    # 현재 워커는 일본어 tokenizer + ONNX BERT만 사용한다.
    required_keys = (
        "deberta-v2-large-japanese-char-wwm",
        "deberta-v2-large-japanese-char-wwm-onnx",
    )

    for key in required_keys:
        spec = models[key]
        target_dir = SBV2_DIR / "bert" / key
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in spec["files"]:
            target_file = target_dir / filename
            if force and target_file.exists():
                target_file.unlink()
            if target_file.exists():
                continue
            print(f"[DOWNLOAD] sbv2_bert:{key}:{filename}")
            hf_hub_download(
                repo_id=spec["repo_id"],
                filename=filename,
                local_dir=str(target_dir),
            )
            print(f"[DONE] sbv2_bert:{key}:{filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="프로젝트 모델을 data/ 와 models/ 아래에 내려받는다.")
    parser.add_argument(
        "--only",
        choices=["all", *sorted(SPECS.keys())],
        default="all",
        help="하나의 대상 키만 내려받는다.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 파일이 있어도 강제로 다시 내려받는다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_keys = [args.only] if args.only != "all" else DEFAULT_TARGETS

    for key in target_keys:
        _download_one(SPECS[key], force=args.force)

    if any(SPECS[key].kind == "tts" for key in target_keys):
        _ensure_sbv2_bert_runtime_assets(force=args.force)

    print("[ALL DONE] initialize.py completed.")


if __name__ == "__main__":
    main()
