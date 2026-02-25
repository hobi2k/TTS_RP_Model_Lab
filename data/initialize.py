"""
Download required local models for data pipelines.

Usage:
  uv run data/initialize.py
  uv run data/initialize.py --force
  uv run data/initialize.py --only tri7b
  uv run data/initialize.py --only bge_m3_ko
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfHubHTTPError, snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_dir: Path
    description: str


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


# Required by current data pipelines:
# - Tri-7B: local generation model
# - BGE-m3-ko: embedding memory model (loaded via data/embedding/BGE-m3-ko)
MODEL_SPECS: dict[str, ModelSpec] = {
    "tri7b": ModelSpec(
        key="tri7b",
        repo_id="trillionlabs/Tri-7B",
        local_dir=DATA_DIR / "generator" / "Tri-7B",
        description="Local RP data generation model",
    ),
    "bge_m3_ko": ModelSpec(
        key="bge_m3_ko",
        repo_id="BAAI/BGE-m3",
        local_dir=DATA_DIR / "embedding" / "BGE-m3-ko",
        description="Embedding memory model for anti-repeat",
    ),
}


def _looks_downloaded(model_dir: Path) -> bool:
    # Fast sanity check for HF snapshots used in this project.
    return (model_dir / "config.json").exists() and (model_dir / "tokenizer.json").exists()


def _download_model(spec: ModelSpec, force: bool) -> None:
    spec.local_dir.mkdir(parents=True, exist_ok=True)

    if not force and _looks_downloaded(spec.local_dir):
        print(f"[SKIP] {spec.key}: already exists at {spec.local_dir}")
        return

    print(f"[DOWNLOAD] {spec.key}")
    print(f"  repo: {spec.repo_id}")
    print(f"  dir : {spec.local_dir}")
    print(f"  note: {spec.description}")

    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(spec.local_dir),
            local_dir_use_symlinks=False,
            revision="main",
        )
    except HfHubHTTPError as e:
        # Tri-7B is gated; this gives an actionable hint.
        if spec.key == "tri7b" and (e.response is not None and e.response.status_code in (401, 403)):
            raise RuntimeError(
                "Failed to download Tri-7B (gated/private access). "
                "Run `huggingface-cli login` with an approved account and retry."
            ) from e
        raise

    print(f"[DONE] {spec.key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required data-pipeline models.")
    parser.add_argument(
        "--only",
        choices=sorted(MODEL_SPECS.keys()),
        default=None,
        help="Download only one model key.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    targets = [MODEL_SPECS[args.only]] if args.only else [MODEL_SPECS["tri7b"], MODEL_SPECS["bge_m3_ko"]]
    for spec in targets:
        _download_model(spec, force=args.force)

    print("[ALL DONE] data model initialization completed.")


if __name__ == "__main__":
    main()
