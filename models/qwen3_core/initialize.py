"""
uv pip install --pre \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --no-cache-dir

python - << 'EOF'
import torch, torchvision
from torchvision.ops import nms

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("nms exists:", nms is not None)
EOF
"""
from __future__ import annotations

from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# 저장 경로(중요)
# - 이 파일이 models/qwen3_core/ 안에 있다고 가정한다.
# - model_assets도 qwen3_core/ 안에 고정한다.

QWEN3_CORE_DIR = Path(__file__).resolve().parent                 # .../models/qwen3_core
MODEL_ASSETS_DIR = QWEN3_CORE_DIR / "model_assets"              # .../models/qwen3_core/model_assets

MODELS = {
    # "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    # "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    # "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "YanoljaNEXT-EEVE-7B-v2": "yanolja/YanoljaNEXT-EEVE-Instruct-7B-v2-Preview",
}


def download_one(repo_id: str, local_subdir: str) -> Path:
    """
    Hugging Face Hub에서 모델 리포지토리 스냅샷을 받아
    지정한 model_assets 하위 폴더에 '실제 파일'로 저장한다.
    """
    local_dir = MODEL_ASSETS_DIR / local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] snapshot_download: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),        # 반드시 내가 지정한 경로
        local_dir_use_symlinks=False,    # 실제 파일로 저장 (중요)
        revision="main",
    )

    print(f"[INFO] tokenizer save_pretrained: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    tokenizer.save_pretrained(local_dir)

    print(f"[DONE] {repo_id} saved to: {local_dir}")
    return local_dir


def main() -> None:
    MODEL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for local_subdir, repo_id in MODELS.items():
        download_one(repo_id=repo_id, local_subdir=local_subdir)

    print("\n[ALL DONE] Qwen3 base models downloaded into qwen3_core/model_assets.")


if __name__ == "__main__":
    main()
