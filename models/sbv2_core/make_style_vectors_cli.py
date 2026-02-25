#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI script:
- Generate style vectors (.wav.npy) if missing
- Create style_vectors.npy from subdirectories (angry/sad/happy)
- Update model_assets/tts/Saya/config.json

No Gradio. No clustering. No visualization.


python make_style_vectors_cli.py \
  --model-dir model_assets/tts/Saya \
  --audio-dir /mnt/d/my_tts_dataset/Saya_style \
  --workers 8

or 
export CUDA_VISIBLE_DEVICES=""
python make_style_vectors_cli.py \
  --model-dir model_assets/tts/Saya \
  --audio-dir /mnt/d/my_tts_dataset/Saya_style \
  --workers 8  
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm

from default_style import save_styles_by_dirs
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_gen import save_style_vector


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


def collect_audio_files(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    ]


def ensure_npy(path: Path):
    npy_path = path.with_name(path.name + ".npy")
    if npy_path.exists():
        return None
    save_style_vector(str(path))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="model_assets/tts/Saya",
        help="Model directory containing config.json",
    )
    parser.add_argument(
        "--audio-dir",
        default="/mnt/d/my_tts_dataset/Saya_style",
        help="Root directory with style subfolders",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    audio_dir = Path(args.audio_dir)

    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(audio_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    audio_files = collect_audio_files(audio_dir)
    if not audio_files:
        raise RuntimeError("No audio files found")

    logger.info(f"Found {len(audio_files)} audio files")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(
            tqdm(
                executor.map(ensure_npy, audio_files),
                total=len(audio_files),
                file=SAFE_STDOUT,
                desc="Generating .wav.npy",
                dynamic_ncols=True,
            )
        )

    style_vectors_path = model_dir / "style_vectors.npy"
    if style_vectors_path.exists():
        shutil.copy(style_vectors_path, style_vectors_path.with_suffix(".npy.bak"))

    shutil.copy(config_path, config_path.with_suffix(".json.bak"))

    save_styles_by_dirs(
        wav_dir=audio_dir,
        output_dir=model_dir,
        config_path=config_path,
        config_output_path=config_path,
    )

    logger.success("Style vectors generated successfully")


if __name__ == "__main__":
    main()
