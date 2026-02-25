# create_dataset_config.py
# ------------------------
# Style-Bert-VITS2 dataset용 config.json 생성 스크립트
#
# 이 파일은 preprocess_text.py에서 요구하는
# dataset/config.json 을 "최소 + 안전" 형태로 생성한다.
"""
python create_dataset_config.py \
  --dataset-root /mnt/d/my_tts_dataset/Saya \
  --speaker saya \
  --language JP
"""
  
import json
from pathlib import Path
import argparse
import soundfile as sf


def detect_sampling_rate(wavs_dir: Path) -> int:
    """
    wavs 디렉토리에서 첫 번째 wav 파일을 열어
    sampling_rate를 자동 추론한다.
    """
    for wav_path in wavs_dir.glob("*.wav"):
        with sf.SoundFile(wav_path) as f:
            return f.samplerate
    raise RuntimeError(f"No wav files found in {wavs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset config.json for Style-Bert-VITS2"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root directory (contains wavs/ and esd.list)",
    )
    parser.add_argument(
        "--speaker",
        default="saya",
        help="Default speaker name (will be updated by preprocess)",
    )
    parser.add_argument(
        "--language",
        default="JP",
        help="Dataset language (JP / EN / ZH)",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    wavs_dir = dataset_root / "raws"
    config_path = dataset_root / "config.json"

    if not wavs_dir.exists():
        raise FileNotFoundError(f"wavs directory not found: {wavs_dir}")

    sampling_rate = detect_sampling_rate(wavs_dir)

    config = {
        "data": {
            "sampling_rate": sampling_rate,
            "spk2id": {
                # preprocess_text.py가 덮어쓰므로 초기값은 의미 없음
                args.speaker: 0
            },
            "n_speakers": 1,
            "language": args.language,
        }
    }

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] config.json created at: {config_path}")
    print(f"     sampling_rate = {sampling_rate}")
    print("     spk2id / n_speakers will be updated by preprocess_text.py")


if __name__ == "__main__":
    main()
