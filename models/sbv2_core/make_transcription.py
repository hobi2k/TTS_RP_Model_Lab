from pathlib import Path

# 사용자 설정
DATA_DIR = Path("/mnt/d/my_tts_dataset/mai")          # Data/모델명 경로에서 실행한다고 가정
WAV_DIR = DATA_DIR / "raws"
TEXT_DIR = DATA_DIR / "text"
OUT_FILE = DATA_DIR / "esd.list"

SPEAKER_ID = "mai"           # 화자 ID (단일 화자)
LANGUAGE = "JP"               # 일본어

# 생성 로직
lines = []

for text_file in sorted(TEXT_DIR.glob("*.txt")):
    utt = text_file.stem
    wav_path = WAV_DIR / f"{utt}.wav"

    if not wav_path.exists():
        raise FileNotFoundError(f"Missing wav file: {wav_path}")

    text = text_file.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Empty text file: {text_file}")

    # Style-Bert-VITS2 표준 포맷
    line = f"{wav_path.name}|{SPEAKER_ID}|{LANGUAGE}|{text}"
    lines.append(line)

# 파일 저장
OUT_FILE.write_text("\n".join(lines), encoding="utf-8")

print(f"[OK] Generated {OUT_FILE} ({len(lines)} lines)")
