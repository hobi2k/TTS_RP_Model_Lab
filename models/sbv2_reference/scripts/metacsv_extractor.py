# import csv
# from pathlib import Path

# wav_dir = Path("wav")
# out_csv = Path("metadata.csv")

# with open(out_csv, "w", encoding="utf-8", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["wav", "text"])

#     for wav in sorted(wav_dir.glob("*.wav")):
#         name = wav.stem
#         text = name.split("_")[-1]
#         writer.writerow([wav.name, text])


from pathlib import Path

# 입력 wav 폴더
wav_dir = Path("/mnt/d/my_tts_dataset/wavs")

# 출력 text 폴더
text_dir = Path("/mnt/d/my_tts_dataset/text")
text_dir.mkdir(exist_ok=True)

for wav in sorted(wav_dir.glob("*.wav")):
    # wav 파일명 (확장자 제외)
    name = wav.stem

    # 파일명에서 마지막 '_' 이후를 대사로 사용
    # 예: 001_zonoko（ノーマル）_どうして私.wav → どうして私
    text = name.split("_")[-1]

    # 대응되는 txt 파일 경로
    txt_path = text_dir / f"{name}.txt"

    # 텍스트 파일 저장 (UTF-8)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)