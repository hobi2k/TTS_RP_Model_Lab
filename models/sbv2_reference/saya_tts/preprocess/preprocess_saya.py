"""
사야 TTS 데이터 전처리 파이프라인 (JP-Extra 기준)

역할:
- raw wav + text 를 입력으로 받아
- Style-Bert-VITS2 학습용 데이터셋 포맷으로 변환한다.

이 스크립트는 '훈련 이전에 1회' 실행된다.
"""

from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

# 내부 모듈
from saya_tts.preprocess.text.japanese_g2p import japanese_g2p
from saya_tts.preprocess.text.text_cleaner import clean_text
from saya_tts.preprocess.audio.mel import wav_to_mel
from style_bert_vits2.nlp.bert_models import load_bert_model, extract_bert_feature


# 설정
DATA_ROOT = Path("data/saya")
RAW_WAV_DIR = Path("raw_data/wavs")     # 원본 wav
RAW_META = Path("raw_data/metadata.csv")

# BERT 설정 (JP-Extra)
BERT_MODEL_NAME = "ku-nlp/deberta-v2-large-japanese-char-wwm"

# mel 설정 (config.json과 반드시 일치)
SAMPLE_RATE = 44100
N_MELS = 80


def main():
    # 출력 디렉토리 생성
    for d in ["wavs", "mels", "bert", "durations", "text"]:
        (DATA_ROOT / d).mkdir(parents=True, exist_ok=True)

    # BERT 로드 (1회)
    bert_tokenizer, bert_model = load_bert_model(
        BERT_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # metadata.csv 읽기
    with open(RAW_META, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 새 metadata 작성용
    out_meta = []

    for row in tqdm(rows):
        utt_id = row["utt_id"]
        text = row["text"]
        style_id = row.get("style_id", 0)
        speaker_id = row.get("speaker_id", 0)

    
        # 1. 텍스트 정규화    
        clean = clean_text(text)

    
        # 2. G2P (일본어)    
        g2p_out = japanese_g2p(clean)

        phonemes = g2p_out["phonemes"] # list[int]
        tones = g2p_out["tones"] # list[int]
        langs = g2p_out["langs"] # list[int]
        word2ph = g2p_out["word2ph"] # list[int]

        # 저장
        np.save(DATA_ROOT / "text" / f"{utt_id}_phoneme.npy", np.array(phonemes))
        np.save(DATA_ROOT / "text" / f"{utt_id}_tone.npy", np.array(tones))
        np.save(DATA_ROOT / "text" / f"{utt_id}_lang.npy", np.array(langs))

    
        # 3. BERT feature 추출    
        bert_feat = extract_bert_feature(
            text=clean,
            tokenizer=bert_tokenizer,
            model=bert_model,
        )
        # shape: [D_bert, T_text]

        np.save(DATA_ROOT / "bert" / f"{utt_id}.npy", bert_feat)

    
        # 4. wav -> mel
    
        wav, sr = sf.read(RAW_WAV_DIR / f"{utt_id}.wav")
        assert sr == SAMPLE_RATE, f"SR mismatch: {sr}"

        mel = wav_to_mel(wav, sr)
        np.save(DATA_ROOT / "mels" / f"{utt_id}.npy", mel)

    
        # 5) duration 계산
    
        # word2ph를 frame 단위 duration으로 변환
        # (Style-Bert-VITS2 방식)
        # word2ph_to_duration 미구현
        durations = word2ph_to_duration(word2ph, mel.shape[1])
        np.save(DATA_ROOT / "durations" / f"{utt_id}.npy", durations)

        # wav 복사
        sf.write(DATA_ROOT / "wavs" / f"{utt_id}.wav", wav, sr)

        out_meta.append(
            {
                "utt_id": utt_id,
                "text": text,
                "style_id": style_id,
                "speaker_id": speaker_id,
            }
        )

    # metadata.csv 저장
    with open(DATA_ROOT / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["utt_id", "text", "style_id", "speaker_id"],
        )
        writer.writeheader()
        writer.writerows(out_meta)


if __name__ == "__main__":
    main()
