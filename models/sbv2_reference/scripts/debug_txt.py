"""
Debug Script
BERT feature extraction sanity check

이 스크립트의 목적:
- clean_text_for_tts()가 정상 동작하는지
- BERT feature가 infer.py 밖에서도 동일하게 생성되는지
- shape / device / 값 범위가 정상인지

※ 오직 BERT까지만 확인
"""

import torch

from style_bert_vits2.constants import Languages

from saya_tts.text.cleaner import clean_text_for_tts
from saya_tts.bert.extractor import BertFeatureExtractor


def debug_bert():
    # 1. 테스트 입력
    text = "こんにちは。私はサヤです。"
    language = Languages.JP
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("[BERT DEBUG]")
    print(f"text      : {text}")
    print(f"language  : {language}")
    print(f"device    : {device}")
    print("=" * 60)

    # 2. clean_text 단계
    phones, tones, word2ph = clean_text_for_tts(
        text=text,
        language=language,
    )

    print("[clean_text_for_tts]")
    print(f"phones len   : {len(phones)}")
    print(f"tones len    : {len(tones)}")
    print(f"word2ph len  : {len(word2ph)}")
    print(f"phones[:10]  : {phones[:10]}")
    print(f"tones[:10]   : {tones[:10]}")
    print("-" * 60)

    # 3. BERT feature 추출
    bert_extractor = BertFeatureExtractor(
        language=language,
        device=device,
    )

    bert_feats = bert_extractor.extract(
        phones=phones,
        word2ph=word2ph,
    )

    # 4. BERT feature 디버깅 출력
    print("[BERT FEATURES]")
    print(f"type         : {type(bert_feats)}")
    print(f"dtype        : {bert_feats.dtype}")
    print(f"device       : {bert_feats.device}")
    print(f"shape        : {tuple(bert_feats.shape)}")
    print(f"D_bert       : {bert_feats.shape[0]}")
    print(f"T_phone      : {bert_feats.shape[1]}")
    print(f"phones len   : {len(phones)}")
    print("-" * 60)

    # 5. 값 범위 체크 (NaN / Inf)
    print("[VALUE CHECK]")
    print(f"min value    : {bert_feats.min().item():.6f}")
    print(f"max value    : {bert_feats.max().item():.6f}")
    print(f"mean value   : {bert_feats.mean().item():.6f}")
    print(f"has NaN      : {torch.isnan(bert_feats).any().item()}")
    print(f"has Inf      : {torch.isinf(bert_feats).any().item()}")
    print("=" * 60)

    # 6. 필수 조건 검사 (assert)
    assert bert_feats.ndim == 2, "bert_feats must be 2D"
    assert bert_feats.shape[1] == len(phones), (
        "T_phone mismatch between phones and bert_feats"
    )
    assert bert_feats.device.type == device, "device mismatch"

    print("BERT feature extraction OK")


if __name__ == "__main__":
    debug_bert()