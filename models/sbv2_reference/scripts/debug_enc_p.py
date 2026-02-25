"""
enc_p(TextEncoderPrior) 단독 디버깅용 스크립트

이 스크립트의 목적:
- 실제 TTS 추론을 하지 않는다.
- enc_p의 입력/출력 shape와 내부 흐름이
  아키텍처 설계 의도대로 동작하는지만 확인한다.

즉:
"이 enc_p는 구조적으로 올바른가?"를 검증하는 용도다.
"""

import torch

from saya_tts.model.enc_p.enc_p import TextEncoderPrior


def main():
    # 1. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. enc_p 하이퍼파라미터 (임시 고정값)
    # 실제 값과 정확히 같을 필요는 없다.
    # 중요한 건 "차원 관계가 논리적으로 맞는지"다.
    B = 2              # batch size
    T = 6              # 텍스트 길이 (phoneme 개수)
    H = 256            # hidden size (Transformer 차원)
    D = 192            # latent size
    D_BERT = 1024      # 일본어 BERT hidden size
    STYLE_DIM = 256    # style vector 차원
    SPEAKER_DIM = 128  # speaker embedding 차원

    NUM_PHONEMES = 128
    NUM_TONES = 8
    NUM_LANGS = 4

    NUM_LAYERS = 2
    NUM_HEADS = 4

    # 3. enc_p 인스턴스 생성
    enc_p = TextEncoderPrior(
        num_phonemes=NUM_PHONEMES,
        num_tones=NUM_TONES,
        num_languages=NUM_LANGS,
        hidden_size=H,
        latent_dim=D,
        bert_dim=D_BERT,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        style_dim=STYLE_DIM,
        speaker_dim=SPEAKER_DIM,
        dropout=0.1,
    ).to(device)

    enc_p.eval()

    # 4. Dummy 입력 생성

    # (1) phoneme_ids
    # 0은 padding, 1~는 실제 음소라고 가정
    phoneme_ids = torch.tensor(
        [
            [5, 12, 9, 3, 7, 2],     # 길이 6
            [4, 11, 0, 0, 0, 0],     # 길이 2 + padding
        ],
        dtype=torch.long,
        device=device,
    )

    # (2) tone_ids
    # phoneme_ids와 동일한 shape
    tone_ids = torch.randint(
        low=0,
        high=NUM_TONES,
        size=(B, T),
        device=device,
    )

    # (3) language_ids
    # JP-Extra라면 실제로는 전부 같은 값일 가능성이 높음
    language_ids = torch.zeros(
        (B, T),
        dtype=torch.long,
        device=device,
    )

    # (4) bert_feats
    # 실제 BERT 출력 흉내
    # shape: [B, D_BERT, T]
    bert_feats = torch.randn(
        B,
        D_BERT,
        T,
        device=device,
    )

    # (5) style_vec
    # 감정/스타일 벡터
    style_vec = torch.randn(
        B,
        STYLE_DIM,
        device=device,
    )

    # (6) speaker embedding g
    # 보통 [B, gin, 1]
    g = torch.randn(
        B,
        SPEAKER_DIM,
        1,
        device=device,
    )

    # 5. enc_p forward 실행
    with torch.no_grad():
        out = enc_p(
            phoneme_ids=phoneme_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_feats=bert_feats,
            style_vec=style_vec,
            g=g,
        )

    # 6. 출력 검증 (핵심)
    print("enc_p DEBUG RESULT")

    print(f"x_enc shape   : {out.x.shape}       (expected: [B, H, T])")
    print(f"m_p shape     : {out.m_p.shape}     (expected: [B, D, T])")
    print(f"logs_p shape  : {out.logs_p.shape}  (expected: [B, D, T])")
    print(f"x_mask shape  : {out.x_mask.shape}  (expected: [B, 1, T])")
    print(f"g shape       : {out.g.shape}")

    # 7. padding mask 시각적 확인
    print("\nx_mask (padding check)")
    print(out.x_mask.squeeze(1))

    # 8. sanity check (assert)
    assert out.x.shape == (B, H, T)
    assert out.m_p.shape == (B, D, T)
    assert out.logs_p.shape == (B, D, T)
    assert out.x_mask.shape == (B, 1, T)

    # padding 위치에서 m_p/logs_p가 0인지 확인
    pad_positions = (phoneme_ids == 0).unsqueeze(1)
    assert torch.all(out.m_p[pad_positions] == 0)
    assert torch.all(out.logs_p[pad_positions] == 0)

    print("\nenc_p 구조 검증 완료")


if __name__ == "__main__":
    main()
