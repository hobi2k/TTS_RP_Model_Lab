"""
enc_p 출력 → infer_after_enc_p 연결 디버깅 스크립트

목적:
- enc_p의 출력(EncPOut)이
- infer_after_enc_p의 입력으로
  논리/shape적으로 정확히 맞는지 검증
"""

import torch

from saya_tts.model.enc_p.enc_p import TextEncoderPrior
from saya_tts.engine.infer_after_enc_p import infer_after_enc_p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. enc_p 더미 생성 (앞서 만든 것과 동일)
    enc_p = TextEncoderPrior(
        num_phonemes=128,
        num_tones=8,
        num_languages=4,
        hidden_size=256,
        latent_dim=192,
        bert_dim=1024,
        num_layers=2,
        num_heads=4,
        style_dim=256,
        speaker_dim=128,
    ).to(device)
    enc_p.eval()

    # 2. dummy 입력
    B, T = 1, 5

    phoneme_ids = torch.tensor([[5, 12, 9, 0, 0]], device=device)
    tone_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    language_ids = torch.zeros((B, T), dtype=torch.long, device=device)

    bert_feats = torch.randn(B, 1024, T, device=device)
    style_vec = torch.randn(B, 256, device=device)
    g = torch.randn(B, 128, 1, device=device)

    with torch.no_grad():
        enc_out = enc_p(
            phoneme_ids=phoneme_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_feats=bert_feats,
            style_vec=style_vec,
            g=g,
        )

    # 3. net_g 더미
    """
    infer_after_enc_p는 net_g를 요구한다.

    net_g는 다음 속성을 반드시 가져야 한다:
    - dp   (deterministic duration predictor)
    - sdp  (stochastic duration predictor)
    - flow
    - dec

     지금 단계에서는 실제 net_g를 쓰지 않는다.
       따라서 이 스크립트는 "구조 설명용"이며,
       실제 실행은 여기서 멈춘다.
    """

    print("enc_p output ready.")
    print(f"x_enc   : {enc_out.x.shape}")
    print(f"m_p     : {enc_out.m_p.shape}")
    print(f"logs_p  : {enc_out.logs_p.shape}")
    print(f"x_mask  : {enc_out.x_mask.shape}")
    print(f"g       : {enc_out.g.shape}")

    print("\n이 EncPOut이 그대로 infer_after_enc_p(...)로 들어간다.")


if __name__ == "__main__":
    main()
