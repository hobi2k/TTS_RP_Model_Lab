"""
enc_p 동치성 디버깅 harness

목표:
- Style-Bert-VITS2 실전 모델(net_g.enc_p)
- 내가 구현한 TextEncoderPrior

두 enc_p에 "완전히 동일한 입력"을 넣고,
출력의 shape / 분포 스케일 / 마스크를 비교한다.

중요:
- 이 스크립트는 실전 코드 기반이다.
- infer.py를 우회하지 않고, net_g를 그대로 사용한다.
"""

from __future__ import annotations

import torch
from pathlib import Path

# 실전 runner (이미 만든 것)
from saya_tts.model.runner import StyleBertVITS2Runner

# 네가 구현한 enc_p
from saya_tts.model.enc_p.enc_p import TextEncoderPrior


def tensor_stats(name: str, x: torch.Tensor):
    """
    텐서 분포를 간단히 요약 출력.
    - 평균, 표준편차, min/max
    """
    x_float = x.float()
    print(
        f"{name:>12s} | "
        f"shape={tuple(x.shape)} | "
        f"mean={x_float.mean():+.4f} | "
        f"std={x_float.std():+.4f} | "
        f"min={x_float.min():+.4f} | "
        f"max={x_float.max():+.4f}"
    )


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 실전 모델 로드 (net_g 확보)
    runner = StyleBertVITS2Runner(
        model_dir=Path("model_assets/base_jp_extra"),  # base JP-Extra 모델
        device=device,
    )
    net_g = runner.net_g
    net_g.eval()

    # 2. infer.py가 실제로 만드는 입력을 재현
    # 여기서는 단순화를 위해 dummy 입력을 직접 만든다.
    # 목적은 "동일 입력 -> enc_p 비교" 이지,
    # 자연스러운 발화가 아니다.

    B = 1
    T = 8

    phoneme_ids = torch.tensor([[5, 12, 9, 7, 3, 0, 0, 0]], device=device)
    tone_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    language_ids = torch.zeros((B, T), dtype=torch.long, device=device)

    # JP-Extra BERT feature: [B, 1024, T]
    bert_feats = torch.randn(B, 1024, T, device=device)

    # style vector (style_vectors.npy에서 뽑았다고 가정)
    style_vec = torch.from_numpy(
        runner.get_style_vec_by_name("Neutral")
    ).unsqueeze(0).to(device)

    # speaker embedding (net_g 내부와 동일 방식)
    # JP-Extra 기준: emb_g(sid) -> [B, gin] -> unsqueeze(-1)
    sid = torch.tensor([0], device=device)
    g = net_g.emb_g(sid).unsqueeze(-1)

    # 3. 실전 enc_p 실행
    enc_real = net_g.enc_p(
        phoneme_ids,
        phoneme_ids.ne(0).sum(dim=1),
        tone_ids,
        language_ids,
        bert_feats,
        style_vec,
        g=g,
    )

    # 4. 내가 구현한 enc_p 실행
    enc_mine = TextEncoderPrior(
        num_phonemes=net_g.enc_p.num_phonemes,
        num_tones=net_g.enc_p.num_tones,
        num_languages=net_g.enc_p.num_languages,
        hidden_size=net_g.enc_p.hidden_size,
        latent_dim=net_g.enc_p.latent_dim,
        bert_dim=1024,
        num_layers=net_g.enc_p.num_layers,
        num_heads=net_g.enc_p.num_heads,
        style_dim=style_vec.shape[1],
        speaker_dim=g.shape[1],
    ).to(device)

    enc_mine.eval()

    out_mine = enc_mine(
        phoneme_ids=phoneme_ids,
        tone_ids=tone_ids,
        language_ids=language_ids,
        bert_feats=bert_feats,
        style_vec=style_vec,
        g=g,
    )

    # 5. 결과 비교 출력
    print("\nenc_p output comparison\n")

    tensor_stats("x_real", enc_real[0])
    tensor_stats("x_mine", out_mine.x)
    print()

    tensor_stats("m_p_real", enc_real[1])
    tensor_stats("m_p_mine", out_mine.m_p)
    print()

    tensor_stats("logs_p_real", enc_real[2])
    tensor_stats("logs_p_mine", out_mine.logs_p)
    print()

    tensor_stats("x_mask_real", enc_real[3])
    tensor_stats("x_mask_mine", out_mine.x_mask)
    print()

    # 6. 차이 정량 확인 (L2 norm)
    def l2_diff(a, b):
        return torch.norm(a.float() - b.float()).item()

    print("\nL2 difference")
    print("x      :", l2_diff(enc_real[0], out_mine.x))
    print("m_p    :", l2_diff(enc_real[1], out_mine.m_p))
    print("logs_p :", l2_diff(enc_real[2], out_mine.logs_p))
    print("x_mask :", l2_diff(enc_real[3], out_mine.x_mask))


if __name__ == "__main__":
    main()
