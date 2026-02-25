"""
Mel Reconstruction Loss

역할:
- decoder가 만든 음성이
- 실제 음성과 얼마나 비슷한지 측정

핵심:
- L1 loss가 가장 안정적
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def mel_l1_loss(
    mel_pred: torch.Tensor,
    mel_gt: torch.Tensor,
    mel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        mel_pred: [B, n_mels, T]
            모델이 생성한 mel
        mel_gt:   [B, n_mels, T]
            GT mel
        mel_mask: [B, 1, T] or None
            padding frame 마스크 (있으면 적용)

    Returns:
        scalar loss
    """
    if mel_mask is not None:
        # 마스크 적용 (padding 프레임 제외)
        loss = torch.abs(mel_pred - mel_gt) * mel_mask
        return loss.sum() / mel_mask.sum().clamp_min(1.0)

    return F.l1_loss(mel_pred, mel_gt)
