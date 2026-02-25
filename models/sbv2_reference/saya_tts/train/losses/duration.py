"""
Duration Loss

DP / SDP 예측이
실제 alignment와 얼마나 맞는지
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def duration_loss(
    logw_pred: torch.Tensor,
    logw_gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        logw_pred: [B, 1, T]
        logw_gt: [B, 1, T]
        mask: [B, 1, T]

    Returns:
        scalar loss
    """
    loss = (logw_pred - logw_gt) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)
