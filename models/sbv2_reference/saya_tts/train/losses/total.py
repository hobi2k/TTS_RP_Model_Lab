"""
Total Loss Assembly

Stage별로 어떤 loss를 쓸지
가중치를 조절할 수 있게 설계
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

from .mel import mel_l1_loss
from .kl import kl_loss
from .duration import duration_loss


@dataclass
class LossWeights:
    mel: float = 45.0
    kl: float = 1.0
    duration: float = 1.0


def compute_total_loss(
    *,
    mel_pred: torch.Tensor,
    mel_gt: torch.Tensor,
    mel_mask: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    dur_pred: torch.Tensor | None,
    dur_gt: torch.Tensor | None,
    dur_mask: torch.Tensor | None,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict with individual losses + total
    """

    losses: dict[str, torch.Tensor] = {}

    losses["mel"] = mel_l1_loss(mel_pred, mel_gt, mel_mask)
    losses["kl"] = kl_loss(m_q, logs_q, m_p, logs_p, mel_mask)

    if dur_pred is not None and dur_gt is not None:
        losses["duration"] = duration_loss(
            dur_pred, dur_gt, dur_mask
        )
    else:
        losses["duration"] = torch.zeros_like(losses["mel"])

    total = (
        weights.mel * losses["mel"]
        + weights.kl * losses["kl"]
        + weights.duration * losses["duration"]
    )

    losses["total"] = total
    return losses
