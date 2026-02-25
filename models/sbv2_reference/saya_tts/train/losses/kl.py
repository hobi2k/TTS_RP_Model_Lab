"""
KL Divergence Loss

prior p(z|text) 와
posterior q(z|audio)를 정렬

Style-Bert-VITS 계열의 핵심 안정화 loss
"""

from __future__ import annotations
import torch


def kl_loss(
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        m_q, logs_q: posterior mean/log-std
        m_p, logs_p: prior mean/log-std
        mask: [B, 1, T]

    Returns:
        scalar loss
    """
    # KL(q || p) for diagonal Gaussian
    kl = (
        logs_p
        - logs_q
        + (torch.exp(2 * logs_q) + (m_q - m_p) ** 2)
        / (2 * torch.exp(2 * logs_p))
        - 0.5
    )

    kl = kl * mask
    return kl.sum() / mask.sum().clamp_min(1.0)
