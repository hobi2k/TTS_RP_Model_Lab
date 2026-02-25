"""
Loss placeholders

실제 사야 파인튜닝 시:
- mel loss
- KL loss
- duration loss
- adversarial loss (있다면)
등으로 교체된다.
"""

import torch


def dummy_loss(output: torch.Tensor) -> torch.Tensor:
    """
    지금 단계에서는 backward가 가능한 더미 loss만 있으면 된다.
    """
    return output.mean()
