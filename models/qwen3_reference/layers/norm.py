# models/qwen/layers/norm.py
from __future__ import annotations

import torch
from torch import nn


class Qwen2RMSNorm(nn.Module):
    """
    Qwen2 RMSNorm

    HuggingFace Qwen2에서 사용하는 RMSNorm 구현과
    수식·파라미터·이름을 그대로 맞춘 버전이다.

    RMSNorm의 핵심 특징:
      - mean을 빼지 않는다 (LayerNorm과의 차이점)
      - 분산(variance)만 사용한다
      - scale 파라미터(weight)만 있고 bias는 없다

    수식:
        y = x * rsqrt(mean(x^2) + eps) * weight

    HF Qwen2와의 가중치 호환 포인트:
      1) 파라미터 이름이 반드시 `weight`여야 한다
      2) shape = (hidden_size,)
      3) eps 값이 config.rms_norm_eps와 동일해야 한다
      4) fp16/bf16 안정성을 위해 내부 계산은 fp32로 수행
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size:
                hidden dimension 크기 (예: 4096)
            eps:
                수치 안정성을 위한 epsilon
                - HF Qwen2 기본값: 1e-6
        """
        super().__init__()

        # HF Qwen2와 동일: bias 없음, weight만 존재
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                (B, S, hidden_size) 또는 (..., hidden_size)

        Returns:
            normalized hidden_states, same shape as input
        """

        # 입력 dtype 저장 (fp16 / bf16 / fp32)
        input_dtype = hidden_states.dtype

        # RMSNorm은 분산 계산이 핵심이므로
        # fp16/bf16에서의 underflow/overflow를 피하기 위해
        # HF 구현처럼 float32로 캐스팅해서 계산한다.
        hidden_states = hidden_states.to(torch.float32)

        # variance = mean(x^2)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)

        # normalize
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )

        # scale 적용 후 원래 dtype으로 복귀
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self) -> str:
        # 디버깅 / print(model) 시 가독성용
        return f"hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon}"
