from __future__ import annotations

import torch
import torch.nn as nn


class PriorHead(nn.Module):
    """
    enc_p의 마지막 블록: Prior Head

    역할:
    - Transformer Encoder의 출력(x_enc)을 받아
    - 잠재 변수 z_p의 분포 파라미터
      (mean=m_p, log-std=logs_p)를 생성한다.

    입력:
    - x_enc: [B, H, T]

    출력:
    - m_p: [B, D, T]
    - logs_p: [B, D, T]
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
    ):
        super().__init__()

        # 평균(mean) 생성용 1x1 Conv
        # 각 시간 프레임마다 H차원 feature를 받아
        # D차원의 평균 벡터로 변환
        self.proj_mean = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=latent_dim,
            kernel_size=1,
        )

        # 로그 표준편차(log-std) 생성용 1x1 Conv
        # mean과 완전히 독립된 파라미터를 갖게 한다
        self.proj_logstd = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=latent_dim,
            kernel_size=1,
        )

        # 초기화 관련 메모
        # log-std는 너무 큰 값으로 시작하면
        # 샘플링 노이즈가 과해질 수 있다.
        #
        # 보통 bias를 음수 쪽으로 초기화해서
        # 초반에는 작은 분산으로 시작하게 만드는 경우도 있다.
        nn.init.zeros_(self.proj_logstd.weight)
        nn.init.zeros_(self.proj_logstd.bias)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_enc:
                [B, H, T]
                Transformer를 통과한 텍스트 인코딩

            x_mask:
                [B, 1, T]
                padding 마스크

        Returns:
            m_p:
                [B, D, T]
                prior 평균

            logs_p:
                [B, D, T]
                prior log-표준편차
        """

        # (1) 평균/로그표준편차 계산
        m_p = self.proj_mean(x_enc) # [B, D, T]
        logs_p = self.proj_logstd(x_enc) # [B, D, T]

        # (2) padding 위치 무력화
        # padding 토큰 위치에서는
        # prior 분포가 의미 없으므로 0으로 눌러준다.
        m_p = m_p * x_mask
        logs_p = logs_p * x_mask

        return m_p, logs_p
