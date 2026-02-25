from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Conditioning 모듈: AdaLayerNorm (Adaptive LayerNorm)
class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (조건 기반 LayerNorm)

    목적:
    - 일반 LayerNorm은 "각 토큰의 feature를 평균0/분산1로 정규화"만 한다.
    - AdaLayerNorm은 정규화 이후에, 조건(style/speaker)으로부터 만든
      scale(확대)와 shift(이동)를 적용해서 feature를 조절한다.

    입력/출력 shape:
    - x: [B, T, H]
    - cond: [B, C]  (예: style_vec, 혹은 style+speaker를 합친 벡터)
    - out: [B, T, H]

    수식 개념:
    - y = LN(x)
    - (gamma, beta) = f(cond)
    - out = y * (1 + gamma) + beta
      여기서 (1 + gamma)를 쓰는 이유는 초기값에서 안정적으로 동작하게 만들기 위함(자주 쓰는 트릭)
    """
    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cond_size = cond_size

        # LayerNorm은 feature 차원(H)에 대해 정규화한다.
        self.ln = nn.LayerNorm(hidden_size)

        # 조건 벡터(cond)를 받아서 gamma/beta를 만든다.
        # 출력 크기 = 2H (앞 H개는 gamma, 뒤 H개는 beta)
        self.cond_to_affine = nn.Linear(cond_size, 2 * hidden_size)

        # 초기에는 cond가 feature를 과하게 흔들지 않도록 0에 가깝게 시작하는 편이 안전하다.
        nn.init.zeros_(self.cond_to_affine.weight)
        nn.init.zeros_(self.cond_to_affine.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        # cond: [B, C]
        y = self.ln(x)  # [B, T, H]

        affine = self.cond_to_affine(cond)  # [B, 2H]
        gamma, beta = affine.chunk(2, dim=-1)  # 각각 [B, H], [B, H]

        # [B, H] -> [B, 1, H] 로 바꿔서 time축(T)에 broadcast
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        out = y * (1.0 + gamma) + beta
        return out


# 2. FeedForward(FFN) 모듈
class FeedForward(nn.Module):
    """
    Transformer의 FFN(2-layer MLP)

    입력/출력:
    - x: [B, T, H]
    - out: [B, T, H]

    구성:
    - Linear(H -> 4H) -> 활성화 -> Dropout -> Linear(4H -> H)
    """
    def __init__(self, hidden_size: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = hidden_size * ff_mult

        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x) # GELU는 Transformer 계열에서 표준에 가깝다.
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. Transformer Encoder Layer (Self-Attention + FFN)
class TransformerEncoderLayer(nn.Module):
    """
    enc_p 내부 Transformer layer 1개.

    입력:
    - x: [B, H, T]  (주의: 우리는 enc_p 파이프라인에 맞춰 [B,H,T]를 유지)
    - x_mask: [B, 1, T]  (유효 토큰=1, padding=0)
    - cond: [B, C]  (조건 벡터: style_vec + speaker_vec 등을 합쳐 만든 것)

    출력:
    - x: [B, H, T]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # PyTorch MultiheadAttention은 기본 입력 형태 [T, B, H] 또는 [B, T, H]를 지원한다.
        # 여기서는 batch_first=True를 사용해서 [B, T, H]로 쓰겠다.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 조건 기반 정규화 (Attention 앞/FFN 앞 각각)
        self.ada_ln1 = AdaLayerNorm(hidden_size, cond_size)
        self.ada_ln2 = AdaLayerNorm(hidden_size, cond_size)

        # FFN
        self.ffn = FeedForward(hidden_size, ff_mult=4, dropout=dropout)

        # Dropout (Residual 연결 전에 한 번 더 안정화)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        # (0) shape 정리: [B, H, T] -> [B, T, H]
        x = x.transpose(1, 2)  # [B, T, H]

        # (1) Attention block
        # (1-1) 조건 기반 LayerNorm
        # - cond가 attention 입력 feature를 조절한다.
        x_norm = self.ada_ln1(x, cond)  # [B, T, H]

        # (1-2) key_padding_mask 준비
        # MultiheadAttention의 key_padding_mask는:
        # - shape: [B, T]
        # - True  = "mask(무시)" 해야 하는 위치 (padding)
        # - False = 유효 토큰
        #
        # x_mask는 유효=1/pad=0 이므로, padding 위치는 (x_mask==0)이다.
        key_padding_mask = (x_mask.squeeze(1) == 0)  # [B, T] (bool)

        # (1-3) self-attention 수행
        # - query/key/value 모두 같은 x_norm (self-attention)
        attn_out, _attn_weights = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,  # 가중치가 필요하면 True로 바꿔도 됨(디버깅/시각화 용)
        )  # attn_out: [B, T, H]

        # (1-4) residual 연결
        x = x + self.dropout(attn_out)

        # (2) FFN block
        # (2-1) 조건 기반 LayerNorm
        x_norm = self.ada_ln2(x, cond)  # [B, T, H]

        # (2-2) FFN
        ffn_out = self.ffn(x_norm)  # [B, T, H]

        # (2-3) residual 연결
        x = x + self.dropout(ffn_out)

        # (3) mask 후처리 (선택이지만 안정성에 도움)
        # padding 위치는 모델 내부에서 계산에 참여하지 않는 게 이상적이다.
        # attention에서 이미 막았더라도, residual로 인해 작은 값이 흘러갈 수 있으니
        # 여기서 한 번 더 0으로 눌러주는 패턴을 종종 쓴다.
        x = x * x_mask.transpose(1, 2)  # [B, T, H] * [B, T, 1] -> [B, T, H]

        # (4) 다시 [B, H, T]로 복구
        x = x.transpose(1, 2)  # [B, H, T]
        return x


# 4. Transformer Encoder Stack (N층 쌓기)
class TransformerEncoderStack(nn.Module):
    """
    enc_p 내부의 Transformer layer들을 N개 쌓은 스택.

    입력:
    - x: [B, H, T]
    - x_mask: [B, 1, T]
    - style_vec: [B, S]
    - g: [B, gin, 1] 또는 [B, gin] (모델마다 다를 수 있음)

    출력:
    - x_enc: [B, H, T]
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        style_dim: int,
        speaker_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # (A) 조건 벡터(cond) 구성
        # style_vec([B,S])와 speaker embedding g([B,gin,1])을
        # 하나의 cond 벡터([B, C])로 합쳐서 AdaLayerNorm에 넣는다.
        #
        # 왜 합치나?
        # - Layer마다 style/speaker를 따로 넣을 수도 있지만,
        #   우선은 "조건 하나"로 통일하면 구현/디버깅이 쉬워진다.
        self.style_dim = style_dim
        self.speaker_dim = speaker_dim
        self.cond_dim = style_dim + speaker_dim

        # speaker embedding g가 [B, gin, 1]로 들어올 때 [B, gin]으로 펴기 위한 용도
        # (이 레이어 자체는 파라미터가 없지만, 코드를 명확히 하기 위해 분리)
        self._dummy = nn.Identity()

        # (B) Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    cond_size=self.cond_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def build_cond(self, style_vec: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        style_vec + speaker embedding을 결합해 cond 벡터를 만든다.

        입력:
        - style_vec: [B, S]
        - g: [B, gin, 1] 또는 [B, gin]

        출력:
        - cond: [B, S+gin]
        """
        # g가 [B, gin, 1]이면 마지막 차원(1)을 제거해 [B, gin]으로 만든다.
        if g.ndim == 3:
            g_flat = g.squeeze(-1)  # [B, gin]
        else:
            g_flat = g  # 이미 [B, gin]인 경우

        # style과 speaker를 feature 차원에서 concat
        cond = torch.cat([style_vec, g_flat], dim=-1)  # [B, S+gin]
        return cond

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transformer stack forward.

        x: [B, H, T]
        x_mask: [B, 1, T]
        style_vec: [B, S]
        g: [B, gin, 1] or [B, gin]
        """
        cond = self.build_cond(style_vec, g)  # [B, C]

        # 층별로 반복 적용
        for layer in self.layers:
            x = layer(x, x_mask, cond)

        return x
