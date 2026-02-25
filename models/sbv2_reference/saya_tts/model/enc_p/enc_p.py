from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from .input_embedding import TextInputEmbedding
from .transformer_stack import TransformerEncoderStack
from .prior_head import PriorHead


@dataclass
class EncPOut:
    """
    enc_p(Text Encoder / Prior Encoder)의 최종 출력 묶음.

    이 구조는 이후 infer_after_enc_p(...)에서
    그대로 입력으로 사용된다.

    x:
        [B, H, T]
        Transformer를 통과한 텍스트 인코딩 결과

    m_p:
        [B, D, T]
        prior 분포의 평균(mean)

    logs_p:
        [B, D, T]
        prior 분포의 로그 표준편차(log-std)

    x_mask:
        [B, 1, T]
        padding 마스크 (유효=1, padding=0)

    g:
        [B, gin, 1] 또는 [B, gin]
        화자/스타일 조건 임베딩
        (duration predictor / flow / decoder에서 재사용됨)
    """
    x: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    x_mask: torch.Tensor
    g: torch.Tensor


class TextEncoderPrior(nn.Module):
    """
    Style-Bert-VITS2의 enc_p(Text Encoder / Prior Encoder) 전체 구현.

    역할 요약:
    - 텍스트(phoneme/tone/lang/bert) + 조건(style/speaker)을 입력으로 받아
    - 잠재 변수 z_p의 "정규분포 파라미터" (m_p, logs_p)를 생성한다.

    즉:
        text -> p(z | text)
    """

    def __init__(
        self,
        *,
        # Vocabulary / embedding 관련
        num_phonemes: int,
        num_tones: int,
        num_languages: int,

        # 모델 차원 관련
        hidden_size: int, # H: Transformer hidden size
        latent_dim: int, # D: latent(z) channel size
        bert_dim: int = 1024, # 일본어 BERT hidden size

        # Transformer 관련
        num_layers: int,
        num_heads: int,

        # 조건 벡터 관련
        style_dim: int, # style_vec 차원
        speaker_dim: int, # speaker embedding 차원

        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Input Embedding Block
        # 텍스트 ID + BERT feature를 hidden space(H)로 정렬
        self.input_embedding = TextInputEmbedding(
            num_phonemes=num_phonemes,
            num_tones=num_tones,
            num_languages=num_languages,
            hidden_size=hidden_size,
            bert_dim=bert_dim,
        )

        # 2. Transformer Encoder Stack
        # 문맥 정보를 반영하는 핵심 블록
        self.transformer = TransformerEncoderStack(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            style_dim=style_dim,
            speaker_dim=speaker_dim,
            dropout=dropout,
        )

        # 3. Prior Head
        # hidden space(H) -> latent space(D)
        self.prior_head = PriorHead(
            hidden_size=hidden_size,
            latent_dim=latent_dim,
        )

    def forward(
        self,
        *,
        phoneme_ids: torch.LongTensor,
        tone_ids: torch.LongTensor,
        language_ids: torch.LongTensor,
        bert_feats: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor,
    ) -> EncPOut:
        """
        enc_p 전체 forward.

        Args:
            phoneme_ids:
                [B, T]
                음소 ID 시퀀스

            tone_ids:
                [B, T]
                억양 ID 시퀀스

            language_ids:
                [B, T]
                언어 ID 시퀀스

            bert_feats:
                [B, D_bert, T]
                BERT에서 추출한 contextual feature

            style_vec:
                [B, S]
                스타일/감정 벡터

            g:
                [B, gin, 1] 또는 [B, gin]
                화자/조건 임베딩

        Returns:
            EncPOut:
                enc_p의 모든 출력
        """

        # padding mask 생성
        # phoneme_ids에서 0을 padding으로 가정
        # padding 위치 = 0
        # 유효 토큰 = 1
        #
        # 결과 shape:
        #   x_mask: [B, 1, T]
        x_mask = (phoneme_ids != 0).unsqueeze(1).to(phoneme_ids.dtype)

        # 1. Input Embedding
        # 텍스트 정보들을 hidden space(H)로 투영
        #
        # 출력:
        #   x_embed: [B, H, T]
        x_embed = self.input_embedding(
            phoneme_ids=phoneme_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_feats=bert_feats,
        )

        # 2. Transformer Encoder Stack
        # 문맥 반영 + 조건(style/speaker) 주입
        #
        # 출력:
        #   x_enc: [B, H, T]
        x_enc = self.transformer(
            x=x_embed,
            x_mask=x_mask,
            style_vec=style_vec,
            g=g,
        )

        # 3. Prior Head
        # 텍스트 인코딩 결과를 잠재분포 파라미터로 변환
        #
        # 출력:
        #   m_p, logs_p: [B, D, T]
        m_p, logs_p = self.prior_head(
            x_enc=x_enc,
            x_mask=x_mask,
        )

        # 4. EncPOut로 묶어서 반환
        return EncPOut(
            x=x_enc,
            m_p=m_p,
            logs_p=logs_p,
            x_mask=x_mask,
            g=g,
        )
