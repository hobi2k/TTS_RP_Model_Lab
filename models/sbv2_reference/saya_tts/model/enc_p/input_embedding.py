from __future__ import annotations

import torch
import torch.nn as nn


class TextInputEmbedding(nn.Module):
    """
    enc_p(Text Encoder)의 첫 번째 블록: Input Embedding Block

    B: 배치 크기
    T: 시간축 및 토큰 길이 (음소 개수)
    H: 히든 차원
    D_bert: bert 히든 차원
    D: 잠재 차원

    역할:
    - phoneme / tone / language / BERT feature를
      동일한 hidden space(H)로 투영한 뒤,
      하나의 텍스트 표현으로 결합한다.

    출력:
    - x_embed: [B, H, T]
    """

    def __init__(
        self,
        num_phonemes: int,
        num_tones: int,
        num_languages: int,
        hidden_size: int,
        bert_dim: int = 1024,
    ):
        super().__init__()

        # 1. Phoneme embedding
        # 각 음소 ID를 hidden_size 차원의 벡터로 변환
        # 가장 기본적인 "텍스트 정체성" 정보
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=num_phonemes,
            embedding_dim=hidden_size,
        )

        # 2. Tone embedding
        # 억양/악센트 정보
        # 일본어에서는 pitch accent를 간접적으로 표현
        self.tone_embedding = nn.Embedding(
            num_embeddings=num_tones,
            embedding_dim=hidden_size,
        )

        # 3. Language embedding
        # 다국어 모델에서 언어 조건을 주기 위한 embedding
        # JP-Extra에서도 구조상 존재 (값은 전부 JP일 수 있음)
        self.language_embedding = nn.Embedding(
            num_embeddings=num_languages,
            embedding_dim=hidden_size,
        )

        # 4. BERT feature projection
        # BERT는 보통 1024차원
        # 이를 enc_p hidden space(H)로 선형 투영
        self.bert_projection = nn.Linear(
            in_features=bert_dim,
            out_features=hidden_size,
            bias=False,
        )

    def forward(
        self,
        phoneme_ids: torch.LongTensor,
        tone_ids: torch.LongTensor,
        language_ids: torch.LongTensor,
        bert_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
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
                BERT에서 추출된 contextual feature
                (JP-Extra에서는 일본어 BERT)

        Returns:
            x_embed:
                [B, H, T]
                enc_p로 전달될 텍스트 임베딩
        """

        # A. Embedding lookup
        # 각 ID 시퀀스를 embedding 벡터로 변환
        # 결과 shape: [B, T, H]
        phoneme_emb = self.phoneme_embedding(phoneme_ids)
        tone_emb = self.tone_embedding(tone_ids)
        language_emb = self.language_embedding(language_ids)

        # B. BERT feature projection
        # bert_feats: [B, D_bert, T]
        # -> Transformer/Conv는 보통 [B, T, D]를 선호
        # -> 차원 교환 후 linear projection
        bert_feats = bert_feats.transpose(1, 2)  # [B, T, D_bert]
        bert_emb = self.bert_projection(bert_feats)  # [B, T, H]

        # C. Embedding fusion (SUM)
        # 모든 embedding은 같은 hidden space(H)에 있음
        # 의미를 "겹쳐서" 하나의 토큰 표현으로 만든다
        x = (
            phoneme_emb
            + tone_emb
            + language_emb
            + bert_emb
        )  # [B, T, H]

        # D. Transformer 계열과 맞추기 위해 차원 정렬
        # 이후 블록들은 [B, H, T]를 기대하는 경우가 많음
        x = x.transpose(1, 2)  # [B, H, T]

        return x
