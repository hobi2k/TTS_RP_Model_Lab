from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

# 내부 모듈 import
# (가중치 호환 핵심)
# - DecoderLayer는 "조립자(orchestrator)" 역할만 해야 한다.
# - 학습 파라미터(Linear/Norm 등)의 이름과 shape는
#   attention.py / mlp.py / norm.py 내부에 이미 정의되어 있어야 한다.
from ..config import SayaQwen3Config
from .attention import Qwen2Attention
from .mlp import Qwen2MLP
from .norm import Qwen2RMSNorm


class Qwen2DecoderLayer(nn.Module):
    """
    Qwen2 Decoder Layer (Transformer block 1개)

    목표:
      1) HuggingFace Qwen 계열과 "가중치 호환"되는 블록 구조를 구성한다.
      2) Attention/MLP/RMSNorm을 HF 계열의 forward 흐름에 맞게 조립한다.
      3) KV cache, RoPE position_embeddings, attention_mask를 "그대로" 전달한다.

    표준 pre-norm decoder 구성:

        입력 x
          ├─ (1) RMSNorm(x) -> self_attn(...) -> residual add
          └─ (2) RMSNorm(x) -> MLP(...)       -> residual add

    즉, 각 sub-block마다:
      - pre-norm
      - sub-layer
      - residual add

    주의:
      - 이 레이어는 "token embedding", "position_ids 생성", "mask 생성"을 하지 않는다.
        전역 처리는 model.py(Qwen3Model.forward)의 책임이다.
      - 여기서는 오직 "한 레이어" 연산만 수행한다.
    """

    def __init__(self, config: SayaQwen3Config, layer_idx: int) -> None:
        """
        Args:
            config:
                SayaQwen3Config.
                hidden_size, rms_norm_eps, layer_types 등 포함.
            layer_idx:
                현재 레이어 index.
                - sliding attention 여부(layer_types[layer_idx]) 결정(선택)
                - cache에 저장될 때 layer_idx로 구분
        """
        super().__init__()

        # -------------------------------------------------------------
        # 0) 메타 정보 (디버깅/가독성용)
        # -------------------------------------------------------------
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # 레이어 타입(옵션):
        # - full_attention
        # - sliding_attention
        #
        # 원칙적으로 mask 선택(슬라이딩 여부)은 model.py에서 해결하는 것이 정석이다.
        # decoder는 "이미 준비된 attention_mask를 전달받는다"는 계약으로 단순화한다.
        self.attention_type = (
            config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        )

        # -------------------------------------------------------------
        # 1) Self-Attention 블록
        # -------------------------------------------------------------
        # HF naming과 동일한 속성명 "self_attn" 사용 (state_dict key 호환 핵심)
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        # -------------------------------------------------------------
        # 2) MLP 블록
        # -------------------------------------------------------------
        # HF naming과 동일한 속성명 "mlp" 사용
        self.mlp = Qwen2MLP(config)

        # -------------------------------------------------------------
        # 3) RMSNorm 2개 (pre-norm 구조)
        # -------------------------------------------------------------
        # HF naming:
        # - input_layernorm
        # - post_attention_layernorm
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:
                (B, S, hidden_size)

            attention_mask:
                additive mask. (보통 (B, 1, S_q, S_k) 형태)
                - model.py에서 causal/padding 및 sliding 여부를 반영해 준비해 주는 것을 권장.
                - 생성 단계에서는 S_q=1, S_k=T_total이 될 수 있다.

            position_ids:
                (B, S) 형태로 들어올 수 있으나,
                이 구현에서는 position_embeddings(cos,sin)를 직접 받는 것을 전제로 한다.
                (HF Qwen 계열은 model 단계에서 rotary_emb로 cos/sin을 계산하여 전달)

            position_embeddings:
                (cos, sin)
                - cos/sin: (B, S, D) 또는 브로드캐스팅 가능한 형태
                - attention 내부에서 q/k에 RoPE를 적용한다.

                중요:
                - position_embeddings는 "없으면 안 된다".
                  (q,k에 RoPE를 적용해야 cache 포함 score가 일관됨)
                - 따라서 None이면 즉시 에러를 내서
                  model.py 계약 위반을 빠르게 드러내는 것이 정답이다.

            past_key_values:
                HF Cache(DynamicCache 등) 또는 HFCompatibleKVCache.
                - attention 내부에서 past_key_values.update(...)를 호출해 KV를 누적한다.

            use_cache:
                True면 cache를 사용한다는 의사표시.
                - 실제 cache update는 attention이 수행한다.
                - model.py에서 use_cache=True일 때 past_key_values를 생성해 전달해야 한다.

            cache_position:
                (S,) 또는 (B, S) 형태의 "절대 위치 인덱스".
                - 생성 단계에서는 보통 S=1이고 값은 현재 토큰의 절대 위치.
                - HF cache 확장(rope 정렬 등)에서 사용될 수 있다.

                주의:
                - use_cache=False면 past_key_values가 None으로 내려가므로
                  cache_position이 있어도 cache update가 일어나지 않는다.

        Returns:
            hidden_states:
                (B, S, hidden_size)
        """

        # -------------------------------------------------------------
        # (필수) position_embeddings 계약 검증
        # -------------------------------------------------------------
        if position_embeddings is None:
            raise ValueError(
                "position_embeddings (cos, sin) is required. "
                "Make sure model.py computes rotary embeddings and passes (cos, sin)."
            )

        # -------------------------------------------------------------
        # Block 1: Self-Attention (Pre-Norm + Residual)
        # -------------------------------------------------------------
        # (1) residual connection을 위해 원본을 저장
        residual = hidden_states

        # (2) pre-norm
        hidden_states = self.input_layernorm(hidden_states)

        # (3) self attention
        # - attention.py가 position_embeddings, attention_mask, cache 등을 처리한다.
        # - use_cache=True일 때만 past_key_values를 전달하여 update()가 실행되게 한다.
        attn_output, _attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values if use_cache else None,
            cache_position=cache_position if use_cache else None,
            **kwargs,
        )

        # (4) residual add
        hidden_states = residual + attn_output

        # -------------------------------------------------------------
        # Block 2: MLP (Pre-Norm + Residual)
        # -------------------------------------------------------------
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        return hidden_states