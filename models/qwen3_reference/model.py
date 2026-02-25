# models/qwen/model.py
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .config import SayaQwen3Config
from .cache import HFCompatibleKVCache
from .layers.decoder import Qwen2DecoderLayer
from .layers.norm import Qwen2RMSNorm
from .layers.rotary import Qwen2RotaryEmbedding


class Qwen3Model(nn.Module):
    """
    Qwen3Model (Transformer 본체, LM head 없음)

    책임:
      - token embedding
      - position_ids / cache_position 생성
      - RoPE cos/sin 생성(공유)
      - attention mask 준비(HF util 사용)
      - decoder layer 순회
      - final norm
      - BaseModelOutputWithPast 반환

    주의:
      - generate() 자체 로직은 CausalLM(PreTrainedModel + GenerationMixin) 쪽 책임
      - 여기서는 "cache 계약"을 만족시키는 past_key_values를 생산/소비한다.
    """

    def __init__(self, config: SayaQwen3Config) -> None:
        super().__init__()
        self.config = config

        # 1) Token embedding
        # HF 호환을 위해 속성명 embed_tokens 유지
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )

        # 2) Decoder layers
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config=config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # 3) Final RMSNorm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4) Rotary Embedding (레이어 공유)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[HFCompatibleKVCache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        Args:
            input_ids:
                (B, S)
            attention_mask:
                (B, S) padding mask(1=valid,0=pad) 또는 이미 4D additive mask일 수 있음
            position_ids:
                (B, S)
            inputs_embeds:
                (B, S, hidden)
            past_key_values:
                HFCompatibleKVCache 또는 HF Cache
            use_cache:
                True면 KV cache 사용
        """

        # ------------------------------------------------------------
        # 0) 입력 검증: input_ids vs inputs_embeds
        # ------------------------------------------------------------
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        # ------------------------------------------------------------
        # 1) Embedding
        # ------------------------------------------------------------
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        # hidden_states: (B, S, hidden)
        batch_size, seq_len, _ = hidden_states.shape

        # ------------------------------------------------------------
        # 2) use_cache 기본값 처리
        # ------------------------------------------------------------
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))

        # ------------------------------------------------------------
        # 3) KV cache 준비
        # ------------------------------------------------------------
        if use_cache:
            if past_key_values is None:
                past_key_values = HFCompatibleKVCache()
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            past_key_values = None
            past_seen_tokens = 0

        # ------------------------------------------------------------
        # 4) cache_position (절대 위치)
        #    - 프리필: [0..S-1]
        #    - 생성:   [past_len] (S=1)
        # ------------------------------------------------------------
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            device=hidden_states.device,
        )

        # ------------------------------------------------------------
        # 5) position_ids 생성
        # ------------------------------------------------------------
        if position_ids is None:
            # (B, S)
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # ------------------------------------------------------------
        # 6) RoPE cos/sin 생성
        #
        # 중요:
        # - 너의 rotary 구현은 RotaryOutput(dataclass)을 반환한다.
        # - attention.py는 (cos, sin) 튜플을 기대한다.
        # - 따라서 여기서 튜플로 "명시적으로" 변환한다.
        # ------------------------------------------------------------
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # ------------------------------------------------------------
        # 7) attention_mask 준비
        # ------------------------------------------------------------
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len),
                device=hidden_states.device,
                dtype=torch.long,
            )

        # (B,S) padding mask -> 4D causal additive mask
        if attention_mask.dim() == 2:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                input_shape=(batch_size, seq_len),
                inputs_embeds=hidden_states,
                past_key_values_length=past_seen_tokens,
            )

        # ------------------------------------------------------------
        # 8) Decoder layers
        # ------------------------------------------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        # ------------------------------------------------------------
        # 9) Final norm
        # ------------------------------------------------------------
        hidden_states = self.norm(hidden_states)

        # ------------------------------------------------------------
        # 10) Output
        # ------------------------------------------------------------
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
