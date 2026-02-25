# models/qwen3_reference/attention.py
from __future__ import annotations

from typing import Optional, Tuple, Any

import torch
from torch import nn

from .rotary import apply_rotary_pos_emb
from .norm import Qwen2RMSNorm


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (B, Hkv, S, D) -> (B, Hq, S, D) where Hq = Hkv * n_rep
    """
    bsz, hkv, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, hkv, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, hkv * n_rep, seq_len, head_dim)


def eager_attention_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout_p: float,
    training: bool,
    num_key_value_groups: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    q: (B, Hq, Sq, D)
    k: (B, Hkv, Sk, D)
    v: (B, Hkv, Sk, D)
    """
    k = repeat_kv(k, num_key_value_groups)
    v = repeat_kv(v, num_key_value_groups)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    if attention_mask is not None:
        sq = q.shape[-2]
        sk = k.shape[-2]
        mask = attention_mask[..., :sq, :sk]
        scores = scores + mask

    attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    if training and dropout_p > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)

    attn_output = torch.matmul(attn_weights, v)
    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """
    Qwen3(GQA) 대응 Attention.

    체크포인트 키 정합 포인트:
    - q_proj / k_proj / v_proj / o_proj
    - q_norm / k_norm (QK-Norm 계열이 존재하는 체크포인트면 필수)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)

        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_attention_heads)

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"Invalid GQA config: num_attention_heads({self.num_attention_heads}) "
                f"must be divisible by num_key_value_heads({self.num_key_value_heads})."
            )

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.is_causal = True

        # NOTE: 여기 bias=True 때문에 체크포인트에 bias가 없으면 "bias가 meta에 남는" 문제가 생긴다.
        # Qwen 계열은 종종 q/k/v projection bias를 두지 않는다.
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm: 체크포인트에 존재하는 경우가 많음 (네 로그에서도 존재)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=eps)

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    def split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()
        return x

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape

        if use_cache is None:
            use_cache = past_key_values is not None

        q_raw, k_raw, v_raw = self.project_qkv(hidden_states)

        q = self.split_heads(q_raw, self.num_attention_heads)
        k = self.split_heads(k_raw, self.num_key_value_heads)
        v = self.split_heads(v_raw, self.num_key_value_heads)

        # QK-Norm 적용 (head_dim 기준)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.apply_rope(q, k, position_embeddings)

        if use_cache and past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_output, attn_weight = eager_attention_forward(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
            num_key_value_groups=self.num_key_value_groups,
        )

        if not output_attentions:
            attn_weight = None

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, self.num_attention_heads * self.head_dim)
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weight
