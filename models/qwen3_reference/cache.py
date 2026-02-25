# models/qwen/cache.py
"""
HFCompatibleKVCache - HuggingFace generate() 호환용 최소 KV Cache

목적:
- HuggingFace Transformers의 Cache 인터페이스를 최소 구현하여,
  custom modeling에서도 generate()가 past_key_values를 정상적으로 재사용하게 한다.

중요 개념:
- Transformer는 레이어마다 attention이 다르므로, KV도 레이어별로 저장한다.
- 저장 텐서 shape:
    key/value: (B, H, T, D)
    B: 배치 크기
    H: (kv) head 수
    T: 누적 토큰 길이
    D: head_dim

주의:
- 이 구현은 성능 최적화용이 아니다.
- static cache / sliding window / rope 재정렬 등 고급 기능은 구현하지 않는다.
- 목적은:
  1) generate()가 깨지지 않게 한다
  2) KV cache 동작 흐름을 명확히 보여준다
"""

# models/qwen3_reference/cache.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
from transformers.cache_utils import Cache


@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class HFCompatibleKVCache(Cache):
    """
    HuggingFace generate() 호환을 위한 최소 Cache 구현.

    - 레이어별 KV를 누적 저장
    - update() 호출 시 새 KV를 append하고 (K_all, V_all) 반환
    - get_seq_length()로 현재 캐시 길이 제공
    - beam search 등을 위해 reorder_cache() 제공
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache: Dict[int, LayerKV] = {}
        self._seq_length: int = 0

    def get_seq_length(self) -> int:
        return self._seq_length

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key:   (B, H, S_new, D)
            value: (B, H, S_new, D)
        Returns:
            key_all, value_all: (B, H, T_total, D)
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = LayerKV(key=key, value=value)
            key_all = key
            value_all = value
        else:
            old = self._cache[layer_idx]
            key_all = torch.cat([old.key, key], dim=-2)
            value_all = torch.cat([old.value, value], dim=-2)
            self._cache[layer_idx] = LayerKV(key=key_all, value=value_all)

        self._seq_length = int(key_all.shape[-2])
        return key_all, value_all

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "HFCompatibleKVCache":
        """
        Beam search에서 배치/빔 순서를 재정렬할 때 호출될 수 있음.
        beam_idx: (new_B,)
        """
        if beam_idx is None:
            return self

        # beam_idx는 보통 GPU 위 텐서이므로, 동일 디바이스 유지
        for layer_idx, kv in self._cache.items():
            self._cache[layer_idx] = LayerKV(
                key=kv.key.index_select(0, beam_idx),
                value=kv.value.index_select(0, beam_idx),
            )
        return self

    # 아래는 Cache 추상 클래스가 내부적으로 기대할 수 있는 속성들에 대한 방어적 구현
    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Any:
        # 일부 코드가 past_key_values[layer_idx] 형태로 접근하는 경우가 있음
        kv = self._cache[idx]
        return (kv.key, kv.value)
