"""
Text Tensorization

phones / tones / langs ->
phoneme_ids / tone_ids / lang_ids / x_mask
"""

from __future__ import annotations
from typing import List
import torch

from .phoneme_vocab import PhonemeVocab


def tensorize_text(
    *,
    phones: List[str],
    tones: List[int],
    phoneme_vocab: PhonemeVocab,
    device: torch.device,
):
    """
    단일 샘플 기준 텐서화

    Returns:
        phoneme_ids: [1, T]
        tone_ids:    [1, T]
        lang_ids:    [1, T]
        x_mask:      [1, 1, T]
    """

    # phoneme → id
    phoneme_ids = phoneme_vocab.encode(phones)

    # tensor 변환
    phoneme_ids = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)
    tone_ids = torch.LongTensor(tones).unsqueeze(0).to(device)

    # 일본어 단일 언어 -> 전부 0
    lang_ids = torch.zeros_like(phoneme_ids)

    # x_mask 생성
    # padding == 0
    # 유효 토큰 == 1
    x_mask = (phoneme_ids != 0).float().unsqueeze(1)

    return phoneme_ids, tone_ids, lang_ids, x_mask