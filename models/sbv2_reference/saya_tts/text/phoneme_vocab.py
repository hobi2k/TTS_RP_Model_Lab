"""
Phoneme Vocabulary

역할:
- G2P 결과로 나온 phones(List[str])를
- 정수 ID(LongTensor)로 변환한다.

Style-Bert-VITS2 규칙:
- 0번은 padding(<pad>)으로 고정
- 나머지는 phoneme 문자열 → ID
"""
from __future__ import annotations
from typing import List, Dict
import json


class PhonemeVocab:
    def __init__(self, phonemes: List[str]):
        """
        Args:
            phonemes:
                전체 데이터셋에서 등장한 phoneme 문자열 목록 (중복 포함 가능)
        """
        uniq = sorted(set(phonemes))

        # 0번은 padding
        self.pad_token = "<pad>"
        self.ph2id: Dict[str, int] = {self.pad_token: 0}
        self.id2ph: Dict[int, str] = {0: self.pad_token}

        for i, ph in enumerate(uniq, start=1):
            self.ph2id[ph] = i
            self.id2ph[i] = ph

    def encode(self, phones: List[str]) -> List[int]:
        """
        phones(List[str]) -> phoneme_ids(List[int])
        """
        return [self.ph2id[p] for p in phones]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ph2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhonemeVocab":
        with open(path, encoding="utf-8") as f:
            ph2id = json.load(f)
        obj = cls([])
        obj.ph2id = ph2id
        obj.id2ph = {v: k for k, v in ph2id.items()}
        return obj