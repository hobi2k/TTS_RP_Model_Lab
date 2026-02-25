"""
saya_tts/data/dataset.py

Dataset 확정판

이 Dataset의 책임(매우 중요):
- "파일에서 전처리 결과를 읽어" 하나의 샘플(dict)로 반환한다.
- 여기서 '모델 계산(BERT 추출, G2P, mel 생성 등)'을 하지 않는다.
  (그건 전처리 파이프라인 단계의 책임)

이번 Dataset이 해결해야 하는 포인트:
1) phones(List[str]) -> phoneme_ids(List[int]) 변환을 여기서 수행한다.
   - 이유: Trainer가 받을 텐서는 항상 정수 ID여야 하며,
           phoneme vocab이 고정되어야 하고,
           padding 규칙(<pad>=0)도 여기서 강제할 수 있다.

2) tone/lang 텐서를 "phones 길이와 정확히 맞춰" 만든다.
   - tone은 g2p 결과를 그대로 쓴다.
   - lang은 일본어 단일 언어이면 전부 0으로 채운다.

3) x_mask는 collate에서도 만들 수 있지만,
   여기서는 "생성 규칙(phoneme_id==0은 padding)"을 강제하기 위해
   샘플 단위에서 기본 마스크를 만들고,
   batch 단위 마스크는 collate에서 재생성해도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from saya_tts.text.phoneme_vocab import PhonemeVocab


@dataclass(frozen=True)
class SayaSample:
    """
    하나의 발화(샘플)를 "학습에 바로 넣을 수 있는 형태"로 정리한 구조.

    NOTE:
    - Dataset __getitem__은 dict를 반환해도 되지만,
      dataclass로 한 번 감싸두면 shape/type 규약을 문서로 고정하는 효과가 크다.

    Shapes (단일 샘플 기준, padding 전):
    - phoneme_ids: [T_text] (Long)
    - tone_ids: [T_text] (Long)
    - lang_ids: [T_text] (Long)
    - x_mask: [1, T_text] (Float; 유효=1, pad=0)  ※ batch에서는 [B,1,T]로 확장
    - bert_feats: [D_bert, T_text] (Float)
    - word2ph: [W] (Long)  # W는 문자/word 단위 길이
    - mel: [n_mels, T_mel] (Float)
    - style_id: scalar (Long)
    - speaker_id: scalar (Long)
    """
    utt_id: str
    raw_text: str

    phoneme_ids: torch.LongTensor
    tone_ids: torch.LongTensor
    lang_ids: torch.LongTensor
    x_mask: torch.FloatTensor

    bert_feats: torch.FloatTensor
    word2ph: torch.LongTensor
    mel: torch.FloatTensor

    style_id: torch.LongTensor
    speaker_id: torch.LongTensor


class SayaTTSDataset(Dataset):
    """
    SayaTTSDataset (JP-Extra / Style-Bert-VITS2 학습 입력 규약 기준)

    data_root 아래 폴더 구조(권장):
    - metadata.csv
    - text/
        <utt_id>_phones.npy      (object array: List[str])
        <utt_id>_tones.npy       (int array)
        <utt_id>_word2ph.npy     (int array)
    - bert/
        <utt_id>.npy             (float array: [D_bert, T_text])
    - mels/
        <utt_id>.npy             (float array: [n_mels, T_mel])

    phoneme vocab:
    - 처음엔 전체 phones를 스캔해서 vocab을 만든 뒤,
      파일로 저장하고 고정해서 쓰는 게 정석.
    - 여기서는 "이미 vocab.json이 존재한다"는 흐름으로 설계한다.
      (전처리 단계에서 build_vocab.py 같은 것으로 한 번 생성)
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        vocab_path: str | Path,
        bert_dim: int = 1024,
        n_mels: int = 80,
        language_id_for_jp: int = 0,
        validate_shapes: bool = True,
    ):
        """
        Args:
            data_root:
                data/saya 같은 폴더

            vocab_path:
                phoneme vocab(json) 경로.
                <pad>=0 규칙을 포함한 phoneme->id 맵.

            bert_dim:
                BERT feature hidden dimension.
                (JP-Extra는 보통 1024)

            n_mels:
                mel 채널 수.

            language_id_for_jp:
                단일 언어 학습이라면 전체 lang_ids를 이 값으로 채운다.
                (SBV2는 언어 embedding을 쓰기 때문에 형태는 유지)

            validate_shapes:
                True면 각 파일 로드시 shape 검증을 해서
                "전처리/저장 단계 오류"를 조기에 잡는다.
                학습 초기에 강력 추천.
        """
        self.data_root = Path(data_root)
        self.bert_dim = bert_dim
        self.n_mels = n_mels
        self.language_id_for_jp = int(language_id_for_jp)
        self.validate_shapes = bool(validate_shapes)

        # 하위 폴더 고정
        self.text_dir = self.data_root / "text"
        self.bert_dir = self.data_root / "bert"
        self.mel_dir = self.data_root / "mels"

        # phoneme vocab 로드(고정)
        self.vocab = PhonemeVocab.load(str(vocab_path))

        # metadata 로드
        self.items = self._load_metadata(self.data_root / "metadata.csv")

    def _load_metadata(self, path: Path) -> List[Dict[str, Any]]:
        """
        metadata.csv 형식 예:
        utt_id,text,style_id,speaker_id
        saya_0001,こんにちは。私はサヤです。,0,0
        """
        if not path.exists():
            raise FileNotFoundError(f"metadata.csv not found: {path}")

        rows: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # 필수 칼럼 검증 (실수 방지)
                for k in ("utt_id", "text", "style_id", "speaker_id"):
                    if k not in r:
                        raise ValueError(f"metadata.csv missing column: {k}")

                rows.append(
                    {
                        "utt_id": r["utt_id"],
                        "text": r["text"],
                        "style_id": int(r["style_id"]),
                        "speaker_id": int(r["speaker_id"]),
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def _load_phones(self, utt_id: str) -> List[str]:
        """
        phones.npy는 dtype=object로 저장된 경우가 많다.
        np.load(..., allow_pickle=True)가 필요할 수 있다.

        저장 형태를 강제하고 싶다면:
        - phones를 문자열을 join해서 저장하거나(json)로 저장해도 되지만,
          여기서는 numpy object로 저장된 케이스를 기본으로 지원한다.
        """
        p = self.text_dir / f"{utt_id}_phones.npy"
        if not p.exists():
            raise FileNotFoundError(f"phones file not found: {p}")

        arr = np.load(p, allow_pickle=True)

        # np.array(list[str], dtype=object) 형태일 때
        phones = arr.tolist()  # List[str]로 복구
        if not isinstance(phones, list) or not all(isinstance(x, str) for x in phones):
            raise TypeError(f"phones.npy must decode to List[str]. got type={type(phones)}")
        return phones

    def _load_int_array_1d(self, path: Path, name: str) -> np.ndarray:
        """
        tones/word2ph 같은 1D int 배열 로드 유틸.
        """
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        arr = np.load(path)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D. got shape={arr.shape}")
        return arr.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        반환 dict는 collate_fn이 batch로 묶을 수 있는 원소여야 한다.

        NOTE:
        - 여기서는 dataclass(SayaSample)로 한 번 만든 뒤 dict로 풀어 반환한다.
          (원하면 dataclass를 그대로 반환해도 되지만, 일반적으로 DataLoader는 dict가 편하다.)
        """
        item = self.items[idx]
        utt_id = item["utt_id"]
        raw_text = item["text"]
        style_id = item["style_id"]
        speaker_id = item["speaker_id"]

        # 1. phones / tones / word2ph 로드
        phones = self._load_phones(utt_id)

        tones = self._load_int_array_1d(self.text_dir / f"{utt_id}_tones.npy", "tones")
        word2ph = self._load_int_array_1d(self.text_dir / f"{utt_id}_word2ph.npy", "word2ph")

        # phones 길이(T_text)와 tones 길이가 동일해야 한다.
        # 이게 깨지면 G2P 저장 단계가 잘못된 것.
        if self.validate_shapes:
            if len(phones) != len(tones):
                raise ValueError(
                    f"phones/tones length mismatch for {utt_id}: {len(phones)} vs {len(tones)}"
                )

        # 2. phones -> phoneme_ids
        # vocab에 없는 phone이 나오면 KeyError가 날 수 있다.
        # 그 경우:
        # - vocab 생성 단계에 해당 phone이 누락된 것이므로
        # - dataset 전체 phones를 다시 스캔해서 vocab을 재생성해야 한다.
        phoneme_ids_list = self.vocab.encode(phones)

        # 3. lang_ids 생성
        # JP 단일 언어 가정:
        # - phones 길이와 동일한 길이로 0 채움
        lang_ids_list = [self.language_id_for_jp] * len(phoneme_ids_list)

        # 4. x_mask 생성 (샘플 단위)
        # padding이 아직 없으므로 전부 1이 되지만,
        # batch padding 이후에는 0이 생긴다.
        # 규칙은 "phoneme_id==0 이 padding" 이므로,
        # padding 이후에도 동일 규칙으로 mask를 재생성할 수 있다.
        x_mask_list = [1.0 if pid != 0 else 0.0 for pid in phoneme_ids_list]

        # 5. BERT feature 로드
        bert_path = self.bert_dir / f"{utt_id}.npy"
        if not bert_path.exists():
            raise FileNotFoundError(f"bert feature not found: {bert_path}")
        bert = np.load(bert_path).astype(np.float32)  # [D_bert, T_text]

        if self.validate_shapes:
            if bert.ndim != 2:
                raise ValueError(f"bert must be 2D [D_bert, T]. got shape={bert.shape}")
            if bert.shape[0] != self.bert_dim:
                raise ValueError(f"bert_dim mismatch: expected {self.bert_dim}, got {bert.shape[0]}")
            if bert.shape[1] != len(phones):
                raise ValueError(
                    f"bert T mismatch: bert.shape[1]={bert.shape[1]} vs len(phones)={len(phones)} for {utt_id}"
                )

        # 6. mel 로드
        mel_path = self.mel_dir / f"{utt_id}.npy"
        if not mel_path.exists():
            raise FileNotFoundError(f"mel not found: {mel_path}")
        mel = np.load(mel_path).astype(np.float32)  # [n_mels, T_mel]

        if self.validate_shapes:
            if mel.ndim != 2:
                raise ValueError(f"mel must be 2D [n_mels, T_mel]. got shape={mel.shape}")
            if mel.shape[0] != self.n_mels:
                raise ValueError(f"n_mels mismatch: expected {self.n_mels}, got {mel.shape[0]}")

        sample = SayaSample(
            utt_id=utt_id,
            raw_text=raw_text,
            phoneme_ids=torch.LongTensor(phoneme_ids_list),
            tone_ids=torch.LongTensor(tones),
            lang_ids=torch.LongTensor(lang_ids_list),
            x_mask=torch.FloatTensor(x_mask_list).unsqueeze(0),  # [1, T_text]
            bert_feats=torch.FloatTensor(bert),
            word2ph=torch.LongTensor(word2ph),
            mel=torch.FloatTensor(mel),
            style_id=torch.LongTensor([style_id]),      # scalar처럼 쓰기 위해 [1]로 둠
            speaker_id=torch.LongTensor([speaker_id]),  # scalar처럼 쓰기 위해 [1]로 둠
        )

        # DataLoader collate가 다루기 쉬운 dict로 반환
        return {
            "utt_id": sample.utt_id,
            "raw_text": sample.raw_text,

            "phoneme_ids": sample.phoneme_ids,
            "tone_ids": sample.tone_ids,
            "language_ids": sample.lang_ids,
            "x_mask": sample.x_mask,  # [1, T] (샘플 단위)

            "bert_feats": sample.bert_feats,  # [D_bert, T]
            "word2ph": sample.word2ph,        # [W]
            "mel": sample.mel,                # [n_mels, T_mel]

            "style_id": sample.style_id,      # [1]
            "speaker_id": sample.speaker_id,  # [1]
        }
