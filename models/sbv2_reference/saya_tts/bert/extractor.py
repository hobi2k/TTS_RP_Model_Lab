"""
BERT Feature Extraction Wrapper

이 모듈은 Style-Bert-VITS2의 infer.py 내부에서 수행되던
BERT feature 추출 과정을 '외부에서 명시적으로 호출 가능한 단계'로 분리한다.

핵심 목표:
- phones + word2ph -> bert_feats
- infer.py 의존 제거
- TextEncoder 입력의 절반을 우리 손에 쥔다
"""
from typing import List

import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.bert_models import get_bert


class BertFeatureExtractor:
    """
    Style-Bert-VITS2에서 사용하는 BERT feature extractor 래퍼.

    이 클래스는 다음을 책임진다:
    - 언어에 맞는 BERT 모델 로딩
    - phoneme 기준으로 정렬된 BERT feature 생성

    중요
    - raw text는 절대 입력으로 받지 않는다.
    - BERT 입력의 기준 축은 'phoneme(T_phone)'이다.
    """

    def __init__(
        self,
        *,
        language: Languages,
        device: str = "cuda",
    ):
        self.language = language
        self.device = device

        # 언어에 맞는 BERT 로드
        # (JP: deberta-v2-large-japanese-char-wwm 등)
        self.bert = get_bert(language)
        self.bert.to(device)
        self.bert.eval()

    @torch.inference_mode()
    def extract(
        self,
        *,
        phones: List[int],
        word2ph: List[int],
    ) -> torch.Tensor:
        """
        phoneme + word2ph 정보를 사용해 BERT feature를 생성한다.

        Parameters
    
        phones : List[int]
            phoneme ID 시퀀스.
            길이 = T_phone

        word2ph : List[int]
            각 단어가 몇 개의 phoneme으로 구성되는지 나타내는 리스트.

        Returns

        bert_feats : torch.Tensor
            shape = (D_bert, T_phone)

        Notes

        - 반환 텐서는 TextEncoder로 바로 입력된다.
        - infer.py 내부의 get_bert_feature()와 동일한 동작을 한다.
        """

        # infer.py 내부에서는 phones가 이미 tensor이지만,
        # 우리는 외부 모듈이므로 여기서 tensor로 변환한다.
        phones_tensor = torch.LongTensor(phones).unsqueeze(0).to(self.device)

        # BERT feature 추출
        bert_feats = self.bert.get_bert_feature(
            phones_tensor,
            word2ph,
            self.device,
        )

        # shape 보장: (D_bert, T_phone)
        return bert_feats
