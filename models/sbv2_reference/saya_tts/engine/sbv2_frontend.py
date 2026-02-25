from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.models import commons
from style_bert_vits2.nlp import clean_text, cleaned_text_to_sequence, extract_bert_feature


@dataclass
class FrontendOut:   
    """
    Style-Bert-VITS2 원본 infer.py의 get_text() 최종 산출물과 동일 컨셉.

    모든 텐서는 아직 "배치 차원"이 없는 1D/2D 상태로 반환한다.
    - phones: [T]
    - tones: [T]
    - lang_ids: [T]
    - bert/ja_bert/en_bert: [1024, T] (원본이 이렇게 다룸)
    """
    phones: torch.LongTensor
    tones: torch.LongTensor
    lang_ids: torch.LongTensor
    bert: torch.Tensor
    ja_bert: torch.Tensor
    en_bert: torch.Tensor


def get_text_like_sbv2(
    text: str,
    language: Languages,
    device: str,
    use_jp_extra: bool,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
) -> FrontendOut:
    """
    원본: style_bert_vits2/models/infer.py:get_text()를 '동일 동작' 목표로 재구현.

    포인트:
    - clean_text()가 norm_text, phone(list[str]), tone(list[int]), word2ph(list[int])를 만든다.
    - cleaned_text_to_sequence()로 phone/tone/lang_ids를 정수 시퀀스로 변환한다.
    - add_blank 옵션이 켜져 있으면 intersperse(+word2ph 보정)한다.
    - extract_bert_feature()는 [1024, sum(word2ph)] 혹은 [1024, len(phone)]에 대응하는 피처를 준다.
    - JP일 때는 ja_bert=bert_ori, bert/en_bert=zeros 규칙. :contentReference[oaicite:6]{index=6}
    """
    # 1. 텍스트 정규화 + G2P/억양 등 전처리(언어별 내부 구현은 SBV2에 맡김)
    norm_text, phone, tone, word2ph = clean_text(
        text,
        language,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=False,  # 원본: infer 시에는 False
    )

    # 2. 사용자가 phone/tone을 직접 주는 케이스(고급). 필요 없으면 그대로 둬도 됨.
    #    원본은 길이 검증 및 JP에 한해 word2ph 조정 로직이 있음. :contentReference[oaicite:8]{index=8}
    if given_phone is not None and given_tone is not None:
        if len(given_phone) != len(given_tone):
            raise ValueError("given_phone 길이와 given_tone 길이가 다릅니다.")
        phone = given_phone
        tone = given_tone
    elif given_tone is not None:
        if len(phone) != len(given_tone):
            raise ValueError("생성된 phone 길이와 given_tone 길이가 다릅니다.")
        tone = given_tone

    # 3. phone/tone/lang -> 정수 시퀀스로 변환 (여기서 SYMBOL ID로 변환됨)
    phone_ids, tone_ids, lang_ids = cleaned_text_to_sequence(phone, tone, language)

    # 4. add_blank 처리는 hps.data.add_blank에 의해 결정되는데,
    #    여기서는 "외부에서 이미 적용 여부를 확정한 상태"로 쓰는 게 안전하다.
    #    여기서는 hps를 호출부에서 알고 있으니, 아래는 호출부에서 옵션으로 처리하는 형태가 더 낫다.
    #    하지만 원본은 get_text() 내부에서 바로 적용하므로, 필요하면 호출부에서 아래 블록을 킨다.
    #
    # if hps.data.add_blank:
    #     phone_ids = commons.intersperse(phone_ids, 0)
    #     tone_ids = commons.intersperse(tone_ids, 0)
    #     lang_ids = commons.intersperse(lang_ids, 0)
    #     word2ph = [w * 2 for w in word2ph]
    #     word2ph[0] += 1

    # 5. BERT feature 추출: [1024, T_phone] 형태가 되도록 word2ph 기반으로 확장/정렬됨
    bert_ori = extract_bert_feature(
        norm_text,
        word2ph,
        language,
        device,
        assist_text,
        assist_text_weight,
    )
    # 원본에서도 길이 assert를 강하게 건다. 
    assert bert_ori.shape[-1] == len(phone_ids), f"bert_len={bert_ori.shape[-1]} vs phone_len={len(phone_ids)}"

    # 6. 언어별로 (bert, ja_bert, en_bert) 3개 채널을 만드는 규칙이 매우 중요
    if language == Languages.ZH:
        bert = bert_ori
        ja_bert = torch.zeros(1024, len(phone_ids))
        en_bert = torch.zeros(1024, len(phone_ids))
    elif language == Languages.JP:
        bert = torch.zeros(1024, len(phone_ids))
        ja_bert = bert_ori
        en_bert = torch.zeros(1024, len(phone_ids))
    elif language == Languages.EN:
        bert = torch.zeros(1024, len(phone_ids))
        ja_bert = torch.zeros(1024, len(phone_ids))
        en_bert = bert_ori
    else:
        raise ValueError("language must be ZH/JP/EN")

    phones = torch.LongTensor(phone_ids)
    tones = torch.LongTensor(tone_ids)
    lang_ids = torch.LongTensor(lang_ids)

    return FrontendOut(
        phones=phones,
        tones=tones,
        lang_ids=lang_ids,
        bert=bert,
        ja_bert=ja_bert,
        en_bert=en_bert,
    )
