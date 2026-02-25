"""
Text Cleaning / Phoneme Extraction Wrapper

이 모듈의 목적은:
- Style-Bert-VITS2의 infer.py 안에 숨어 있던 clean_text()를
- '우리 파이프라인이 직접 호출할 수 있는 함수'로 끌어내는 것

중요
- 여기서는 clean_text의 로직을 재구현하지 않는다.
- 원본 구현을 그대로 사용한다.
- 우리가 하는 일은 "경계 분리 + 의미 명시"다.
"""

from typing import List, Tuple

from style_bert_vits2.constants import Languages
from style_bert_vits2.text.cleaner import clean_text as _clean_text


def clean_text_for_tts(
    text: str,
    *,
    language: Languages,
) -> Tuple[List[int], List[int], List[int]]:
    """
    텍스트를 TTS 모델이 이해할 수 있는 '발음 단위 시퀀스'로 변환한다.

    이 함수는 다음을 수행한다:

    1. 텍스트 정규화
       - 기호 처리
       - 언어별 전처리 (JP / EN / ZH 등)

    2. G2P (Grapheme-to-Phoneme)
       - 문자 -> 발음 기호 ID 시퀀스

    3. Tone / Accent 정보 추출
       - 일본어의 경우 pitch accent 정보 포함

    4. Word-to-Phoneme 정렬 정보 생성
       - BERT feature를 phoneme 단위로 확장하기 위해 필요

    Parameters

    text : str
        사용자가 입력한 원문 텍스트.
        예) "こんにちは。私はサヤです。"

    language : Languages
        사용할 언어.
        예) Languages.JP

    Returns

    phones : List[int]
        phoneme ID 시퀀스.
        길이 = T_phone

    tones : List[int]
        phoneme 단위 tone / accent 정보.
        길이 = T_phone

    word2ph : List[int]
        각 단어가 몇 개의 phoneme으로 이루어져 있는지 나타내는 리스트.
        길이 = 단어 개수

    Notes
    - 반환되는 phones / tones는 아직 tensor가 아니다.
    - infer.py 내부에서는 여기서 바로 torch.LongTensor로 변환된다.
    - 이후 단계에서는 이 지점을 기준으로
      BERT feature, TextEncoder를 단계적으로 분리한다.
    """

    # 원본 clean_text 호출
    phones, tones, word2ph = _clean_text(
        text=text,
        language=language,
    )

    # 여기서는 절대 shape을 바꾸지 않는다.
    # 우리는 "의미 있는 경계"만 만든다.
    return phones, tones, word2ph