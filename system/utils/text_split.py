"""텍스트 분할 유틸리티."""

import re


def split_sentences(text: str) -> list[str]:
    """문장 경계(., !, ?) 기준으로 텍스트를 분할한다.

    Args:
        text: 원본 문자열.

    Returns:
        list[str]: 공백/빈 문자열이 제거된 문장 리스트.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]
