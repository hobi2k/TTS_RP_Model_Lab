import re

def split_sentences(text: str) -> list[str]:
    """
    Very simple sentence splitter (EN).
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]