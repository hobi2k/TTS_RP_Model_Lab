"""RP 출력 파서."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RPBlock:
    """파싱된 역할극 출력 컨테이너."""
    narration: str
    action: str
    dialogue_en: str


class RPParser:
    """큰따옴표 기반 RP 텍스트를 구조화 블록으로 파싱한다."""

    def parse(self, text: str) -> RPBlock:
        """LLM 원문 텍스트를 RPBlock으로 변환한다."""

        if not text or not text.strip():
            return RPBlock("", "", "")

        narration, dialogue = self._split_by_quote(text)

        return RPBlock(
            narration=narration,
            action="",
            dialogue_en=dialogue,
        )

    def _split_by_quote(self, text: str) -> tuple[str, str]:
        """첫 큰따옴표 블록을 대사로 추출하고, 앞부분을 서술로 반환한다."""
        src = text.strip()
        m = re.search(r'"([^"\n]{1,600})"', src)
        if not m:
            # 큰따옴표가 없으면 전체를 대사로 간주
            return "", src

        dialogue = m.group(1).strip()
        before = src[: m.start()].strip()

        # 서술은 인용부 앞의 마지막 유효 줄을 사용
        narration = ""
        if before:
            lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
            if lines:
                narration = lines[-1]

        return narration, dialogue
