"""
State Machine

목적:
- 캐릭터의 가변 상태를 중앙에서 관리한다.
- 사용자 입력/시스템 이벤트에 따라 상태를 갱신한다.
- LLM은 상태를 "읽기만" 하며, 상태 변경 권한은 없다.

설계 원칙:
- 상태는 단순한 값으로 표현한다.
- 전이 규칙은 명시적으로 코드화한다.
- 모델(LLM/TTS)과 완전히 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


# 상태 정의
@dataclass
class CharacterState:
    """
    캐릭터의 가변 상태 컨테이너.

    주의:
    - 이 구조는 prompt_compiler에서 그대로 사용된다.
    - 필드는 늘릴 수 있지만 의미는 명확해야 한다.
    """
    mood: str = "neutral"
    affection: int = 0
    memory_summary: Optional[str] = None


# 상태 머신
class StateMachine:
    """
    캐릭터 상태 머신.

    책임:
    - 현재 상태 보관
    - 이벤트 기반 상태 업데이트
    - 상태 스냅샷 제공
    """

    def __init__(self, initial_state: Optional[CharacterState] = None) -> None:
        """
        Args:
            initial_state:
                시작 상태. None이면 기본 상태를 사용한다.
        """
        self._state = initial_state or CharacterState()

    # 공개 API
    def get_state(self) -> CharacterState:
        """
        현재 상태 스냅샷을 반환한다.

        주의:
        - 반환된 객체는 읽기 전용으로 취급해야 한다.
        """
        return self._state

    def update_from_user_input(self, text: str) -> None:
        """
        사용자 입력으로부터 상태를 갱신한다.

        이 함수는:
        - 간단한 규칙 기반 로직만 포함한다.
        - 복잡한 감정 분석/분류는 추후 모듈로 분리 가능하다.
        """

        # 예시 규칙 1: 긍정 키워드가 있으면 affection 증가
        positive_keywords = ["고마워", "좋아", "사랑", "멋져"]
        if any(word in text for word in positive_keywords):
            self._state.affection += 1

        # 예시 규칙 2: 부정 키워드가 있으면 mood 변화
        negative_keywords = ["싫어", "짜증", "화나", "미워"]
        if any(word in text for word in negative_keywords):
            self._state.mood = "upset"

        # 예시 규칙 3: 질문 형태면 mood를 attentive로 설정
        if text.strip().endswith("?"):
            self._state.mood = "attentive"

    def update_from_system_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        시스템 이벤트로부터 상태를 갱신한다.

        Args:
            event:
                이벤트 이름 (예: "scene_end", "time_passed")
            payload:
                이벤트 부가 정보
        """

        if event == "scene_end":
            # 장면 종료 시 감정을 중립으로 복귀
            self._state.mood = "neutral"

        elif event == "time_passed":
            # 시간이 지나면 affection이 서서히 감소
            self._state.affection = max(0, self._state.affection - 1)

        elif event == "memory_update" and payload:
            # 외부 요약 모듈이 생성한 기억 요약 반영
            summary = payload.get("summary")
            if isinstance(summary, str):
                self._state.memory_summary = summary

    def reset(self) -> None:
        """
        상태를 초기화한다.
        """
        self._state = CharacterState()
