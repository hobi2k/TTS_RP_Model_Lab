"""
LLM Router - Minimal System Layer

목적:
- 현재 상태, 입력 특성, 시스템 정책에 따라
  어떤 LLM 엔진을 사용할지 결정한다.
- 상위 로직은 "어떤 모델을 썼는지"를 알 필요가 없다.

설계 원칙:
- 라우팅 규칙은 명시적이다.
- 모델 호출 로직은 포함하지 않는다.
- 엔진 교체는 이 파일에서만 발생한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# 라우팅 정책 정의
@dataclass
class RoutingDecision:
    """
    라우팅 결과 컨테이너.

    engine_name:
        사용할 엔진의 논리적 이름.
        예: "qwen_3b", "qwen_7b"
    """
    engine_name: str
    reason: Optional[str] = None


# 라우터
class Router:
    """
    LLM 라우터.

    입력:
    - user_text
    - character_state
    - 시스템 정책

    출력:
    - RoutingDecision
    """

    def __init__(
        self,
        *,
        default_engine: str = "qwen_3b",
        long_context_engine: str = "qwen_7b",
        complexity_threshold: int = 120,
    ) -> None:
        """
        Args:
            default_engine:
                기본적으로 사용할 엔진 이름.

            long_context_engine:
                복잡한 요청에 사용할 엔진 이름.

            complexity_threshold:
                입력 길이가 이 값을 초과하면
                long_context_engine을 선택한다.
        """
        self.default_engine = default_engine
        self.long_context_engine = long_context_engine
        self.complexity_threshold = complexity_threshold

    # 공개 API
    def route(self, user_text: str) -> RoutingDecision:
        """
        어떤 LLM 엔진을 사용할지 결정한다.

        현재 최소 구현 기준:
        - 입력 길이 기반 규칙만 사용한다.
        """

        text_length = len(user_text)

        # 규칙 1:
        # 입력이 길면 더 큰 모델 사용
        if text_length >= self.complexity_threshold:
            return RoutingDecision(
                engine_name=self.long_context_engine,
                reason="input_length_exceeded_threshold",
            )

        # 기본 규칙
        return RoutingDecision(
            engine_name=self.default_engine,
            reason="default_route",
        )
