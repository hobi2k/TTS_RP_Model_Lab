"""
v7_qwen_fsm_engine.py

Qwen VN FSM Engine
- ScenarioBook 기반 전역 규칙 파싱
- 상태 FSM + 전역 헌법 병합

"""

from __future__ import annotations

import re
import yaml
from typing import Dict, Any


class QwenFSMEngine:
    """FSM 규칙을 해석하고 상태 전이를 계산하는 엔진 클래스"""


    def __init__(self, fsm_yaml_path: str, system_lore: str):
        """내부 헬퍼로 `__init__` 계산 절차를 수행한다."""
        with open(fsm_yaml_path, "r", encoding="utf-8") as f:
            self.spec = yaml.safe_load(f)

        self.state: str = self.spec["initial_state"]
        self.states: Dict[str, Any] = self.spec.get("states", {})
        self.transitions: Dict[str, Any] = self.spec.get("transitions", {})
        self.state_flags: Dict[str, Dict[str, Any]] = self.spec.get("state_flags", {})

 # 시나리오북에서 뽑은 절대 규칙
        self.global_flags: Dict[str, Any] = self._parse_global_flags(system_lore)
        self.relation_status: str = self._parse_relation_status(system_lore)


    def get_state(self) -> str:
        """
        현재 FSM 상태 조회

        내부 상태 머신이 유지 중인 현재 상태 이름을 반환한다.

        Args:
            없음.

        Returns:
            str: 현재 상태 키.
        """
        return self.state

    def get_flags(self) -> Dict[str, Any]:
        """
        상태 플래그 병합 조회

        상태별 플래그와 시나리오북 전역 플래그를 합쳐
        현재 턴에서 사용할 최종 플래그 딕셔너리를 만든다.

        Args:
            없음.

        Returns:
            Dict[str, Any]: 현재 상태에 적용할 플래그 집합.
        """
        flags: Dict[str, Any] = {}

        # 1) 상태별 규칙
        flags.update(self.state_flags.get(self.state, {}))

        # 2) 전역 규칙은 무조건 덮어씀 (헌법)
        for k, v in self.global_flags.items():
            flags[k] = v

        if self.relation_status:
            flags["relation_status"] = self.relation_status

        return flags

    def step(self, acc: Dict[str, Any]) -> bool:
        """
        FSM 한 단계 전이 실행

        누적 신호(acc)를 사용해 현재 상태의 전이 조건을 검사하고,
        만족하는 전이가 있으면 상태를 이동시킨다.

        Args:
            acc: 평가 신호 및 장르 정보가 포함된 입력 딕셔너리.

        Returns:
            bool: 상태 전이가 발생하면 True, 아니면 False.
        """
        rules = self.transitions.get(self.state, [])

        for rule in rules:
            conds = rule.get("if", {})
            if self._check_conditions(conds, acc):
                self.state = rule["to"]
                self.relation_status = self._infer_relation_status(acc, self.relation_status)
                return True

        self.relation_status = self._infer_relation_status(acc, self.relation_status)
        return False


    def _parse_global_flags(self, system_lore: str) -> Dict[str, Any]:
        """내부 헬퍼로 `_parse_global_flags` 계산 절차를 수행한다."""
        flags: Dict[str, Any] = {}

        # 성행위 가능 여부 (헌법)
        # 예시 허용 포맷:
        # - 성행위 가능 여부 (True)
        # - 성행위 가능 여부 (False)
        # - 성행위 가능 여부: True
        # - 성행위 가능 여부 True 시 ...
        m = re.search(
            r"성행위\s*가능\s*여부[^T|F]*(True|False)",
            system_lore
        )
        if m:
            flags["allow_sexual"] = (m.group(1) == "True")
        else:
            flags["allow_sexual"] = False

        return flags

    @staticmethod
    def _parse_relation_status(system_lore: str) -> str:
        """내부 헬퍼로 `_parse_relation_status` 계산 절차를 수행한다."""
        text = system_lore or ""
        m7 = re.search(r"\n\s*7\.\s*가장\s*최근\s*상호작용.*?(?=\n\s*\d+\.\s|\Z)", text, flags=re.DOTALL)
        block = m7.group(0) if m7 else text
        m = re.search(r"관계\s*상태\s*[:：]\s*(적대|거리감|친밀|사랑)", block)
        return m.group(1) if m else ""

    @staticmethod
    def _infer_relation_status(acc: Dict[str, Any], fallback: str = "") -> str:
        """내부 헬퍼로 `_infer_relation_status` 계산 절차를 수행한다."""
        try:
            intimacy = int(acc.get("intimacy", 0))
        except Exception:
            intimacy = 0
        try:
            threat = int(acc.get("threat", 0))
        except Exception:
            threat = 0

        if threat >= 2 and intimacy <= 1:
            return "적대"
        if intimacy >= 3:
            return "사랑"
        if intimacy == 2:
            return "친밀"
        if intimacy <= 1:
            return "거리감"
        return fallback

    def _check_conditions(self, conds: Dict[str, str], acc: Dict[str, Any]) -> bool:
        """내부 헬퍼로 `_check_conditions` 계산 절차를 수행한다."""
        for key, expr in conds.items():
            value = acc.get(key, None)

            # string 조건
            if isinstance(value, str):
                if expr.startswith("=="):
                    if value != expr[2:]:
                        return False
                else:
                    if value != expr:
                        return False
                continue

            # boolean 조건
            if isinstance(value, bool):
                expected = expr.lower() == "true"
                if value != expected:
                    return False
                continue

            # numeric 조건
            if not self._eval_expr(value, expr):
                return False

        return True

    @staticmethod
    def _eval_expr(value: int | None, expr: str) -> bool:
        """내부 헬퍼로 `_eval_expr` 계산 절차를 수행한다."""
        if value is None:
            return False

        if expr.startswith(">="):
            return value >= int(expr[2:])
        if expr.startswith("=="):
            return value == int(expr[2:])
        if expr.startswith("<="):
            return value <= int(expr[2:])
        if expr.startswith(">"):
            return value > int(expr[1:])
        if expr.startswith("<"):
            return value < int(expr[1:])

        raise ValueError(f"Invalid FSM expr: {expr}")
