"""
StageController

역할:
- step 기준으로 stage 전환을 관리
- stage 변경 시:
  - net_g freeze/unfreeze 재적용
  - optimizer 재생성
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .stages import StageSpec
from .freezer import configure_stage, print_stage_report


@dataclass
class StageSchedule:
    """
    언제(stage_step) 어떤 stage를 적용할지 정의.
    """
    stage_name: str
    start_step: int


class StageController:
    def __init__(
        self,
        *,
        net_g: nn.Module,
        stage_plan: Dict[str, StageSpec],
        schedule: List[StageSchedule],
        device: str,
    ):
        self.net_g = net_g
        self.stage_plan = stage_plan
        self.schedule = sorted(schedule, key=lambda x: x.start_step)
        self.device = device

        self.current_stage_name: str | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def maybe_update_stage(
        self,
        global_step: int,
    ) -> torch.optim.Optimizer:
        """
        global_step에 따라 stage를 변경해야 하면
        freeze/unfreeze + optimizer 재생성 수행.
        """
        # 적용되어야 할 stage 찾기
        target_stage = None
        for s in self.schedule:
            if global_step >= s.start_step:
                target_stage = s.stage_name

        if target_stage is None:
            raise RuntimeError("No stage matched for current step.")

        # stage가 바뀌지 않았다면 기존 optimizer 유지
        if target_stage == self.current_stage_name:
            assert self.optimizer is not None
            return self.optimizer

        # stage 변경
        stage_spec = self.stage_plan[target_stage]

        print(f"\n[StageController] Switching stage -> {target_stage}")

        param_groups, resolved = configure_stage(
            net_g=self.net_g,
            stage_name=target_stage,
            stage=stage_spec,
            strict=False,
            freeze_all_first=True,
        )

        # optimizer 재생성
        optimizer = torch.optim.AdamW(param_groups)

        print_stage_report(
            self.net_g,
            target_stage,
            stage_spec,
            resolved,
        )

        self.current_stage_name = target_stage
        self.optimizer = optimizer
        return optimizer
