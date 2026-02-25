"""
Stage 적용 디버그 스크립트

목적:
- stage0 / stage1 / stage2 중 하나를 적용했을 때
  실제로 어떤 모듈이 학습 대상으로 풀렸는지 확인한다.

주의:
- model assets가 있어야 runner가 net_g를 로드할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import torch

from saya_tts.runner import StyleBertVITS2Runner
from saya_tts.train.stages import build_default_stage_plan
from saya_tts.train.freezer import configure_stage, print_stage_report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 실전 net_g 로드 (assets 필요)
    runner = StyleBertVITS2Runner(
        model_dir=Path("model_assets/base_jp_extra"),
        device=device,
    )
    net_g = runner.net_g
    net_g.eval()

    # 2. stage plan 로드
    plan = build_default_stage_plan()

    # 3. 원하는 stage 선택
    stage_name = "stage0"  # stage0 / stage1 / stage2
    stage = plan[stage_name]

    # 4. stage 적용 + param group 생성
    param_groups, resolved = configure_stage(
        net_g=net_g,
        stage_name=stage_name,
        stage=stage,
        strict=False,  # repo 구조 차이를 허용
        freeze_all_first=True, # 안전 모드
    )

    # 5. 결과 리포트 출력
    print_stage_report(net_g, stage_name, stage, resolved)

    # 6. optimizer 예시(실제 훈련 코드에서 사용)
    # optimizer = torch.optim.AdamW(param_groups)

    # param_groups 내용 확인
    print("Optimizer param groups:")
    for g in param_groups:
        print(f"  - {g['name']}: lr={g['lr']} wd={g['weight_decay']} params={len(g['params'])}")


if __name__ == "__main__":
    main()
