"""
freeze / unfreeze 설계도(Stage Plan)

원칙:
- "어떤 데이터를 쓸지" 아직 몰라도, 모델(net_g)만 있으면 stage 적용 가능해야 한다.

Stage 개념:
- Stage 0: "사야처럼 말하게 만들기" 최소 업데이트 (가장 안전)
- Stage 1: "사야 음색" 붙이기 (가장 실전적)
- Stage 2: "리듬/억양"까지 사야화 (가장 위험, 신중)

이 파일은 '정책'만 가진다.
실제 freeze 적용 / optimizer 생성은 freezer.py가 담당한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ParamGroupSpec:
    """
    Optimizer param group을 만들기 위한 스펙.

    name:
        그룹 이름. 로깅/디버깅에 사용.
    module_paths:
        net_g에서 찾아낼 모듈 경로 후보들.
        예: ["emb_g"] or ["dec", "decoder"] 등.
        - 실제 attribute 이름이 repo마다 다를 수 있으므로 '후보 리스트'를 둔다.
    lr:
        이 그룹에 적용할 learning rate.
    weight_decay:
        weight decay(필요 시).
    """
    name: str
    module_paths: List[str]
    lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True)
class StageSpec:
    """
    Stage 하나의 규칙.

    freeze_modules:
        이 stage에서 "반드시" 얼릴 모듈들(후보 이름 리스트)
    unfreeze_groups:
        이 stage에서 학습할 param group 정의(ParamGroupSpec 리스트)
    note:
        사람이 읽기 위한 설명(로그 출력용)
    """
    freeze_modules: List[str]
    unfreeze_groups: List[ParamGroupSpec]
    note: str


def build_default_stage_plan() -> Dict[str, StageSpec]:
    """
    JP-Extra 계열을 염두에 둔 기본 stage plan.

    이름 후보 규칙(중요):
    - Style-Bert-VITS2의 구현체마다 속성명이 조금 다르다.
    - 그래서 "가능성이 높은 이름 후보"를 여러 개 적는다.
    - freezer.py가 net_g에서 실제 존재하는 것을 찾아서 적용한다.
    """
    # 공통적으로 '웬만하면 얼려야 안전한 것들' 후보
    # (실제 모델 속성명이 다르면 freezer가 무시하거나, strict 모드면 에러를 낸다.)
    base_freeze_candidates = [
        # 텍스트 인코더(= enc_p)
        "enc_p",
        "text_encoder",  # 일부 구현체
        # duration predictors
        "dp",
        "duration_predictor",
        "sdp",
        "stochastic_duration_predictor",
        # flow
        "flow",
        # decoder / vocoder
        "dec",
        "decoder",
        # BERT 계열은 보통 net_g 바깥(infer.py)에서 로드되므로 여기엔 없을 수 있음
    ]

    # Stage 0: 최소 업데이트
    # - speaker embedding + style 관련 경로만 학습
    stage0 = StageSpec(
        freeze_modules=base_freeze_candidates,
        unfreeze_groups=[
            ParamGroupSpec(
                name="speaker_embedding",
                module_paths=[
                    "emb_g", # 가장 흔한 이름
                    "speaker_emb", # 후보
                    "spk_emb", # 후보
                ],
                lr=1e-3,
                weight_decay=0.0,
            ),
            # style_vec은 보통 "입력 벡터"라 파라미터가 아니다.
            # 하지만 모델 내부에 style projection(선형층)이 있는 경우가 있어 후보를 둔다.
            ParamGroupSpec(
                name="style_projection",
                module_paths=[
                    "style_proj",
                    "style_linear",
                    "style_embedding",
                    "emb_style",
                ],
                lr=5e-4,
                weight_decay=0.0,
            ),
            # 옵션: decoder 최종단(있다면)만 아주 낮은 lr로 풀 수 있음
            ParamGroupSpec(
                name="decoder_last",
                module_paths=[
                    "dec.post", # 후보(구조에 따라)
                    "dec.out_proj", # 후보
                    "decoder.post",
                    "decoder.out_proj",
                ],
                lr=5e-6,
                weight_decay=0.0,
            ),
        ],
        note="Stage 0: speaker/style 경로만 학습 (가장 안전, 음질 유지 최우선)",
    )

    # Stage 1: 음색/질감 붙이기
    # - prior_head(또는 enc_p 일부 head) + decoder 후반을 주로 학습
    stage1 = StageSpec(
        freeze_modules=[
            # Stage1에서도 웬만하면 고정하는 것
            "flow",
            "sdp",
            "stochastic_duration_predictor",
            # BERT는 net_g 외부라 여기 없음 (대부분)
        ],
        unfreeze_groups=[
            ParamGroupSpec(
                name="speaker_embedding",
                module_paths=["emb_g", "speaker_emb", "spk_emb"],
                lr=5e-4,
            ),
            ParamGroupSpec(
                name="style_projection",
                module_paths=["style_proj", "style_linear", "style_embedding", "emb_style"],
                lr=3e-4,
            ),
            # enc_p 전체를 푸는 대신 "head"만 푸는 전략
            ParamGroupSpec(
                name="prior_head",
                module_paths=[
                    "enc_p.prior_head",
                    "enc_p.proj", # 후보
                    "enc_p.prior", # 후보
                    "prior_head", # 후보(분리 구조)
                ],
                lr=1e-4,
            ),
            # decoder: 전체를 풀 수도 있지만, 안전하게는 후반부/전체를 낮은 lr로
            ParamGroupSpec(
                name="decoder",
                module_paths=["dec", "decoder"],
                lr=2e-5,
                weight_decay=0.0,
            ),
        ],
        note="Stage 1: prior head + decoder 중심으로 음색/질감 학습",
    )

    # Stage 2: 리듬/억양까지 맞추기 (위험)
    # - dp를 먼저 풀고, 필요하면 enc_p 상단 레이어만 일부 해제
    stage2 = StageSpec(
        freeze_modules=[
            "flow",
            # sdp는 매우 불안정할 수 있어 기본적으로는 여전히 freeze 권장
        ],
        unfreeze_groups=[
            ParamGroupSpec(
                name="speaker_embedding",
                module_paths=["emb_g", "speaker_emb", "spk_emb"],
                lr=3e-4,
            ),
            ParamGroupSpec(
                name="style_projection",
                module_paths=["style_proj", "style_linear", "style_embedding", "emb_style"],
                lr=2e-4,
            ),
            ParamGroupSpec(
                name="prior_head",
                module_paths=["enc_p.prior_head", "prior_head", "enc_p.proj", "enc_p.prior"],
                lr=5e-5,
            ),
            ParamGroupSpec(
                name="decoder",
                module_paths=["dec", "decoder"],
                lr=2e-5,
            ),
            ParamGroupSpec(
                name="duration_dp",
                module_paths=["dp", "duration_predictor"],
                lr=3e-5,
            ),
            # 옵션: sdp를 아주 낮은 lr로 풀고 싶다면 아래를 켠다.
            ParamGroupSpec(
                name="duration_sdp_optional",
                module_paths=["sdp", "stochastic_duration_predictor"],
                lr=5e-6,
            ),
        ],
        note="Stage 2: dp(길이)까지 학습하여 리듬/억양 보정 (가장 신중)",
    )

    return {
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
    }
