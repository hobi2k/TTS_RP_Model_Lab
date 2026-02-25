"""
Style-Bert-VITS2 파인튜닝용 Freeze/Unfreeze 스케줄 (실전용 + 공부용)

왜 freeze/unfreeze가 중요한가?

Style-Bert-VITS2(net_g)는 큰 모듈들이 결합된 모델이다.
- enc_p  : 텍스트(prior) 인코더 (BERT feature 포함)
- dp/sdp : duration predictor (정렬/길이 예측)
- enc_q  : posterior encoder (학습 시에 mel로부터 latent 추정)
- flow   : normalizing flow (latent 변환)
- dec    : decoder (waveform 생성; 품질을 좌우)

파인튜닝에서 가장 흔한 실패는:
1. decoder가 너무 빨리 깨져서 "음질이 붕괴"
2. duration(dp/sdp)이 흔들려서 "발화 리듬이 붕괴"
3. enc_p가 불안정해서 "발음/억양이 붕괴"

따라서 보통은:
- 초반에는 decoder/flow를 보호
- 텍스트/조건 임베딩을 먼저 맞추고
- 정렬(dp/sdp) 안정화 후
- 마지막에 필요한 부분만 점진적으로 unfreeze

이 파일은 그 규칙을 코드로 고정한다.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import torch
import torch.nn as nn


# 0. 유틸: requires_grad 일괄 설정
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """
    module 아래의 모든 파라미터에 requires_grad를 일괄 적용한다.

    Args:
        module: torch.nn.Module
        flag: True면 학습, False면 freeze
    """
    for p in module.parameters():
        p.requires_grad = flag


def count_trainable_params(module: nn.Module) -> int:
    """
    현재 requires_grad=True인 파라미터 개수 합을 반환한다.
    (디버깅/로그용)
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# 1. 스케줄 정의
@dataclass(frozen=True)
class FreezeScheduleConfig:
    """
    Freeze/Unfreeze 스케줄 구성

    stage별로 어느 모듈을 학습할지 정한다.

    stage0 (warmup):
        - speaker/style 관련 임베딩(또는 조건 관련 작은 부분)만 학습
        - 목표: "목소리 아이덴티티"를 모델에 먼저 주입
        - decoder/flow 등은 완전 보호

    stage1 (text alignment):
        - enc_p + duration predictor(dp/sdp) 학습
        - 목표: 텍스트->프레임 정렬을 안정화
        - decoder는 여전히 freeze (음질 붕괴 방지)

    stage2 (latent mapping):
        - enc_q + flow 일부(또는 전부) 학습
        - 목표: mel latent 공간 적응
        - decoder는 아직 제한적으로만 풀거나 그대로 유지

    stage3 (full fine-tune - optional):
        - decoder를 아주 낮은 lr로 부분/전체 unfreeze
        - 목표: 최종 음색/디테일 보정
        - 가장 위험한 단계: 과하면 음질 망가짐
    """
    stage0_steps: int = 2_000
    stage1_steps: int = 20_000
    stage2_steps: int = 50_000
    # stage3는 선택적으로 켠다
    enable_stage3: bool = False
    stage3_steps: int = 80_000

    # decoder를 푸는 경우, decoder lr을 별도로 낮추는게 정석
    decoder_lr_scale: float = 0.1


class Freezer:
    """
    net_g의 모듈들을 freeze/unfreeze 하는 컨트롤러.

    설계 의도:
    - "원본 Style-Bert-VITS2 모듈 이름"을 최대한 유지한다.
    - 다만 포크/버전에 따라 속성명이 다를 수 있으므로
      안전하게 getattr로 접근하며,
      존재 여부를 체크한다.
    """

    def __init__(self, net_g: nn.Module, cfg: FreezeScheduleConfig):
        self.net_g = net_g
        self.cfg = cfg

        # 흔히 등장하는 모듈명을 후보로 정리
        # (JP-Extra 포함해도 개념적으로 동일)
        self.modules = self._resolve_modules()

    def _resolve_modules(self) -> Dict[str, Optional[nn.Module]]:
        """
        net_g에서 주요 모듈들을 찾아서 dict로 반환한다.

        IMPORTANT:
        - 어떤 포크는 dp/sdp가 없거나 하나만 있을 수 있다.
        - 어떤 포크는 enc_q/flow/dec 속성명이 다를 수 있다.
        - 그래서 여기서는 안전하게 후보명을 순회하며 찾는다.
        """

        def pick(*names: str) -> Optional[nn.Module]:
            for n in names:
                m = getattr(self.net_g, n, None)
                if isinstance(m, nn.Module):
                    return m
            return None

        resolved = {
            # prior(text) encoder
            "enc_p": pick("enc_p", "text_encoder", "enc_p_jp_extra"),

            # duration predictors
            "dp": pick("dp", "duration_predictor", "duration_predictor_dp"),
            "sdp": pick("sdp", "stochastic_duration_predictor", "duration_predictor_sdp"),

            # posterior encoder (mel->latent)
            "enc_q": pick("enc_q", "posterior_encoder", "encoder_q"),

            # flow & decoder
            "flow": pick("flow", "flows", "normalizing_flow"),
            "dec": pick("dec", "decoder", "generator"),

            # speaker/style embedding 후보
            # (모델에 따라 embedding이 enc_p 내부에 있을 수도 있고,
            #  별도 테이블로 있을 수도 있다.)
            "emb_g": pick("emb_g", "speaker_emb", "spk_emb", "emb_speaker"),
            "emb_style": pick("emb_style", "style_emb", "emotion_emb", "style_embedding"),
        }
        return resolved


    # 2) Stage별 freeze 정책

    def apply_stage0(self) -> None:
        """
        Stage0: warmup

        목적:
        - "목소리 아이덴티티"를 먼저 맞춘다.
        - 가장 안전한 방법은 speaker/style embedding만 학습하는 것.

        동작:
        - 기본적으로 전체 freeze
        - emb_g / emb_style이 있으면 그것만 unfreeze
        - 없으면 enc_p만 아주 제한적으로 unfreeze하는 대안도 가능
          (여기서는 기본은 embedding-only로 둔다)
        """
        # 1) 전체 freeze
        set_requires_grad(self.net_g, False)

        # 2) speaker/style embedding만 푼다 (존재할 때)
        if self.modules["emb_g"] is not None:
            set_requires_grad(self.modules["emb_g"], True)

        if self.modules["emb_style"] is not None:
            set_requires_grad(self.modules["emb_style"], True)

        # 3) 만약 embedding 모듈이 net_g 바깥에 없고,
        #    "enc_p 내부에만 존재"할 수 있다.
        #    그런 경우 stage0에서 아무 것도 학습이 안 되므로,
        #    최소한 enc_p의 일부를 열어줘야 한다.
        #
        # 여기서는 안전장치로:
        # - emb_g와 emb_style이 둘 다 없으면 enc_p를 연다.
        if self.modules["emb_g"] is None and self.modules["emb_style"] is None:
            if self.modules["enc_p"] is not None:
                set_requires_grad(self.modules["enc_p"], True)

    def apply_stage1(self) -> None:
        """
        Stage1: text alignment

        목적:
        - 텍스트->프레임 길이/정렬(duration)을 안정화
        - 여기서 음질(decoder)을 건드리면 망가질 확률이 큼

        동작:
        - enc_p + dp(+sdp)를 학습
        - enc_q/flow/dec는 freeze 유지
        """
        set_requires_grad(self.net_g, False)

        if self.modules["enc_p"] is not None:
            set_requires_grad(self.modules["enc_p"], True)

        if self.modules["dp"] is not None:
            set_requires_grad(self.modules["dp"], True)

        if self.modules["sdp"] is not None:
            set_requires_grad(self.modules["sdp"], True)

        # speaker/style embedding이 별도라면 함께 학습해도 무방
        if self.modules["emb_g"] is not None:
            set_requires_grad(self.modules["emb_g"], True)

        if self.modules["emb_style"] is not None:
            set_requires_grad(self.modules["emb_style"], True)

    def apply_stage2(self) -> None:
        """
        Stage2: latent mapping

        목적:
        - posterior encoder(enc_q)와 flow를 적응시켜
          mel->latent 변환/복원을 데이터 도메인에 맞춘다.
        - decoder는 아직 보호(또는 아주 제한적으로만 푸는게 안전)

        동작:
        - enc_p, dp/sdp는 계속 학습 (정렬이 무너지면 안 됨)
        - enc_q, flow 학습
        - dec는 기본 freeze 유지
        """
        set_requires_grad(self.net_g, False)

        for name in ("enc_p", "dp", "sdp", "enc_q", "flow", "emb_g", "emb_style"):
            m = self.modules.get(name)
            if m is not None:
                set_requires_grad(m, True)

        # decoder(dec)는 기본적으로 계속 freeze
        # (enable_stage3에서만 푼다)

    def apply_stage3(self) -> None:
        """
        Stage3: full fine-tune (optional)

        목적:
        - 최종 음색/디테일을 decoder까지 포함해 미세 조정
        - 가장 위험함: 너무 빨리/강하게 풀면 음질 붕괴

        동작:
        - 대부분 모듈을 학습
        - decoder도 학습 (하지만 lr을 별도로 낮추는 게 정석)
        """
        set_requires_grad(self.net_g, True)


    # 3) step 기반 스케줄 적용

    def apply_by_step(self, global_step: int) -> str:
        """
        현재 global_step에 따라 stage를 결정하고
        해당 stage의 freeze 정책을 적용한다.

        Returns:
            stage_name (로그용)
        """
        if global_step < self.cfg.stage0_steps:
            self.apply_stage0()
            return "stage0_warmup"

        if global_step < self.cfg.stage1_steps:
            self.apply_stage1()
            return "stage1_alignment"

        if global_step < self.cfg.stage2_steps:
            self.apply_stage2()
            return "stage2_latent"

        if self.cfg.enable_stage3 and global_step < self.cfg.stage3_steps:
            self.apply_stage3()
            return "stage3_full"

        # stage 종료 이후에는 stage2 유지(보수적으로)
        self.apply_stage2()
        return "stage2_latent_hold"
