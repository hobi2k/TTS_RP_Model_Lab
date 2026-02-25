"""
JP-Extra enc_p 이후 추론 엔진

역할:
- Style-Bert-VITS2 JP-Extra 모델의
  SynthesizerTrnJPExtra.infer() 내부에서
  enc_p(TextEncoderPrior) 이후에 수행되는 모든 계산을
  "함수 단위로 명시적으로 분리"한 구현이다.

주의:
- Runner / pipeline 에서 직접 호출하지 않는다.
- 실제 실행 진입점은 style_bert_vits2.models.infer.infer()
- 이 파일은 infer.py 내부 로직을 '드러내기' 위한 엔진이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from style_bert_vits2.models import commons


# enc_p 출력 컨테이너
@dataclass
class EncPOutJPExtra:
    """
    JP-Extra TextEncoder(enc_p)의 출력 묶음.

    이 구조는 'infer.py 내부 tuple unpacking'을
    명시적인 구조체로 드러낸 것이다.

    각 텐서의 의미:

    x:
        [B, H, T_text]
        - TextEncoderPrior의 최종 출력
        - Transformer를 거친 텍스트 히든 시퀀스

    m_p:
        [B, D, T_text]
        - prior 분포의 평균 (mean)
        - 텍스트 기준 잠재 변수 z_p의 평균

    logs_p:
        [B, D, T_text]
        - prior 분포의 log-표준편차
        - std = exp(logs_p)

    x_mask:
        [B, 1, T_text]
        - 텍스트 padding 마스크
        - 유효 토큰 = 1, padding = 0

    g:
        [B, gin, 1]
        - 화자 / 스타일 조건 임베딩
        - JP-Extra에서는 speaker + style이 합쳐진 조건
    """
    x: torch.Tensor
    m_p: torch.Tensor
    logs_p: torch.Tensor
    x_mask: torch.Tensor
    g: torch.Tensor


# 최종 추론 결과 컨테이너
@dataclass
class InferOutJPExtra:
    """
    JP-Extra 최종 추론 결과 묶음.

    o:
        [B, 1, T_audio]
        - decoder가 생성한 최종 waveform

    attn:
        [B, 1, T_y, T_text]
        - duration 기반 monotonic alignment 경로
        - 각 텍스트 토큰이 어떤 프레임을 담당하는지 나타냄

    y_mask:
        [B, 1, T_y]
        - 오디오 프레임 padding 마스크

    debug_pack:
        (z, z_p, m_p_exp, logs_p_exp)
        - 실전 추론에는 사용하지 않지만
        - 분포 / flow / 정렬 분석을 위한 중간 결과
    """
    o: torch.Tensor
    attn: torch.Tensor
    y_mask: torch.Tensor
    debug_pack: Tuple[torch.Tensor, ...]


# enc_p 이후 JP-Extra 추론 로직
@torch.inference_mode()
def infer_after_enc_p_jp_extra(
    net_g,
    enc: EncPOutJPExtra,
    *,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
    noise_scale_w: float = 0.8,
    max_len: Optional[int] = None,
    sdp_ratio: float = 0.0,
) -> InferOutJPExtra:
    """
    JP-Extra SynthesizerTrnJPExtra.infer()에서
    enc_p(TextEncoderPrior) 이후 부분을 그대로 재현한 함수.

    이 함수가 하는 일 (요약):

    1. 텍스트 히든(x) -> duration 예측 (sdp / dp)
    2. duration -> frame 단위 정렬(attn) 생성
    3. prior(m_p, logs_p)를 frame-time으로 확장
    4. prior에서 latent z_p 샘플링
    5. normalizing flow 역변환 (z_p -> z)
    6. decoder로 waveform 생성

    주의:
    - 이 함수는 net_g.infer()의 내부 구성 요소다.
    - Runner에서는 직접 호출하지 않는다.
    """
    # enc_p 출력 unpack
    x = enc.x
    m_p = enc.m_p
    logs_p = enc.logs_p
    x_mask = enc.x_mask
    g = enc.g

    # 1. Duration Prediction (텍스트 -> 프레임 길이)
    """
    logw:
        각 텍스트 토큰이 차지할 프레임 수의 '로그 값'

    JP-Extra는 두 종류의 길이 예측기를 혼합한다.

    - sdp (Stochastic Duration Predictor):
        확률적 길이 예측
        reverse=True -> 샘플링 방향
        noise_scale_w로 랜덤성 조절

    - dp (Deterministic Duration Predictor):
        평균적인 길이 예측
        항상 안정적인 결과

    sdp_ratio:
        0.0 -> 완전 결정적(dp)
        1.0 -> 완전 확률적(sdp)
    """
    logw = (
        net_g.sdp(
            x, # [B, H, T_text]
            x_mask, # [B, 1, T_text]
            g=g, # 조건 임베딩
            reverse=True,
            noise_scale=noise_scale_w,
        ) * sdp_ratio
        + net_g.dp(
            x,
            x_mask,
            g=g,
        ) * (1.0 - sdp_ratio)
    )

    # 2. duration -> 정수 프레임 길이
    """
    w:
        exp(logw)로 양수 duration 확보
        length_scale로 말 빠르기 조절

    w_ceil:
        실제 프레임 개수 (정수)
        generate_path는 정수를 요구한다.
    """
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)

    # 3. 프레임 마스크 생성
    """
    y_lengths:
        각 배치 샘플의 총 프레임 길이
        최소 1 프레임 보장

    y_mask:
        [B, 1, T_y]
        padding 프레임 = 0
    """
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(
        commons.sequence_mask(y_lengths, None),
        1
    ).to(x_mask.dtype)

    # 4. Monotonic Alignment 생성
    """
    attn_mask:
        텍스트 마스크 × 프레임 마스크

    attn:
        duration 기반 정렬 경로
        각 텍스트 토큰이 어떤 프레임 구간을 담당하는지 명시
    """
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    # 5. prior를 frame-time으로 확장
    """
    m_p, logs_p:
        [B, D, T_text]

    attn을 사용해:
        [B, D, T_y]로 확장
    """
    m_p_exp = torch.matmul(
        attn.squeeze(1),
        m_p.transpose(1, 2)
    ).transpose(1, 2)

    logs_p_exp = torch.matmul(
        attn.squeeze(1),
        logs_p.transpose(1, 2)
    ).transpose(1, 2)

    # 6. prior에서 latent 샘플링
    """
    z_p = m_p_exp + ε * exp(logs_p_exp) * noise_scale

    noise_scale:
        클수록 발화 다양성 증가
        작을수록 안정적인 발화
    """
    z_p = (
        m_p_exp
        + torch.randn_like(m_p_exp)
        * torch.exp(logs_p_exp)
        * noise_scale
    )

    # 7. Flow 역변환 (prior -> decoder latent)
    z = net_g.flow(
        z_p,
        y_mask,
        g=g,
        reverse=True
    )

    # 8. Decoder -> waveform
    o = net_g.dec(
        (z * y_mask)[:, :, :max_len],
        g=g
    )

    return InferOutJPExtra(
        o=o,
        attn=attn,
        y_mask=y_mask,
        debug_pack=(z, z_p, m_p_exp, logs_p_exp),
    )
