from __future__ import annotations

"""
enc_p(Text Encoder / Prior Encoder) 계약(Contract) 검증 스크립트

이 스크립트의 목적:
- Style-Bert-VITS2 (JP-Extra)의 enc_p(Text Encoder)가
  어떤 입력을 받고, 어떤 출력을 내는지 "계약 수준"에서 확정한다.
- 이후 enc_p를 직접 구현할 때, 이 계약을 절대 깨지 않도록 기준으로 사용한다.

핵심 개념 정리:
- enc_p = Text Encoder = Prior Encoder
- 역할: 텍스트 조건 → 잠재변수 z_p의 분포 파라미터 (m_p, logs_p) 생성
"""

import numpy as np
import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g

from saya_tts.text.cleaner import clean_text_for_tts
from saya_tts.bert.extractor import BertFeatureExtractor


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    """
    텐서의 핵심 정보를 사람이 읽기 쉬운 형태로 출력한다.

    출력 정보:
    - shape : 텐서 차원
    - dtype : 자료형
    - device : CPU / CUDA
    - min/max/mean : 값 분포 (NaN/Inf 탐지에 유용)
    """
    print(
        f"{name:<10} | "
        f"shape={tuple(tensor.shape)} | "
        f"dtype={tensor.dtype} | "
        f"device={tensor.device} | "
        f"min={tensor.min().item():.4f} | "
        f"max={tensor.max().item():.4f} | "
        f"mean={tensor.mean().item():.4f}"
    )


@torch.inference_mode()
def main() -> None:
    # 1. 사용자 설정 영역
    # Style-Bert-VITS2 JP-Extra 모델 파일 경로
    model_path = "model_assets/saya_model/model.safetensors"

    # 모델 config.json 경로 (HyperParameters 로드용)
    config_path = "model_assets/saya_model/config.json"

    # 스타일 벡터 파일 경로 (style_vectors.npy)
    style_vectors_path = "model_assets/saya_model/style_vectors.npy"

    # 테스트용 입력 텍스트
    text = "こんにちは。私はサヤです。"

    # 언어 설정 (JP-Extra이므로 일본어)
    language = Languages.JP

    # 화자 ID (single-speaker 모델이면 보통 0)
    speaker_id = 0

    # 사용할 스타일 벡터 인덱스 (예: Neutral = 0)
    style_index = 0

    # 실행 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. HyperParameters / net_g 로드
    # config.json -> HyperParameters 객체
    hps = HyperParameters.load_from_json(config_path)

    # JP-Extra 여부 확인 (버전 문자열로 판단)
    if not str(hps.version).endswith("JP-Extra"):
        raise RuntimeError(
            f"이 스크립트는 JP-Extra 기준이다. 현재 version={hps.version}"
        )

    # Style-Bert-VITS2 생성기(SynthesizerTrnJPExtra) 로드
    net_g = get_net_g(
        model_path=model_path,
        version=hps.version,
        device=device,
        hps=hps,
    )

    # 3. 스타일 벡터 로드
    # style_vectors.npy: [num_styles, style_dim]
    style_vectors = np.load(style_vectors_path).astype(np.float32)

    # 선택한 스타일 벡터
    style_vec = torch.from_numpy(style_vectors[style_index]).to(device)
    # [B, style_dim] 형태로 맞추기 위해 batch 차원 추가
    style_vec = style_vec.unsqueeze(0)

    # 4. 텍스트 -> 음소 / 톤 / word2ph (Frontend 1)
    phones, tones, word2ph = clean_text_for_tts(
        text=text,
        language=language,
    )

    # 음소 시퀀스 길이 (T_text)
    text_length = len(phones)

    # 5. BERT 특징 추출 (Frontend 2)
    # JP-Extra에서는 일본어 BERT 특징만 사용
    bert_extractor = BertFeatureExtractor(
        language=language,
        device=device,
    )

    # bert_feats: [D_bert, T_text] (보통 D_bert=1024)
    bert_feats = bert_extractor.extract(
        phones=phones,
        word2ph=word2ph,
    )

    # 6. enc_p 입력 텐서 구성 (JP-Extra 계약)
    # phoneme ids -> [B, T]
    x = torch.tensor(phones, dtype=torch.long, device=device).unsqueeze(0)

    # tone ids -> [B, T]
    tone = torch.tensor(tones, dtype=torch.long, device=device).unsqueeze(0)

    # language ids -> [B, T]
    # 실제 값은 모델에 따라 다를 수 있으므로,
    # 실패 시 SBV2의 get_text() 로직으로 대체해야 한다.
    lang_id = 1  # 일본어 ID (모델별 상이 가능)
    lang = torch.full_like(x, fill_value=lang_id)

    # 유효 길이 -> [B]
    x_lengths = torch.tensor([text_length], dtype=torch.long, device=device)

    # BERT 특징 -> [B, D_bert, T]
    bert = bert_feats.to(device).unsqueeze(0)

    # 화자 임베딩 g -> [B, gin, 1]
    speaker_tensor = torch.tensor([speaker_id], dtype=torch.long, device=device)
    g = net_g.emb_g(speaker_tensor).unsqueeze(-1)

    # 7. enc_p 호출 (JP-Extra Text Encoder)
    # enc_p의 역할:
    # - 텍스트 조건으로부터
    #   잠재 분포의 평균(m_p)과 로그표준편차(logs_p)를 생성
    x_enc, m_p, logs_p, x_mask = net_g.enc_p(
        x,
        x_lengths,
        tone,
        lang,
        bert,
        style_vec,
        g=g,
    )

    # 8. 출력 계약 검증
    print("[enc_p OUTPUT CONTRACT CHECK]")
    print_tensor_info("x_enc", x_enc)
    print_tensor_info("m_p", m_p)
    print_tensor_info("logs_p", logs_p)
    print_tensor_info("x_mask", x_mask)
    print_tensor_info("g", g)

    # 차원 계약
    assert x_enc.ndim == 3, "x_enc는 [B, H, T] 형태여야 한다."
    assert m_p.ndim == 3, "m_p는 [B, D, T] 형태여야 한다."
    assert logs_p.ndim == 3, "logs_p는 [B, D, T] 형태여야 한다."
    assert x_mask.ndim == 3, "x_mask는 [B, 1, T] 형태여야 한다."

    # 시간축(T) 계약
    assert x_enc.shape[-1] == text_length, "x_enc의 T가 입력 phoneme 길이와 일치해야 한다."
    assert m_p.shape[-1] == text_length, "m_p의 T가 x_enc의 T와 같아야 한다."
    assert logs_p.shape[-1] == text_length, "logs_p의 T가 x_enc의 T와 같아야 한다."
    assert x_mask.shape[-1] == text_length, "x_mask의 T가 x_enc의 T와 같아야 한다."

    # NaN / Inf 검증
    assert torch.isfinite(x_enc).all(), "x_enc에 NaN 또는 Inf가 포함되어 있다."
    assert torch.isfinite(m_p).all(), "m_p에 NaN 또는 Inf가 포함되어 있다."
    assert torch.isfinite(logs_p).all(), "logs_p에 NaN 또는 Inf가 포함되어 있다."

    print("enc_p Contract 검증 완료 (JP-Extra 기준)")


if __name__ == "__main__":
    main()
