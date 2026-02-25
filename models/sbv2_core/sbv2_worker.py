# models/sbv2_core/sbv2_worker.py
"""
SBV2 Worker Process (Style-BERT-VITS2 Inference Worker)

이 파일의 목적:
- Style-BERT-VITS2 TTS 모델을 "프로세스 단위"로 한 번만 로드
- stdin으로 텍스트(JSON)를 입력받아
- 음성(wav)을 생성하고
- 결과 파일 경로를 stdout(JSON)으로 반환

왜 이런 구조인가?
1) 모델 로드는 매우 무겁다 (수 GB, 수 초~수십 초)
2) 매 요청마다 모델을 다시 로드하면 실시간 서비스 불가능
3) 따라서:
   - 워커 프로세스 1회 실행
   - 이후 stdin/stdout으로만 통신
   - Gradio / FastAPI / 게임 엔진 / Ren'Py 등과 연동 가능
"""

from __future__ import annotations  # Python 3.7+ 타입 힌트 전방 참조 허용

# 표준 라이브러리
import sys        # stdin / stdout 스트림 접근
import json       # 프로세스 간 통신을 위한 JSON 직렬화
from pathlib import Path  # OS 독립적인 경로 처리

# 서드파티 라이브러리
import numpy as np        # 스타일 벡터 처리
import torch              # PyTorch (모델 추론)
import soundfile as sf    # wav 파일 저장

# Style-BERT-VITS2 내부 모듈
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, infer


# 경로 설정

# 현재 파일 기준 디렉토리 (sbv2_worker.py가 있는 위치)
BASE_DIR = Path(__file__).resolve().parent

# 학습된 TTS 모델 가중치
MODEL_PATH = (
    BASE_DIR
    / "model_assets"
    / "tts"
    / "Saya"
    / "Saya_e126_s117000.safetensors"
)

# 스타일 벡터 파일
# - shape: (num_styles, style_dim)
STYLE_PATH = (
    BASE_DIR
    / "model_assets"
    / "tts"
    / "Saya"
    / "style_vectors.npy"
)

# 출력 wav 저장 디렉토리
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CUDA 사용 가능 여부에 따라 디바이스 결정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_config_path() -> Path:
    """모델 폴더에서 config JSON을 우선 탐색하고 없으면 기본 경로를 사용한다."""
    model_dir = MODEL_PATH.parent
    candidates = sorted(model_dir.glob("config.json"))
    if candidates:
        return candidates[0]
    return BASE_DIR / "configs" / "config_jp_extra.json"


# TTS 엔진 로드 (프로세스 시작 시 단 1회)
def load_engine():
    """
    Style-BERT-VITS2 추론에 필요한 모든 리소스를
    "한 번만" 로드하여 dict로 반환

    반환되는 구성요소:
    - hps          : 하이퍼파라미터 객체
    - net_g        : Generator (Text → Mel → Wave)
    - style_vectors: 전체 스타일 벡터 배열
    - mean_style   : 기본(중립) 스타일 벡터
    - sid          : 화자 ID (멀티스피커 모델 대비)
    """

    # 1) 하이퍼파라미터 로드
    config_path = resolve_config_path()
    print(f"[SBV2_WORKER] config 경로: {config_path}", flush=True)
    hps = HyperParameters.load_from_json(str(config_path))

    # 2) Generator 네트워크 로드
    # get_net_g 내부에서:
    # - 모델 구조 생성
    # - safetensors 가중치 로드
    # - device 이동
    net_g = get_net_g(
        model_path=str(MODEL_PATH),
        version=hps.version,
        device=DEVICE,
        hps=hps,
    )

    # 3) 스타일 벡터 로드
    # shape 예:
    #   (N_styles, style_dim)
    style_vectors = np.load(STYLE_PATH)

    # 관례적으로 index 0을 mean / neutral style로 사용
    mean_style = style_vectors[0]

    # 4) 화자 ID 설정
    # 단일 화자 모델이라도 내부 구조상 sid 필요하다.
    # 학습/병합 설정에 따라 spk2id 키 이름 대소문자/표기가 달라질 수 있어
    # 안전하게 자동 매핑한다.
    spk2id = dict(hps.data.spk2id or {})
    if not spk2id:
        raise RuntimeError("spk2id가 비어 있어 화자 ID를 결정할 수 없습니다.")

    sid = None
    preferred_keys = ("saya", "mai")
    lower_map = {str(k).lower(): v for k, v in spk2id.items()}

    # 우선순위 1: 선호 키 정확/대소문자 무시 매칭
    for key in preferred_keys:
        if key in lower_map:
            sid = lower_map[key]
            print(
                f"[SBV2_WORKER] 화자 키 '{key}'를 사용합니다. "
                f"사용 가능한 키: {list(spk2id.keys())}",
                flush=True,
            )
            break

    # 우선순위 2: 부분 일치
    if sid is None:
        for k, v in spk2id.items():
            lk = str(k).lower()
            if any(pk in lk for pk in preferred_keys) or "사야" in str(k):
                sid = v
                print(
                    f"[SBV2_WORKER] 화자 키 '{k}'를 부분 일치로 선택했습니다.",
                    flush=True,
                )
                break

    # 우선순위 4: 첫 화자 fallback
    if sid is None:
        sid = next(iter(spk2id.values()))
        print(
            f"[SBV2_WORKER] 경고: 'saya' 화자 키를 찾지 못해 첫 화자 ID({sid})를 사용합니다. "
            f"사용 가능한 키: {list(spk2id.keys())}",
            flush=True,
        )

    return {
        "hps": hps,
        "net_g": net_g,
        "style_vectors": style_vectors,
        "mean_style": mean_style,
        "sid": sid,
    }


# 엔진 초기화 (프로세스 시작 시 실행)
ENGINE = load_engine()

# 외부 컨트롤러에게 "모델 로드 완료" 신호
print("__SBV2_READY__", flush=True)


# 메인 루프
# stdin → 추론 → wav 생성 → stdout
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        # 1) 입력 파싱
        # payload 예:
        # {
        #   "text": "なんとなく、気分が乗らない朝かな。",
        #   "style": 3,
        #   "style_weight": 0.8
        # }
        payload = json.loads(line)

        text = payload["text"]
        style_index = payload.get("style", 0)
        style_weight = payload.get("style_weight", 1.0)

        # 2) 스타일 벡터 계산
        style_vectors = ENGINE["style_vectors"]
        mean_style = ENGINE["mean_style"]

        target_style = style_vectors[style_index]

        # 스타일 보간 공식:
        # style = mean + (target - mean) * weight
        style_vec = mean_style + (target_style - mean_style) * style_weight

        # 3) TTS 추론
        with torch.no_grad():  # 추론이므로 gradient 불필요
            audio = infer(
                text=text,
                style_vec=style_vec,
                sdp_ratio=0.2,      # duration predictor stochasticity
                noise_scale=0.6,    # 음색 랜덤성
                noise_scale_w=0.8,  # prosody 랜덤성
                length_scale=1.0,   # 말 속도 (↓ 느림, ↑ 빠름)
                sid=ENGINE["sid"],
                language=Languages.JP,
                hps=ENGINE["hps"],
                net_g=ENGINE["net_g"],
                device=DEVICE,
            )

        # 4) wav 파일 저장
        out_path = OUT_DIR / f"tts_{abs(hash(text))}.wav"
        sf.write(
            out_path,
            audio,
            ENGINE["hps"].data.sampling_rate,
        )

        # 5) 결과 반환
        print(
            json.dumps({"wav_path": str(out_path)}),
            flush=True
        )

    except Exception as e:
        # 어떤 오류든 JSON 형태로 반환
        print(
            json.dumps({"error": str(e)}),
            flush=True
        )
