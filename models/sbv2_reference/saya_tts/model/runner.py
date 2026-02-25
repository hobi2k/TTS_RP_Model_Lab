"""
Inference Runner

목표:
  - Style-Bert-VITS2 "실제" 모델 에셋을 로드한다.
  - SynthesizerTrn(or JP-Extra)을 구성하고 체크포인트를 주입한다.
  - 원본 infer() 경로로 wav(np.ndarray float32)를 얻는다.

필수 파일(모델 공유 단위):
  model_assets/<model_name>/
    - config.json
    - *.safetensors (또는 .pth/.pt)
    - style_vectors.npy
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, infer

@dataclass(frozen=True)
class TTSInferenceResult:
    """
    audio: float32 1D numpy array (mono)
      - infer.py에서 최종적으로 output[0][0,0]을 numpy로 뽑아 반환
    sampling_rate: int
      - config.json -> HyperParameters.data.sampling_rate (기본 44100)
    """
    audio: np.ndarray
    sampling_rate: int

class StyleBertVITS2Runner:
    """
    Style-Bert-VITS2 실전 inference 러너.

    핵심 포인트:
      1. model_dir에서 config.json 로드 -> HyperParameters 생성
      2. model 파일(.safetensors/.pth/.pt) 로드해서 net_g 구성 + weight 주입
      3. style_vectors.npy에서 style_id(혹은 style_name)로 style_vec 선택
      4. infer() 호출해서 audio 획득
    """
    def __init__(
            self,
            model_dir: Union[str, Path],
            device: str = "cuda",
            model_filename: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device


        # 1. config.json -> HyperParameters
        # HyperParameters.load_from_json이 json을 pydantic으로 검증/파싱해준다. :contentReference[oaicite:8]{index=8}
        self.config_path = self.model_dir / "config.json"
        if not self.config_path.exists():
            raise FileNotFoundError(f"config.json not found: {self.config_path}")

        self.hps: HyperParameters = HyperParameters.load_from_json(self.config_path)

        # 2. style_vectors.npy 로드
        self.style_path = self.model_dir / "style_vectors.npy"
        if not self.style_path.exists():
            raise FileNotFoundError(f"style_vectors.npy not found: {self.style_path}")

        # shape 예시(일반적으로):
        # (num_styles, style_dim)
        # 실제 사용 시 infer()는 style_vec를 np.ndarray로 받는다.
        self.style_vectors: np.ndarray = np.load(self.style_path)

        # 3. 모델 파일 선택
        # README는 *.safetensors를 기본으로 안내하지만, infer.py는 .pth/.pt도 지원한다.
        self.model_path = self._resolve_model_path(model_filename)

        # 4. net_g 구성 + weight 로드
        # infer.py의 get_net_g는 version이 "JP-Extra"로 끝나는지에 따라
        # SynthesizerTrnJPExtra vs SynthesizerTrn을 선택하고,
        # 확장자별로 체크포인트 로더를 호출한다.
        self.net_g = get_net_g(
            model_path=str(self.model_path),
            version=self.hps.version,
            device=self.device,
            hps=self.hps,
        )

    def _resolve_model_path(self, model_filename: Optional[str]) -> Path:
        """
        model_filename이 주어지면 그것을 사용.
        없으면 model_dir 내에서 우선순위로 검색:
          1) .safetensors
          2) .pth
          3) .pt
        """
        if model_filename is not None:
            p = self.model_dir / model_filename
            if not p.exists():
                raise FileNotFoundError(f"model file not found: {p}")
            return p

        # 자동 탐색
        for ext in (".safetensors", ".pth", ".pt"):
            cands = sorted(self.model_dir.glob(f"*{ext}"))
            if cands:
                return cands[0]

        raise FileNotFoundError(
            f"No model checkpoint found in {self.model_dir} "
            f"(expected .safetensors/.pth/.pt)"
        )
    
    def get_style_vec_by_id(self, style_id: int) -> np.ndarray:
        """
        style_vectors.npy에서 style_id를 선택.
        - style2id는 config.json 내부에 있을 수 있지만,
          여기서는 'id를 주면 곧장 벡터'로 가는 경로만 고정한다.
        """
        if not (0 <= style_id < self.style_vectors.shape[0]):
            raise ValueError(
                f"style_id out of range: {style_id}, available: 0..{self.style_vectors.shape[0]-1}"
            )
        return self.style_vectors[style_id]

    def get_style_vec_by_name(self, style_name: str) -> np.ndarray:
        """
        style 이름 -> id -> 벡터.
        HyperParameters.data.style2id에 기본 {"Neutral": 0} 등이 존재.
        """
        style2id = self.hps.data.style2id
        if style_name not in style2id:
            raise KeyError(f"Unknown style '{style_name}'. Available: {list(style2id.keys())}")
        return self.get_style_vec_by_id(style2id[style_name])

    def synthesize(
        self,
        text: str,
        *,
        language: Languages = Languages.JP,
        style_name: str = "Neutral",
        speaker_id: int = 0,
        # 아래 3개는 infer.py 시그니처의 핵심 컨트롤 파라미터들
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        # 문장 앞/뒤 토큰 스킵
        skip_start: bool = False,
        skip_end: bool = False,
        # assist text
        assist_text: Optional[str] = None,
        assist_text_weight: float = 0.7,
    ) -> TTSInferenceResult:
        """
        infer.py의 infer()를 그대로 호출해서 audio를 얻는다.

        infer()가 내부에서 하는 일(중요):
          - clean_text()로 phone/tone/word2ph 생성
          - extract_bert_feature()로 (1024, len(phone)) 형태 BERT feature 생성
          - 언어별로 bert/ja_bert/en_bert를 구성한 뒤
          - net_g.infer(...)를 JP-Extra/Normal 모델에 맞게 호출
        """
        style_vec = self.get_style_vec_by_name(style_name)

        audio = infer(
            text=text,
            style_vec=style_vec,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker_id,
            language=language,
            hps=self.hps,
            net_g=self.net_g,
            device=self.device,
            skip_start=skip_start,
            skip_end=skip_end,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
        )

        # sampling_rate는 config.json(hps.data.sampling_rate)을 신뢰하는 게 정석 (기본 44100)
        return TTSInferenceResult(audio=np.asarray(audio, dtype=np.float32), sampling_rate=self.hps.data.sampling_rate)
