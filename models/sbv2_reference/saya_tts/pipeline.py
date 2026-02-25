from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g

from saya_tts.engine.sbv2_frontend import get_text_like_sbv2
from saya_tts.engine.infer_after_enc_p import EncPOut, infer_after_enc_p
from saya_tts.engine.infer_after_enc_p import (
    EncPOutJPExtra,
    InferOutJPExtra,
    infer_after_enc_p_jp_extra,
)


@dataclass
class SayaTTSConfig3B4:
    model_path: str # 모델 파일(.safetensors/.pth)
    device: str # "cuda" 또는 "cpu"
    hps: HyperParameters # config.json 로드 결과


class SayaTTSPipeline3B4:
    """
    PART 3-B-4:
    - normal 모델 + JP-Extra 모델을 모두 지원하는 실전 파이프라인
    - 공통: frontend(get_text 동등) -> enc_p 분리 호출 -> enc_p 이후 수식 재현 -> wav
    - 차이: JP-Extra는 bert 입력 1개(ja_bert)만 사용 :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, cfg: SayaTTSConfig3B4):
        self.cfg = cfg
        self.device = cfg.device
        self.hps = cfg.hps
        self.is_jp_extra = self.hps.version.endswith("JP-Extra")

        self.net_g = get_net_g(
            model_path=cfg.model_path,
            version=self.hps.version,
            device=self.device,
            hps=self.hps,
        )

    @torch.inference_mode()
    def debug_enc_p(
        self,
        text: str,
        *,
        style_vec: np.ndarray,
        sid: int,
        language: Languages = Languages.JP,
    ):
        """
        enc_p(TextEncoder) 출력만 뽑아서 shape/NaN 검증.
        - 반환 타입은 normal/JP-Extra에 따라 다르다.
        """
        fe = get_text_like_sbv2(
            text=text,
            language=language,
            device=self.device,
            use_jp_extra=self.is_jp_extra,
        )

        # 배치 차원 추가 (원본 infer.py와 동일한 패턴) :contentReference[oaicite:6]{index=6}
        x_tst = fe.phones.to(self.device).unsqueeze(0) # x(음소 id)
        tone_t = fe.tones.to(self.device).unsqueeze(0) # tone(억양 id)
        lang_t = fe.lang_ids.to(self.device).unsqueeze(0) # language(언어 id)
        x_len = torch.LongTensor([fe.phones.size(0)]).to(self.device)  # x_lengths(유효 길이)
        sid_t = torch.LongTensor([sid]).to(self.device)  # sid(화자 id)
        style_t = torch.from_numpy(style_vec).to(self.device).unsqueeze(0)  # style_vec(스타일 벡터)

        # g(화자 조건 임베딩)
        if self.net_g.n_speakers > 0:
            g = self.net_g.emb_g(sid_t).unsqueeze(-1)  # [B, gin, 1]
        else:
            raise ValueError("n_speakers==0")

        if self.is_jp_extra:
            # JP-Extra는 ja_bert만 infer로 넘긴다.
            bert_jp = fe.ja_bert.to(self.device).unsqueeze(0)   # bert(일본어 특징량)

            # JP-Extra enc_p 시그니처: enc_p(x, x_lengths, tone, language, bert, style_vec, g=g)
            x_enc, m_p, logs_p, x_mask = self.net_g.enc_p(
                x_tst, x_len, tone_t, lang_t, bert_jp, style_t, g=g
            )

            assert torch.isfinite(x_enc).all(), "JP-Extra enc_p 출력 x에서 inf/nan"
            assert torch.isfinite(m_p).all(), "JP-Extra enc_p 출력 m_p에서 inf/nan"
            assert torch.isfinite(logs_p).all(), "JP-Extra enc_p 출력 logs_p에서 inf/nan"

            return EncPOutJPExtra(x=x_enc, m_p=m_p, logs_p=logs_p, x_mask=x_mask, g=g)

        else:
            # normal은 bert/ja_bert/en_bert 3개를 모두 받는다. 
            bert = fe.bert.to(self.device).unsqueeze(0)
            ja_bert = fe.ja_bert.to(self.device).unsqueeze(0)
            en_bert = fe.en_bert.to(self.device).unsqueeze(0)

            # normal enc_p 시그니처는 모델 구현에 따라 인자가 더 많을 수 있으니,
            # 너의 net_g.enc_p 정의와 정확히 맞춰라. (현재 흐름은 “우리가 PART 3-B-3에서 맞춘 형태”)
            x_enc, m_p, logs_p, x_mask = self.net_g.enc_p(
                x_tst, x_len, tone_t, lang_t, bert, ja_bert, en_bert, style_t, sid_t, g=g
            )

            assert torch.isfinite(x_enc).all(), "normal enc_p 출력 x에서 inf/nan"
            assert torch.isfinite(m_p).all(), "normal enc_p 출력 m_p에서 inf/nan"
            assert torch.isfinite(logs_p).all(), "normal enc_p 출력 logs_p에서 inf/nan"

            return EncPOut(x=x_enc, m_p=m_p, logs_p=logs_p, x_mask=x_mask, g=g)

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        *,
        style_vec: np.ndarray,
        sid: int,
        language: Languages = Languages.JP,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        max_len: Optional[int] = None,
    ) -> tuple[int, np.ndarray]:
        """
        최종 WAV 생성.
        - normal/JP-Extra를 자동 분기한다.
        """
        enc = self.debug_enc_p(text, style_vec=style_vec, sid=sid, language=language)

        if self.is_jp_extra:
            out = infer_after_enc_p_jp_extra(
                self.net_g,
                enc,  # EncPOutJPExtra
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                max_len=max_len,
            )
            audio = out.o[0, 0].detach().cpu().float().numpy()

        else:
            out = infer_after_enc_p(
                self.net_g,
                enc,  # EncPOut
                sid=torch.LongTensor([sid]).to(self.device),
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                max_len=max_len,
            )
            audio = out.o[0, 0].detach().cpu().float().numpy()

        sr = int(self.hps.data.sampling_rate)  # config 기준이 정답
        return sr, audio
