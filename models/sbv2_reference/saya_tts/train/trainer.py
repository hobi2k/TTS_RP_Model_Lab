"""
saya_tts/train/trainer.py

Style-Bert-VITS2 파인튜닝용 Trainer

이 Trainer의 책임:
- DataLoader가 만든 batch(dict)를 받아
- 사전학습된 net_g를 forward
- loss를 계산
- backward / optimizer step 수행
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SayaTTSTrainer:
    """
    Saya TTS Trainer

    이 클래스는 '훈련 1 step'의 논리를 캡슐화한다.
    (epoch / logging / checkpoint 관리는 바깥에서 한다)

    net_g:
        Style-Bert-VITS2의 SynthesizerTrn or SynthesizerTrnJPExtra
        - get_net_g(...)로 로드된 사전학습 모델
    """

    def __init__(
        self,
        *,
        net_g: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = False,
    ):
        self.net_g = net_g
        self.optimizer = optimizer
        self.device = device
        self.use_amp = bool(use_amp)

        # AMP 사용 시 scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 학습 모드 고정
        self.net_g.train()

    # 1) 단일 training step
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        하나의 batch로 forward + loss + backward + step을 수행한다.

        Args:
            batch:
                collate_saya_tts(...)가 반환한 dict

        Returns:
            losses(dict):
                로그 기록용 scalar loss 값들
        """

        # 1. batch를 device로 이동
        phoneme_ids = batch["phoneme_ids"].to(self.device)     # [B, T_text]
        tone_ids = batch["tone_ids"].to(self.device)           # [B, T_text]
        language_ids = batch["language_ids"].to(self.device)   # [B, T_text]
        x_mask = batch["x_mask"].to(self.device)               # [B, 1, T_text]
        bert_feats = batch["bert_feats"].to(self.device)       # [B, D_bert, T_text]

        mel = batch["mel"].to(self.device)                     # [B, n_mels, T_mel]

        style_id = batch["style_id"].to(self.device)           # [B]
        speaker_id = batch["speaker_id"].to(self.device)       # [B]

        # 2. optimizer 초기화
        self.optimizer.zero_grad(set_to_none=True)

        # 3. forward
        # Style-Bert-VITS2의 net_g.forward(...)는
        # 내부적으로 다음을 수행한다:
        #
        # - enc_p(...)
        # - duration predictor (dp / sdp)
        # - posterior encoder (enc_q)
        # - flow
        # - decoder
        #
        # 그리고 loss 계산에 필요한 중간 텐서들을 반환한다.
        #
        # 주의:
        # forward 시그니처는 SBV2 버전에 따라 다를 수 있으므로
        # 여기서는 "개념적으로 필요한 입력"을 모두 넘긴다는 점이 중요하다.

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.net_g(
                phoneme_ids=phoneme_ids,
                tone_ids=tone_ids,
                language_ids=language_ids,
                bert_feats=bert_feats,
                mel=mel,
                style_id=style_id,
                speaker_id=speaker_id,
                x_mask=x_mask,
            )

            # outputs는 보통 dict 또는 tuple 형태
            # (버전/포크마다 다름)
            #
            # 여기서는 아래처럼 가정한다:
            # outputs = {
            #   "loss_total": scalar,
            #   "loss_mel": scalar,
            #   "loss_kl": scalar,
            #   "loss_dur": scalar,
            #   ...
            # }

            loss_total = outputs["loss_total"]

        # 4. backward
        if self.use_amp:
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_total.backward()
            self.optimizer.step()

        # 5. loss 정리 (로그용)
        losses = {
            "loss_total": float(loss_total.detach().cpu()),
        }

        # 선택적으로 세부 loss 기록
        for k, v in outputs.items():
            if k.startswith("loss_") and k != "loss_total":
                losses[k] = float(v.detach().cpu())

        return losses

    # 2) validation step (선택)
    @torch.no_grad()
    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        validation step.
        구조는 train_step과 동일하지만 backward를 하지 않는다.
        """
        self.net_g.eval()

        phoneme_ids = batch["phoneme_ids"].to(self.device)
        tone_ids = batch["tone_ids"].to(self.device)
        language_ids = batch["language_ids"].to(self.device)
        x_mask = batch["x_mask"].to(self.device)
        bert_feats = batch["bert_feats"].to(self.device)
        mel = batch["mel"].to(self.device)
        style_id = batch["style_id"].to(self.device)
        speaker_id = batch["speaker_id"].to(self.device)

        outputs = self.net_g(
            phoneme_ids=phoneme_ids,
            tone_ids=tone_ids,
            language_ids=language_ids,
            bert_feats=bert_feats,
            mel=mel,
            style_id=style_id,
            speaker_id=speaker_id,
            x_mask=x_mask,
        )

        losses = {
            "loss_total": float(outputs["loss_total"].detach().cpu())
        }
        for k, v in outputs.items():
            if k.startswith("loss_") and k != "loss_total":
                losses[k] = float(v.detach().cpu())

        self.net_g.train()
        return losses
