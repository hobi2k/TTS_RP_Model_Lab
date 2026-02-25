"""
saya_tts/train/train.py

Style-Bert-VITS2 기반 "사야" 파인튜닝 실행 스크립트.

이 파일의 역할:
- Dataset / DataLoader 구성
- 사전학습 net_g 로드
- Optimizer / Trainer / Freezer 연결
- global_step 기준 학습 루프 실행
- 체크포인트 저장

중요:
- 이 파일은 '실제로 실행되는 코드'다.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from saya_tts.data.dataset import SayaTTSDataset
from saya_tts.data.collate import collate_saya_tts

from saya_tts.train.trainer import SayaTTSTrainer
from saya_tts.train.freezer import Freezer, FreezeScheduleConfig

# Style-Bert-VITS2 공식 API
from style_bert_vits2.models.infer import get_net_g
from style_bert_vits2.models.hyper_parameters import HyperParameters


# 0. 기본 설정
def main():
    # 경로 설정
    project_root = Path(__file__).resolve().parents[3]

    # 사전학습 모델 에셋 디렉토리
    model_dir = project_root / "model_assets" / "tts" / "style_bert_vits2_pretrained"

    # 학습 데이터 루트
    # 예: data/saya/
    data_root = project_root / "data" / "saya"

    # 체크포인트 저장 경로
    ckpt_dir = project_root / "checkpoints" / "saya"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터
    batch_size = 8
    num_workers = 4
    learning_rate = 2e-4
    max_steps = 100_000
    log_interval = 100
    save_interval = 5_000

    use_amp = True  # mixed precision

    # 1. Dataset / DataLoader
    """
    Dataset이 책임지는 것:
    - 텍스트
    - G2P -> phoneme / tone / language
    - mel 로드
    - style_id / speaker_id 제공

    collate_fn이 책임지는 것:
    - padding
    - x_mask 생성
    - batch 텐서화
    """

    train_dataset = SayaTTSDataset(
        root_dir=data_root,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_saya_tts,
    )

    # 2. Style-Bert-VITS2 사전학습 모델 로드
    """
    이 단계에서 중요한 사실:

    - net_g는 "우리가 구현한 enc_p"를 쓰지 않는다.
    - Style-Bert-VITS2의 '공식' SynthesizerTrn을 그대로 쓴다.
    - 우리가 enc_p를 구현한 이유는:
        * 구조 이해
        * 디버깅
        * freeze 전략 설계
    """

    # config.json -> HyperParameters
    hps = HyperParameters.load_from_json(model_dir / "config.json")

    # net_g 생성 + 사전학습 가중치 로드
    net_g = get_net_g(
        model_path=str(model_dir),
        version=hps.version,
        device=device,
        hps=hps,
    ).to(device)

    # 3. Optimizer
    """
    freeze/unfreeze 때문에:
    - requires_grad=True인 파라미터만 optimizer에 들어간다.
    """

    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, net_g.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=1e-2,
    )

    # 4. Trainer
    trainer = SayaTTSTrainer(
        net_g=net_g,
        optimizer=optimizer,
        device=device,
        use_amp=use_amp,
    )

    # 5. Freezer (freeze / unfreeze 스케줄)
    freeze_cfg = FreezeScheduleConfig(
        stage0_steps=2_000,
        stage1_steps=20_000,
        stage2_steps=50_000,
        enable_stage3=False,  # 처음엔 꺼두는 게 안전
    )

    freezer = Freezer(net_g=net_g, cfg=freeze_cfg)

    # 6. Training Loop
    global_step = 0

    print("Start Saya fine-tuning")

    while global_step < max_steps:
        for batch in train_loader:
            # (1) freeze / unfreeze 적용
            stage_name = freezer.apply_by_step(global_step)

            # (2) training step
            losses = trainer.train_step(batch)

            global_step += 1

            # (3) logging
            if global_step % log_interval == 0:
                loss_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in losses.items()
                )
                print(
                    f"[step {global_step:>7}] "
                    f"[{stage_name}] "
                    f"{loss_str}"
                )

            # (4) checkpoint
            if global_step % save_interval == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "net_g": net_g.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"checkpoint saved: {ckpt_path}")

            if global_step >= max_steps:
                break

    print("Training finished")


if __name__ == "__main__":
    main()
