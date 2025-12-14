"""

실행 방법
python scripts/infer_saya.py \
  --text "……ねえ。本当に、私でいいの？" \
  --ckpt dummy.pth \
  --hps config.json
"""
import argparse
from pathlib import Path

from saya_tts.config import SayaTTSConfig
from saya_tts.pipeline import SayaTTSPipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hps", required=True)
    args = p.parse_args()

    cfg = SayaTTSConfig(
        ckpt_path=Path(args.ckpt),
        hps_path=Path(args.hps),
    )

    pipe = SayaTTSPipeline(cfg)

    res = pipe.extract_bert(args.text)
    print("BERT hidden states shape:", tuple(res.hidden_states.shape))


if __name__ == "__main__":
    main()
