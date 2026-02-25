import soundfile as sf
import numpy as np
import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.infer import get_net_g, infer

CONFIG = "model_assets/tts/mai/config.json"
MODEL  = "model_assets/tts/mai/mai_e281_s263000.safetensors"
STYLE  = "model_assets/tts/mai/style_vectors.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load config / model
# ----------------------------
hps = HyperParameters.load_from_json(CONFIG)

net_g = get_net_g(
    model_path=MODEL,
    version=hps.version,
    device=device,
    hps=hps,
)

# ----------------------------
# Load style vectors
# ----------------------------
style_vectors = np.load(STYLE)

mean_style = style_vectors[0]      # Neutral
target_style = style_vectors[0]    # 예: Angry / Happy / Sad 등

style_weight = 4.0  # 핵심 파라미터 (0.5 ~ 2.0 실험 권장)

style_vec = mean_style + (target_style - mean_style) * style_weight

# ----------------------------
# Speaker
# ----------------------------
sid = hps.data.spk2id["mai"]

# ----------------------------
# Inference
# ----------------------------
with torch.no_grad():
    audio = infer(
        text="今日はちょっと寒いな。今日、うちに来る？",
        style_vec=style_vec,
        sdp_ratio=0.2,
        noise_scale=0.4,
        noise_scale_w=0.6,
        length_scale=1.0,
        sid=sid,
        language=Languages.JP,
        hps=hps,
        net_g=net_g,
        device=device,
    )

sf.write("out_mai.wav", audio, hps.data.sampling_rate)
