import numpy as np
import soundfile as sf

from style_bert_vits2.constants import Languages
from style_bert_vits2.models.hyper_parameters import HyperParameters

from saya_tts.pipeline import SayaTTSConfig3B4, SayaTTSPipeline3B4


# 1. HyperParameters 로드 (보통 model_dir/config.json에서 읽어옴)
#    여기서는 예시로 “이미 로드된 hps를 넣는다” 가정.
hps = HyperParameters.load_from_json("model_assets/saya_model/config.json")  # 경로는 구조에 맞춰 수정

cfg = SayaTTSConfig3B4(
    model_path="model_assets/saya_model/saya.safetensors",  # 파일명에 맞춰 수정
    device="cuda",
    hps=hps,
)

tts = SayaTTSPipeline3B4(cfg)

# style_vec는 보통 style_vectors.npy에서 뽑는다.
# 예: Neutral이 0번이면 style_vec = style_vectors[0]
style_vec = np.load("model_assets/saya_model/style_vectors.npy")[0].astype(np.float32)

sr, audio = tts.synthesize(
    "こんにちは。私はサヤです。",
    style_vec=style_vec,
    sid=0,
    language=Languages.JP,
    sdp_ratio=0.2,
    noise_scale=0.6,
    noise_scale_w=0.8,
    length_scale=1.0,
)

sf.write("saya_out.wav", audio, sr)
print("saved:", "saya_out.wav", "sr:", sr)
