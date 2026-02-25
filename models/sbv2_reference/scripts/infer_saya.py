import soundfile as sf
from style_bert_vits2.constants import Languages
from saya_tts.pipeline_real import SayaTTSConfig3A, SayaTTSPipeline3A

cfg = SayaTTSConfig3A(
    model_dir="model_assets/saya_model",  # 네 saya 모델 에셋 폴더
    device="cuda",
)

tts = SayaTTSPipeline3A(cfg)

res = tts.synthesize(
    "こんにちは。私はサヤです。",
    style_name="Neutral",
    speaker_id=0,
    language=Languages.JP,
    sdp_ratio=0.2,
    noise_scale=0.6,
    noise_scale_w=0.8,
    length_scale=1.0,
)

sf.write("saya_out.wav", res.audio, res.sampling_rate)
print("saved:", "saya_out.wav", "sr:", res.sampling_rate)