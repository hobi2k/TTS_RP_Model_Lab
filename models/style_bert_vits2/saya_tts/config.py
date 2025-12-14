from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SayaTTSConfig:
    """
    Inference 전용 설정.
    학습 관련 필드는 의도적으로 제거한다.
    """

    # checkpoint / config
    ckpt_path: Path
    hps_path: Path

    # BERT
    # Style-Bert-VITS2 원본에서 쓰는 모델을 그대로 고정
    # Style-Bert-VITS2는 "일본어 문맥 특징"을 위해 특정 모델을 전제로 한다.
    # 이 값을 임의로 바꾸면:
    # - hidden_size(출력 채널)가 달라져서 TextEncoder의 bert_proj Conv1d가 깨지거나
    # - tokenizer 특성이 달라져서 길이 정렬 로직이 달라질 수 있다.
    bert_model_name: str = (
        "ku-nlp/deberta-v2-large-japanese-char-wwm"
    )

    # 오디오 샘플링 레이트
    # 이 값은 '모델이 학습한 기준'이다.
    # 모델이 44.1kHz로 학습됐는데 48kHz로 저장하면:
    # - 재생 속도가 변하고
    # - 피치가 바뀌고
    # - 캐릭터 음색이 망가진 것처럼 들린다.
    sampling_rate: int = 44100

    # 런타임 디바이스 세팅
    device: str = "cuda"
