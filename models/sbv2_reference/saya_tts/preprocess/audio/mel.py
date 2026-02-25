"""
wav -> mel 변환.

반드시:
- config.json과 동일한 파라미터를 써야 한다.
"""

import numpy as np
import librosa


def wav_to_mel(wav: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=80,
        fmin=0,
        fmax=sr // 2,
        power=1.0,
    )
    mel = np.log(np.clip(mel, 1e-5, None))
    return mel