import torch
import soundfile as sf


def save_wav(path: str, wav: torch.Tensor, sr: int):
    x = wav.detach().cpu().float()
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    x = torch.clamp(x, -1.0, 1.0).numpy()
    sf.write(path, x, sr)
