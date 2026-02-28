import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.serialization import add_safe_globals
from numpy.typing import NDArray

# pyannote.audio still references the legacy torchaudio.AudioMetaData symbol at import
# time, but recent nightly torchaudio builds no longer export it.
if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class AudioMetaData:  # pragma: no cover - compatibility shim
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda backend: None  # type: ignore[attr-defined]

if not hasattr(torchaudio, "info"):
    def _compat_info(path: str) -> AudioMetaData:
        info = sf.info(path)
        subtype = info.subtype or ""
        bits = int("".join(ch for ch in subtype if ch.isdigit()) or 0)
        return AudioMetaData(
            sample_rate=info.samplerate,
            num_frames=info.frames,
            num_channels=info.channels,
            bits_per_sample=bits,
            encoding=subtype or "UNKNOWN",
        )

    torchaudio.info = _compat_info  # type: ignore[attr-defined]

if not hasattr(torchaudio, "_original_load"):
    torchaudio._original_load = torchaudio.load  # type: ignore[attr-defined]


def _compat_load(path: str, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T.copy())
    return waveform, int(sample_rate)


torchaudio.load = _compat_load  # type: ignore[assignment]

from pyannote.audio import Inference, Model
from pyannote.audio.core.task import Problem, Resolution, Specifications
from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()

# PyTorch nightly defaults torch.load(weights_only=True), but pyannote's checkpoint
# includes TorchVersion in the serialized state.
add_safe_globals(
    [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
)

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path: str) -> NDArray[Any]:
    return inference(wav_path)  # type: ignore


def save_style_vector(wav_path: str):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    # 値にNaNが含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def process_line(line: str):
    wav_path = line.split("|")[0]
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        return line, "nan_error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.style_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.style_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path: str = args.config
    num_processes: int = args.num_processes

    hps = HyperParameters.load_from_json(config_path)

    device = config.style_gen_config.device

    training_lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        training_lines.extend(f.readlines())
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(process_line, training_lines),
                total=len(training_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )

    val_lines: list[str] = []
    with open(hps.data.validation_files, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(process_line, val_lines),
                total=len(val_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )

    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")
