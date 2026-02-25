"""
resample.py
-----------
오디오 파일을 지정된 샘플레이트로 리샘플링(resampling)하고(또는 변환) 필요시 음량(normalize)과 무음제거(trim)를 수행한 뒤
출력 디렉토리에 입력 디렉토리 구조를 보존하여 저장하는 편리한 유틸리티 스크립트입니다.

주요 기능:
- librosa로 파일 로드 및 리샘플링
- pyloudnorm을 이용한 방송 표준(BS.1770) 기반 라우드니스(normalize)
- soundfile을 이용한 wav로 저장
- 멀티스레드(ThreadPoolExecutor)를 이용해 병렬 처리
- tqdm으로 진행 표시

문법 팁:
- type hints (예: NDArray[Any])는 함수의 기대 입력/출력 타입을 문서화하고 정적 분석을 돕습니다.
- Path.rglob("*")는 하위 디렉터리를 포함해 모든 파일을 재귀적으로 찾을 때 유용합니다.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import librosa
import pyloudnorm as pyln
import soundfile
from numpy.typing import NDArray
from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# pyloudnorm의 block_size(초) 기본값. 너무 짧은 오디오의 경우 측정 실패할 수 있음.
DEFAULT_BLOCK_SIZE: float = 0.400  # seconds


class BlockSizeException(Exception):
    """라운드니스 측정에 필요한 최소 블록 길이보다 오디오가 짧을 때 발생시키는 예외입니다.

    pyloudnorm의 Meter.integrated_loudness()는 내부적으로 충분한 길이의 블록을 요구할 수 있으며,
    너무 짧은 입력(예: 수십 ms)에서는 ValueError가 발생할 수 있습니다. 이 경우 호출부에서 적절히 처리합니다.
    """
    pass


def normalize_audio(data: NDArray[Any], sr: int) -> NDArray[Any]:
    """입력 오디오의 라우드니스를 측정하고 목표 라우드니스(-23 LUFS)로 정규화합니다.

    Args:
        data (NDArray[Any]): 오디오 신호(1D numpy array, float)
        sr (int): 샘플링 레이트

    Returns:
        NDArray[Any]: 정규화된 오디오 신호

    Raises:
        BlockSizeException: 오디오가 너무 짧아 라우드니스 측정에 실패한 경우

    설명:
        - pyln.Meter(...).integrated_loudness()는 BS.1770 표준에 따른 라우드니스 측정입니다.
        - 측정값을 얻은 후 pyln.normalize.loudness()로 목표 라우드니스(-23.0 LUFS)로 맞춥니다.
    """
    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)  # create BS.1770 meter
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as e:
        # 측정에 실패하면 더 상위 레벨에서 처리하도록 특정 예외로 래핑
        raise BlockSizeException(e)

    # -23.0 LUFS로 정규화 (방송 표준 예시)
    data = pyln.normalize.loudness(data, loudness, -23.0)
    return data


def resample(
    file: Path,
    input_dir: Path,
    output_dir: Path,
    target_sr: int,
    normalize: bool,
    trim: bool,
) -> None:
    """단일 파일을 읽어 리샘플링/정규화/트림을 적용하고 출력 디렉토리에 저장합니다.

    동작 순서:
    1. librosa.load(file, sr=target_sr)를 사용해 파일을 로드(리샘플링 포함).
    2. normalize=True이면 normalize_audio() 호출하여 LUFS 기준 정규화.
       - 만약 오디오가 너무 짧아 측정 불가하면 해당 파일은 정규화를 건너뜁니다.
    3. trim=True이면 librosa.effects.trim을 사용해 앞/뒤 무음 부분을 제거.
    4. output_dir 내에서 input_dir로부터의 상대경로를 유지하여 .wav 확장자로 저장.

    Args:
        file (Path): 처리할 파일 경로
        input_dir (Path): 원본 입력 디렉토리 루트 (출력 시 상대경로 계산에 사용)
        output_dir (Path): 결과를 저장할 출력 디렉토리 루트
        target_sr (int): 목표 샘플레이트
        normalize (bool): 라우드니스 정규화 여부
        trim (bool): 앞/뒤 무음 제거 여부

    Notes:
        - librosa.load는 mp3, ogg, flac 등 다양한 포맷을 읽을 수 있으며, `sr`을 지정하면 로드 시 리샘플링을 수행합니다.
        - 출력은 항상 .wav로 저장됩니다(원본이 .mp3여도 .wav로 저장됨).
        - 예외는 로깅 후 무시하여 전체 배치 작업이 중단되지 않게 설계되어 있습니다.
    """
    try:
        # librosaが読めるファイルかチェック
        # wav以外にもmp3やoggやflacなども読める
        wav: NDArray[Any]
        sr: int
        # sr=target_sr로 지정하면 librosa가 로드 시 리샘플링을 수행한다
        wav, sr = librosa.load(file, sr=target_sr)

        # 라우드니스 정규화 (LUFS)
        if normalize:
            try:
                wav = normalize_audio(wav, sr)
            except BlockSizeException:
                # 오디오가 너무 짧아 라우드니스 측정/정규화가 불가한 경우
                # 해당 파일은 정규화를 건너뛰고 계속 진행
                print("")
                logger.info(
                    f"Skip normalize due to less than {DEFAULT_BLOCK_SIZE} second audio: {file}"
                )

        # 앞뒤 무음 제거 (trim) - top_db는 무음으로 판단할 기준 dB 값
        if trim:
            wav, _ = librosa.effects.trim(wav, top_db=30)

        # input_dir로부터 상대경로를 유지하여 output에 저장
        relative_path = file.relative_to(input_dir)
        # ここで拡張子が.wav以外でも.wavに置き換えられる
        output_path = output_dir / relative_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # soundfile.write는 numpy array와 샘플레이트를 받아 파일로 저장
        soundfile.write(output_path, wav, sr)
    except Exception as e:
        # 파일을 읽거나 처리하는 중 문제가 있으면 로그를 남기고 스킵
        logger.warning(f"Cannot load file, so skipping: {file}, {e}")


if __name__ == "__main__":
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="cpu_processes",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        default=False,
        help="trim silence (start and end only)",
    )
    args = parser.parse_args()

    # 프로세스 수 결정: CLI에서 0을 지정하면 자동으로 cpu_count() 기반으로 결정
    # (cpu_count()가 4보다 큰 경우엔 충분한 CPU를 남기기 위해 -2 하는 정책)
    if args.num_processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes: int = args.num_processes

    # 입력/출력 경로와 옵션 변수들
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logger.info(f"Resampling {input_dir} to {output_dir}")
    sr = int(args.sr)
    normalize: bool = args.normalize
    trim: bool = args.trim

    # 모든 파일을 재귀적으로 수집 (각 파일을 librosa로 읽어 처리 가능 여부를 체크함)
    # Path.rglob("*")는 하위 디렉터리의 파일도 포함해서 반환
    original_files = [f for f in input_dir.rglob("*") if f.is_file()]

    if len(original_files) == 0:
        logger.error(f"No files found in {input_dir}")
        raise ValueError(f"No files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ThreadPoolExecutor로 병렬 처리
    # - max_workers = processes: 동시 실행할 워커 스레드 수
    # - executor.submit(...) 으로 resample 작업을 제출
    # - as_completed(futures)를 tqdm과 같이 사용하면 완료된 태스크 순서로 진행 상태를 표시 가능
    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(resample, file, input_dir, output_dir, sr, normalize, trim)
            for file in original_files
        ]
        # tqdm에 as_completed를 넣어 '완료된 태스크 수' 기준으로 프로그레스 바를 업데이트
        for future in tqdm(
            as_completed(futures),
            total=len(original_files),
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        ):
            # future.result()를 호출하지 않고 단순히 완료 여부만 기다림 (resample 내부에서 로깅)
            pass

    logger.info("Resampling Done!")
