"""
slice.py
--------
오디오 파일을 VAD(Voice Activity Detection)를 사용해 발화 단위로 분할(slice)하는 유틸리티입니다.
주요 기능:
- Silero VAD 모델을 사용하여 음성 구간을 검출
- 검출된 구간에 margin(앞뒤 여유)을 붙여 segment로 잘라 .wav 파일로 저장
- 멀티스레드 워커 풀로 빠르게 처리하고 에러를 집계

사용 예시:
python slice.py --model_name <MODEL> --input_dir <INPUT_DIR>

문법 팁:
- Queue와 Thread를 사용해 생산자-소비자 패턴으로 파일 처리 워커를 구성합니다.
- tqdm과 SAFE_STDOUT를 함께 사용하면 멀티스레드 환경에서도 깔끔한 진행바 출력이 가능합니다.
"""

import argparse
import shutil
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional

import soundfile as sf
import torch
from tqdm import tqdm

from config import get_path_config
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


def is_audio_file(file: Path) -> bool:
    """파일의 확장자로 오디오 파일 여부를 판정합니다.

    Args:
        file (Path): 검사할 파일 경로

    Returns:
        bool: 지원하는 오디오 확장자이면 True

    Note:
        - 확장자는 소문자로 변환되어 비교됩니다.
        - 필요시 여기에 확장자를 추가하면 그 포맷도 처리 대상이 됩니다.
    """
    supported_extensions = [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"]
    return file.suffix.lower() in supported_extensions


def get_stamps(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    min_silence_dur_ms: int = 700,
    min_sec: float = 2,
    max_sec: float = 12,
):
    """주어진 오디오 파일에서 VAD를 사용해 음성 타임스탬프 목록을 반환합니다.

    Args:
        vad_model: Silero VAD 모델 인스턴스
        utils: Silero VAD에서 제공하는 유틸 튜플 (get_speech_timestamps, ..., read_audio, ...)
        audio_file (Path): 입력 오디오 파일 경로
        min_silence_dur_ms (int): 무음으로 간주할 최소 길이(ms)
        min_sec (float): 최소 발화 길이(초) - 이보다 짧으면 무시
        max_sec (float): 최대 발화 길이(초) - 이보다 길면 무시

    Returns:
        list[dict]: VAD가 반환하는 timestamp dict들의 리스트 (각 dict는 'start'/'end' 등을 포함)

    Notes:
        - sampling_rate는 16kHz로 고정되어 있습니다(모델/유틸 구현에 의존).
        - 반환되는 timestamps의 단위 및 스케일은 get_speech_timestamps 구현에 의존하므로, downstream에서
          (예: split_wav) 적절히 변환하여 사용해야 합니다.
    """

    (get_speech_timestamps, _, read_audio, *_) = utils
    # 이 스크립트는 16kHz를 기준으로 동작하도록 설계됨
    sampling_rate = 16000  # 16kHzか8kHzのみ対応

    min_ms = int(min_sec * 1000)

    # read_audio는 파일을 읽어 waveform numpy array를 반환합니다.
    wav = read_audio(str(audio_file), sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate,
        min_silence_duration_ms=min_silence_dur_ms,
        min_speech_duration_ms=min_ms,
        max_speech_duration_s=max_sec,
    )

    return speech_timestamps


def split_wav(
    vad_model: Any,
    utils: Any,
    audio_file: Path,
    target_dir: Path,
    min_sec: float = 2,
    max_sec: float = 12,
    min_silence_dur_ms: int = 700,
    time_suffix: bool = False,
) -> tuple[float, int]:
    """오디오 파일을 VAD 타임스탬프에 따라 분할하고 잘라낸 세그먼트를 파일로 저장합니다.

    Args:
        vad_model: Silero VAD 모델 인스턴스
        utils: Silero 유틸( get_speech_timestamps, read_audio 등 )
        audio_file (Path): 입력 오디오 파일
        target_dir (Path): 세그먼트를 저장할 폴더
        min_sec (float): 최소 발화 길이(초)
        max_sec (float): 최대 발화 길이(초)
        min_silence_dur_ms (int): 분할 기준이 되는 무음 길이(ms)
        time_suffix (bool): True이면 파일명을 start_ms-end_ms 형태로 저장

    Returns:
        (total_time_sec, count): 잘려진 세그먼트의 누적 재생 시간(초)과 파일 개수

    Notes:
        - ts['start'] 및 ts['end']의 단위는 get_speech_timestamps의 출력 형식에 의존합니다.
          이 코드에서는 기존 구현대로 `ts['start'] / 16` 등으로 스케일링을 수행하여 ms 단위로 변환합니다.
          (해당 상수 16은 모델/유틸의 내부 표현에 의존하는 값이므로 필요시 조정이 필요합니다.)
        - margin은 각 세그먼트 앞뒤에 추가하는 여유(ms)로, 발화가 잘리는 것을 방지합니다.
    """
    margin: int = 200  # ミリ秒単位で、音声の前後に余裕を持たせる
    speech_timestamps = get_stamps(
        vad_model=vad_model,
        utils=utils,
        audio_file=audio_file,
        min_silence_dur_ms=min_silence_dur_ms,
        min_sec=min_sec,
        max_sec=max_sec,
    )

    # 전체 오디오 파일을 읽어 numpy 배열과 샘플레이트를 획득
    data, sr = sf.read(audio_file)

    total_ms = len(data) / sr * 1000

    file_name = audio_file.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    total_time_ms: float = 0
    count = 0

    # 타임스탬프 순회: start/end를 ms로 변환하여 세그먼트를 추출
    for i, ts in enumerate(speech_timestamps):
        # 주의: ts['start']/16 등은 기존 코드의 스케일 보정을 따릅니다.
        # 이 스케일이 왜 필요한지는 get_speech_timestamps의 반환 단위를 확인してください.
        start_ms = max(ts["start"] / 16 - margin, 0)
        end_ms = min(ts["end"] / 16 + margin, total_ms)

        # ms -> sample 인덱스로 변환
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        segment = data[start_sample:end_sample]

        if time_suffix:
            file = f"{file_name}-{int(start_ms)}-{int(end_ms)}.wav"
        else:
            file = f"{file_name}-{i}.wav"
        # 파일 저장
        sf.write(str(target_dir / file), segment, sr)
        total_time_ms += end_ms - start_ms
        count += 1

    # 총 재생 시간(초), 분할된 파일 개수
    return total_time_ms / 1000, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min_sec", "-m", type=float, default=2, help="Minimum seconds of a slice"
    )
    parser.add_argument(
        "--max_sec", "-M", type=float, default=12, help="Maximum seconds of a slice"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default="inputs",
        help="Directory of input wav files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The result will be in Data/{model_name}/raw/ (if Data is dataset_root in configs/paths.yml)",
    )
    parser.add_argument(
        "--min_silence_dur_ms",
        "-s",
        type=int,
        default=700,
        help="Silence above this duration (ms) is considered as a split point.",
    )
    parser.add_argument(
        "--time_suffix",
        "-t",
        action="store_true",
        help="Make the filename end with -start_ms-end_ms when saving wav.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=3,
        help="Number of processes to use. Default 3 seems to be the best.",
    )
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    model_name = str(args.model_name)
    input_dir = Path(args.input_dir)
    output_dir = dataset_root / model_name / "raw"
    min_sec: float = args.min_sec
    max_sec: float = args.max_sec
    min_silence_dur_ms: int = args.min_silence_dur_ms
    time_suffix: bool = args.time_suffix
    num_processes: int = args.num_processes

    audio_files = [file for file in input_dir.rglob("*") if is_audio_file(file)]

    logger.info(f"Found {len(audio_files)} audio files.")
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists, deleting...")
        shutil.rmtree(output_dir)

    # モデルをダウンロードしておく
    _ = torch.hub.load(
        repo_or_dir="litagin02/silero-vad",
        model="silero_vad",
        onnx=True,
        trust_repo=True,
    )

    # Silero VADのモデルは、同じインスタンスで並列処理するとおかしくなるらしい
    # ワーカーごとにモデルをロードするようにするため、Queueを使って処理する
    def process_queue(
        q: Queue[Optional[Path]],
        result_queue: Queue[tuple[float, int]],
        error_queue: Queue[tuple[Path, Exception]],
    ):
        """워크러 함수를 정의합니다. 각 워커는 자체 Silero VAD 모델을 로드한 뒤 큐에서 파일을 받아 처리합니다.

        Args:
            q: 입력 파일 경로를 받는 Queue (None은 종료 신호)
            result_queue: (time_sec, count)를 받는 Queue (처리 시간 및 파일 수 통계)
            error_queue: (file, Exception) 쌍을 담는 Queue (에러 집계)

        이유/설명:
            - Silero VAD는 동일한 모델 인스턴스를 여러 스레드에서 공유하면 문제가 발생할 수 있어
              워커마다 모델을 로드하도록 설계되어 있습니다.
            - q.get()으로 파일을 받아, None을 만나면 종료 시그널로 간주합니다.
        """
        # 워커별로 모델을 로드 (onnx=True로 로드하여 ONNX 런타임 사용 가능)
        vad_model, utils = torch.hub.load(
            repo_or_dir="litagin02/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True,
        )
        while True:
            file = q.get()
            if file is None:  # 종료 신호 확인
                q.task_done()
                break
            try:
                rel_path = file.relative_to(input_dir)
                # split_wav를 호출하여 파일을 분할하고 결과(총 시간, 파일 개수)를 받음
                time_sec, count = split_wav(
                    vad_model=vad_model,
                    utils=utils,
                    audio_file=file,
                    target_dir=output_dir / rel_path.parent,
                    min_sec=min_sec,
                    max_sec=max_sec,
                    min_silence_dur_ms=min_silence_dur_ms,
                    time_suffix=time_suffix,
                )
                # 통계 저장
                result_queue.put((time_sec, count))
            except Exception as e:
                # 개별 파일 처리 중 오류 발생 시 에러 큐에 추가하고 (0,0)으로 표시
                logger.error(f"Error processing {file}: {e}")
                error_queue.put((file, e))
                result_queue.put((0, 0))
            finally:
                q.task_done()

    # 세 개의 큐 초기화
    q: Queue[Optional[Path]] = Queue()
    result_queue: Queue[tuple[float, int]] = Queue()
    error_queue: Queue[tuple[Path, Exception]] = Queue()

    # 파일 수가 워커 수보다 적으면 워커 수를 파일 수로 제한
    num_processes = min(num_processes, len(audio_files))

    # 워커 스레드 생성 및 시작
    threads = [
        Thread(target=process_queue, args=(q, result_queue, error_queue))
        for _ in range(num_processes)
    ]
    for t in threads:
        t.start()

    # 작업을 큐에 추가
    pbar = tqdm(total=len(audio_files), file=SAFE_STDOUT, dynamic_ncols=True)
    for file in audio_files:
        q.put(file)

    # result_queue를 모니터링하면서 처리 결과(시간, 개수)를 합산하고 프로그레스바를 업데이트
    total_sec = 0
    total_count = 0
    for _ in range(len(audio_files)):
        time, count = result_queue.get()
        total_sec += time
        total_count += count
        pbar.update(1)

    # 모든 처리 작업이 끝날 때까지 대기(q.join은 모든 task_done 호출을 기다림)
    q.join()

    # 워커에게 종료 신호(None)를 보냄
    for _ in range(num_processes):
        q.put(None)

    # 모든 스레드가 종료할 때까지 조인
    for t in threads:
        t.join()

    pbar.close()

    # 처리 중 에러가 수집되었으면 상세 정보를 모아서 예외 발생
    if not error_queue.empty():
        error_str = "Error slicing some files:"
        while not error_queue.empty():
            file, e = error_queue.get()
            error_str += f"\n{file}: {e}"
        raise RuntimeError(error_str)

    # 최종 요약 로그
    logger.info(
        f"Slice done! Total time: {total_sec / 60:.2f} min, {total_count} files."
    )