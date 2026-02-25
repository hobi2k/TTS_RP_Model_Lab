"""
preprocess_text.py
------------------
텍스트 전처리(요미 변환, phone/tones 추출 등)와 train/validation 분할을 수행하는 스크립트입니다.
- 텍스트 입력 파일(각 행: 'utt|spk|language|text')을 읽어, `clean_text()`로 정규화 및 음운/요미 정보를 생성합니다.
- 처리된 라인을 `cleaned_path`에 출력하고, 오디오 파일 존재 여부를 점검한 뒤 화자별로 train/validation을 분배합니다.
- 최종적으로 `train_path`, `val_path`에 파일 목록을 저장하고 `config_path`의 speaker 정보(spk2id, n_speakers)를 갱신합니다.

Python 문법 팁(이 파일과 관련된 것):
- Path.open(): pathlib.Path를 사용하면 파일 경로 관리를 안전하고 명료하게 할 수 있습니다.
- with 문: 리소스(파일 등)를 자동으로 닫아주는 컨텍스트 매니저입니다.
- typing.Optional[T]: 인자가 None일 수도 있음을 명시합니다.
- f-strings: f"{var}" 형태로 문자열 내에 값을 삽입할 때 사용합니다.

uv run --active preprocess_text.py \
  --transcription-path /mnt/d/my_tts_dataset/mai/esd.list \
  --correct_path \
  --use_jp_extra
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import Optional

from tqdm import tqdm

from config import get_config
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import clean_text
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# PyOpenJTalk 워커 초기화: 일본어 전처리에 사용하는 외부 워커를 현재 프로세스에서 시작합니다.
# 주의: 워커 초기화는 한번만 수행하면 되고, 이후 `clean_text` 등에서 내부적으로 워커를 활용합니다.
pyopenjtalk_worker.initialize_worker()

# 사용자 사전(dict_data/*)을 PyOpenJTalk에 적용합니다.
# update_dict()는 로컬 사용자 정의 사전을 읽어 pyopenjtalk의 사전에 병합합니다.
update_dict()


preprocess_text_config = get_config().preprocess_text_config


# 파일의 총 라인 수를 셈 (tqdm 등 진행바에 사용).
# 문법 팁: generator expression `sum(1 for _ in file)` 은 메모리를 더 적게 사용하면서
# 파일의 행 수를 효율적으로 계산합니다.
def count_lines(file_path: Path) -> int:
    """주어진 파일의 줄 수를 반환합니다.

    Args:
        file_path (Path): 읽을 파일의 Path 객체

    Returns:
        int: 파일의 총 행 수
    """
    with file_path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def write_error_log(error_log_path: Path, line: str, error: Exception) -> None:
    """에러가 발생한 라인과 예외 정보를 에러 로그 파일에 append 모드로 기록합니다.

    Args:
        error_log_path (Path): 에러 로그 파일 경로
        line (str): 에러가 발생한 원본 텍스트 라인
        error (Exception): 발생한 예외 객체
    """
    # 'a' 모드는 파일 끝에 추가(append)합니다. 기존 로그를 유지하면서 새로운 로그를 기록합니다.
    with error_log_path.open("a", encoding="utf-8") as error_log:
        error_log.write(f"{line.strip()}\n{error}\n\n")


def process_line(
    line: str,
    transcription_path: Path,
    correct_path: bool,
    use_jp_extra: bool,
    yomi_error: str,
) -> str:
    """한 라인을 파싱하고 정규화 및 음운/요미 정보를 추출한 뒤 포맷에 맞춰 반환합니다.

    입력 형식(한 줄):
        utt|spk|language|text
    - utt: 오디오 파일명(또는 `wavs/<utt>` 형태의 상대경로)
    - spk: 화자 ID 문자열
    - language: 언어 태그(예: 'JP', 'EN')
    - text: 원문 텍스트

    반환 형식(한 줄):
        utt|spk|language|norm_text|phones(space-separated)|tones(space-separated)|word2ph(space-separated)\n

    Args:
        line (str): 입력 라인
        transcription_path (Path): 원본 transcription 파일 경로 (경로 보정 시 사용)
        correct_path (bool): True이면 utt를 transcription 파일 기준의 wavs 디렉터리로 보정
        use_jp_extra (bool): 일본어 추가 옵션 사용 여부
        yomi_error (str): 요미(yomi) 에러 처리 방식 ('raise'|'skip'|'use')

    Raises:
        ValueError: 입력 라인 포맷이 잘못된 경우
    """
    # '|'로 분리하여 필드 개수를 검사
    splitted_line = line.strip().split("|")
    if len(splitted_line) != 4:
        # 잘못된 포맷의 라인은 명확히 예외를 던져 에러 로그로 남깁니다.
        raise ValueError(f"Invalid line format: {line.strip()}")

    utt, spk, language, text = splitted_line

    # clean_text는 다음을 반환: (norm_text, phones, tones, word2ph)
    # - norm_text: 정규화된 텍스트
    # - phones: phone sequence (list of str)
    # - tones: 톤/악센트 정보 (list of int)
    # - word2ph: 각 단어에 대한 phone 인덱스(또는 길이) 정보
    norm_text, phones, tones, word2ph = clean_text(
        text=text,
        language=language,  # type: ignore (typing 보강이 없는 경우 안전하게 무시)
        use_jp_extra=use_jp_extra,
        # yomi_error == 'use'인 경우 raise_yomi_error=False (예외 대신 일부 보정/대체 동작)
        raise_yomi_error=(yomi_error != "use"),
    )

    # correct_path가 True이면 transcription 파일의 상위 디렉토리 내 'wavs' 하위 경로로 utt를 보정
    if correct_path:
        utt = str(transcription_path.parent / "wavs" / utt)

    # format 문자열으로 출력 라인을 조합
    return "{}|{}|{}|{}|{}|{}|{}\n".format(
        utt,
        spk,
        language,
        norm_text,
        " ".join(phones),
        " ".join([str(i) for i in tones]),
        " ".join([str(i) for i in word2ph]),
    )


def preprocess(
    transcription_path: Path,
    cleaned_path: Optional[Path],
    train_path: Path,
    val_path: Path,
    config_path: Path,
    val_per_lang: int,
    max_val_total: int,
    # clean: bool,
    use_jp_extra: bool,
    yomi_error: str,
    correct_path: bool,
):
    """텍스트 파일을 정규화하고 train/validation 리스트 및 config를 업데이트하는 메인 함수.

    Args:
        transcription_path (Path): 원본 텍스트 파일(각 라인: utt|spk|language|text)
        cleaned_path (Optional[Path]): 전처리된 결과를 쓸 파일(지정 없으면 `<transcription>.cleaned`)
        train_path (Path): train 파일 리스트를 저장할 경로
        val_path (Path): val 파일 리스트를 저장할 경로
        config_path (Path): JSON config 파일 경로 (spk2id, n_speakers를 업데이트)
        val_per_lang (int): 화자별로 뽑을 validation 샘플 수 (언어별이 아님, 변수명 유의)
        max_val_total (int): 전체 validation 샘플 상한
        use_jp_extra (bool): 일본어 특별 옵션 사용여부
        yomi_error (str): 요미 에러 처리 방식: 'raise'|'skip'|'use'
        correct_path (bool): utt를 wavs 디렉토리 기준으로 보정할지 여부
    """
    assert yomi_error in ["raise", "skip", "use"]

    # cleaned_path가 빈 문자열이거나 None이면 transcription_path 기반으로 기본값 생성
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path.with_name(
            transcription_path.name + ".cleaned"
        )

    # 에러 로그 초기화(있으면 삭제)
    error_log_path = transcription_path.parent / "text_error.log"
    if error_log_path.exists():
        error_log_path.unlink()
    error_count = 0

    # 진행바를 위해 총 라인 수를 미리 센다
    total_lines = count_lines(transcription_path)

    # transcription_path로부터 한 줄씩 읽어 처리 후 cleaned_path에 기록
    # tqdm 및 SAFE_STDOUT 사용으로 콘솔에서 프로그레스 바 출력이 안정적이도록 함
    with (
        transcription_path.open("r", encoding="utf-8") as trans_file,
        cleaned_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in tqdm(
            trans_file, file=SAFE_STDOUT, total=total_lines, dynamic_ncols=True
        ):
            try:
                processed_line = process_line(
                    line,
                    transcription_path,
                    correct_path,
                    use_jp_extra,
                    yomi_error,
                )
                out_file.write(processed_line)
            except Exception as e:
                # 예외 발생시 해당 라인과 에러를 로그/파일에 기록
                logger.error(
                    f"An error occurred at line:\n{line.strip()}\n{e}", encoding="utf-8"
                )
                write_error_log(error_log_path, line, e)
                error_count += 1

    # cleaned_path를 읽을 준비
    transcription_path = cleaned_path

    # 화자별 발화 라인들을 수집하는 사전: {spk: [lines,...]}
    spk_utt_map: dict[str, list[str]] = defaultdict(list)

    # 화자 -> integer ID 매핑 (config에 저장할 spk2id 용도)
    spk_id_map: dict[str, int] = {}

    # 할당할 다음 화자 ID
    current_sid: int = 0

    # transcription 파일을 순회하며 오디오 존재 여부 검사 및 spk 목록 구성
    with transcription_path.open("r", encoding="utf-8") as f:
        audio_paths: set[str] = set()
        count_same = 0
        count_not_found = 0
        for line in f.readlines():
            # 파싱: 'utt|spk|...'
            utt, spk = line.strip().split("|")[:2]
            # 동일 오디오가 여러 라인에 등장하면 경고 및 스킵
            if utt in audio_paths:
                logger.warning(f"Same audio file appears multiple times: {utt}")
                count_same += 1
                continue
            # 오디오 파일이 실제로 존재하는지 확인
            if not Path(utt).is_file():
                logger.warning(f"Audio not found: {utt}")
                count_not_found += 1
                continue
            # 유효한 오디오는 집합에 추가하고 화자별 리스트에 라인을 append
            audio_paths.add(utt)
            spk_utt_map[spk].append(line)

            # 새로운 화자를 만나면 ID를 할당하고 증가시킴
            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1
        # 존재하지 않거나 중복된 오디오 통계 출력(필요시 사용자에게 알림)
        if count_same > 0 or count_not_found > 0:
            logger.warning(
                f"Total repeated audios: {count_same}, Total number of audio not found: {count_not_found}"
            )

    train_list: list[str] = []
    val_list: list[str] = []

    # 각 화자별로 발화(utts) 목록을 처리하여 validation 샘플을 뽑음
    # 주의: val_per_lang는 'language'가 아닌 'SPEAKER' 당 개수입니다 (변수명 호환성 때문에 유지됨)
    for spk, utts in spk_utt_map.items():
        if val_per_lang == 0:
            # validation을 뽑지 않으면 모두 train에 추가
            train_list.extend(utts)
            continue
        # 각 화자에서 랜덤하게 val_per_lang개의 인덱스를 선택
        val_indices = set(sample(range(len(utts)), val_per_lang))
        # 원래 순서를 유지하면서 val/train으로 분리
        for index, utt in enumerate(utts):
            if index in val_indices:
                val_list.append(utt)
            else:
                train_list.append(utt)

    # 전체 validation 수가 제한(max_val_total)을 초과하면 잘라서 초과분은 train으로 되돌림
    if len(val_list) > max_val_total:
        extra_val = val_list[max_val_total:]
        val_list = val_list[:max_val_total]
        # 초과된 validation 샘플을 train에 추가(순서 유지)
        train_list.extend(extra_val)

    # train/val 파일 쓰기
    with train_path.open("w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with val_path.open("w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # config JSON 로드, spk2id 및 n_speakers 업데이트
    with config_path.open("r", encoding="utf-8") as f:
        json_config = json.load(f)

    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)

    with config_path.open("w", encoding="utf-8") as f:
        # ensure_ascii=False로 한글/일본어가 깨지지 않게 저장
        json.dump(json_config, f, indent=2, ensure_ascii=False)

    # 에러 처리 요약: 처리 중 에러가 발생한 라인이 있으면 사용자에게 알립니다.
    if error_count > 0:
        if yomi_error == "skip":
            # 'skip' 모드면 에러 라인은 건너뛰고 나머지로 진행
            logger.warning(
                f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
            )
        else:
            # 'raise' 또는 'use'인 경우 (use는 내부에서 예외를 발생시키지 않도록 처리하므로,
            # 여기까지 도달하면 yomi 외의 예외가 발생한 경우임), 예외를 발생시켜 중단
            logger.error(
                f"An error occurred in {error_count} lines. Please check {error_log_path} for details."
            )
            raise Exception(
                f"An error occurred in {error_count} lines. Please check `Data/you_model_name/text_error.log` file for details."
            )
            # 何故か{error_log_path}をraiseすると文字コードエラーが起きるので上のように書いている
    else:
        logger.info(
            "Training set and validation set generation from texts is complete!"
        )


if __name__ == "__main__":
    # CLI: 각 인자는 기본값을 preprocess_text_config에서 가져오며 필요 시 오버라이드할 수 있습니다.
    parser = argparse.ArgumentParser(
        description="Preprocess transcription texts and generate train/val lists and update config."
    )
    parser.add_argument(
        "--transcription-path", default=preprocess_text_config.transcription_path,
        help="Path to the raw transcription file (each line: utt|spk|language|text)",
    )
    parser.add_argument(
        "--cleaned-path", default=preprocess_text_config.cleaned_path,
        help="Output path for cleaned texts (optional)",
    )
    parser.add_argument("--train-path", default=preprocess_text_config.train_path, help="Output train list path")
    parser.add_argument("--val-path", default=preprocess_text_config.val_path, help="Output validation list path")
    parser.add_argument("--config-path", default=preprocess_text_config.config_path, help="JSON config to update spk2id/n_speakers")

    # val_per_lang는 SPEAKER 당 validation 개수임(변수명이 혼동될 수 있으니 주의)
    parser.add_argument(
        "--val-per-lang",
        default=preprocess_text_config.val_per_lang,
        help="Number of validation data per SPEAKER, not per language (due to compatibility with the original code).",
    )
    parser.add_argument("--max-val-total", default=preprocess_text_config.max_val_total, help="Maximum total number of validation samples")
    parser.add_argument("--use_jp_extra", action="store_true", help="Enable Japanese extra mode for cleaning")
    parser.add_argument("--yomi_error", default="raise", help="Yomi error handling: 'raise'|'skip'|'use'")
    parser.add_argument("--correct_path", action="store_true", help="Correct utt path to use <transcription_parent>/wavs/<utt>")

    args = parser.parse_args()

    # CLI에서 전달된 인자들을 적절한 타입/Path로 변환
    transcription_path = Path(args.transcription_path)
    cleaned_path = Path(args.cleaned_path) if args.cleaned_path else None
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    config_path = Path(args.config_path)
    val_per_lang = int(args.val_per_lang)
    max_val_total = int(args.max_val_total)
    use_jp_extra: bool = args.use_jp_extra
    yomi_error: str = args.yomi_error
    correct_path: bool = args.correct_path

    preprocess(
        transcription_path=transcription_path,
        cleaned_path=cleaned_path,
        train_path=train_path,
        val_path=val_path,
        config_path=config_path,
        val_per_lang=val_per_lang,
        max_val_total=max_val_total,
        use_jp_extra=use_jp_extra,
        yomi_error=yomi_error,
        correct_path=correct_path,
    )
