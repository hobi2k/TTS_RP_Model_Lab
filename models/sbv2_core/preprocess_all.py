"""
preprocess_all.py
-----------------
이 스크립트는 데이터 전처리 파이프라인을 일괄 실행하기 위한 엔트리 포인트입니다.
주요 역할:
- 일본어 전처리를 위한 pyopenjtalk 워커를 초기화
- 사용자 사전(user dict)을 적용하여 yomi(요미) 변환을 보정
- 커맨드라인 인자를 받아 `gradio_tabs.train.preprocess_all` 함수를 호출

Python 문법/패턴 설명(이 파일에서 주로 사용되는 것):
- if __name__ == "__main__":
    - 이 블록은 파일을 직접 실행할 때만 실행되며, 다른 모듈에서 import될 때는 실행되지 않습니다.
- argparse: 커맨드라인 인자 파싱을 위한 표준 모듈로, `add_argument()`로 각 옵션을 정의합니다.
- action="store_true": 해당 옵션을 명령행에서 주어지면 True, 없으면 False로 설정합니다.
- cpu_count(): 시스템의 CPU 코어 수를 반환합니다. 병렬 프로세스 수 기본값 계산에 사용합니다.

아래에서는 모듈 임포트와 초기화, 그리고 argparse를 통한 인자 파싱 예제를 확인할 수 있습니다.
"""

import argparse
from multiprocessing import cpu_count

# 실제 전처리 로직은 gradio_tabs.train 모듈의 preprocess_all 함수에 구현되어 있음
# 이 스크립트는 그 함수를 편리하게 CLI에서 호출하게 해 주는 역할을 수행한다.
from gradio_tabs.train import preprocess_all

# 일본어 전처리용 워커/유틸을 제공하는 내부 모듈
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict


# ※ 실행 시점에 워커를 초기화합니다.
# 설명: pyopenjtalk은 내부적으로 subprocess나 별도 워커 프로세스를 이용하는 경우가 있으므로,
# 이 스크립트 프로세스에서 미리 워커를 초기화하면 이후 전처리 과정에서 동일한 워커를 재사용할 수 있습니다.
# 문법 팁: 함수 호출은 `module.func()` 형태이며, 반환값을 변수로 받지 않으면 단순히 부수효과(부작용)를 발생시킵니다.
pyopenjtalk_worker.initialize_worker()

# 사용자 정의 사전(dict_data/)을 pyopenjtalk에 적용
# update_dict()는 디스크의 사용자 사전을 읽어 pyopenjtalk의 내부 사전에 병합합니다.
update_dict()


if __name__ == "__main__":
    # argparse.ArgumentParser(): 커맨드라인 인자를 정의하고 파싱하기 위한 객체
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for dataset preparation."
    )

    # 필수 인자 예시: --model_name (-m) (required=True)
    # type: 인자 값의 타입을 지정 (e.g., str, int)
    # help: --help 출력 시 표시될 설명 문자열
    parser.add_argument(
        "--model_name", "-m", type=str, help="Model name", required=True
    )

    # 선택 인자 예시: 기본값(default)을 지정해 옵션이 없을 때의 동작을 정의
    parser.add_argument(
        "--batch_size", "-b", type=int, help="Batch size", default=2
    )
    parser.add_argument("--epochs", "-e", type=int, help="Epochs", default=100)
    parser.add_argument(
        "--save_every_steps",
        "-s",
        type=int,
        help="Save every steps",
        default=1000,
    )

    # 시스템 코어 수 활용: 기본적으로 CPU 코어의 반을 사용하도록 설정
    # cpu_count()는 OS에 따라 정확한 코어 수를 반환합니다.
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes",
        default=cpu_count() // 2,
    )

    # boolean flag: action='store_true' 를 사용하면 옵션이 주어졌을 때 True
    # (예: --normalize 를 명령행에 쓰면 args.normalize == True)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Trim silence",
    )

    # 모델/훈련 관련 freeze 옵션들: 지정하면 해당 부분을 고정(freeze)하여 업데이트하지 않음
    parser.add_argument(
        "--freeze_EN_bert",
        action="store_true",
        help="Freeze English BERT",
    )
    parser.add_argument(
        "--freeze_JP_bert",
        action="store_true",
        help="Freeze Japanese BERT",
    )
    parser.add_argument(
        "--freeze_ZH_bert",
        action="store_true",
        help="Freeze Chinese BERT",
    )
    parser.add_argument(
        "--freeze_style",
        action="store_true",
        help="Freeze style vector",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        help="Freeze decoder",
    )

    # JP extra 모델 사용 여부 (일본어 특화 모드)
    parser.add_argument(
        "--use_jp_extra",
        action="store_true",
        help="Use JP-Extra model",
    )

    # 검증/로깅 관련 옵션
    parser.add_argument(
        "--val_per_lang",
        type=int,
        help="Validation per language",
        default=0,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Log interval",
        default=200,
    )

    # yomi(요미) 처리 오류 시 동작 방식: 'raise' (예외 발생), 'skip' (스킵), 'use' (대체 사용)
    parser.add_argument(
        "--yomi_error",
        type=str,
        help="Yomi error handling. Options: 'raise' (예외), 'skip', 'use'",
        default="raise",
    )

    # parse_args(): 실제 커맨드라인 인자들을 읽어 args 네임스페이스로 반환
    args = parser.parse_args()

    # 실제 전처리 함수 호출: 인자를 명시적으로 전달(키워드 인자 사용)
    # 문법 팁: 함수 호출 시 `func(a=..., b=...)` 형태로 키워드 인자를 사용하면
    # 전달 순서에 의존하지 않고 명확하게 값을 지정할 수 있습니다.
    preprocess_all(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        num_processes=args.num_processes,
        normalize=args.normalize,
        trim=args.trim,
        freeze_EN_bert=args.freeze_EN_bert,
        freeze_JP_bert=args.freeze_JP_bert,
        freeze_ZH_bert=args.freeze_ZH_bert,
        freeze_style=args.freeze_style,
        freeze_decoder=args.freeze_decoder,
        use_jp_extra=args.use_jp_extra,
        val_per_lang=args.val_per_lang,
        log_interval=args.log_interval,
        yomi_error=args.yomi_error,
    )

