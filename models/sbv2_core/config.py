"""
config.py
---------
프로젝트 전반에서 사용하는 설정(config)을 읽어 객체로 제공하는 모듈입니다.
- `configs/paths.yml`에서 데이터셋/자산(root) 경로를 읽고 `PathConfig`를 생성합니다.
- `config.yml`(또는 default_config.yml을 복사해 생성)에서 모델별 설정을 읽어 `Config` 인스턴스를 생성합니다.

이 모듈의 주요 책임:
- YAML 파일로부터 설정 읽기
- 경로(상대경로)를 적절히 Path 객체로 변환
- 서브 설정(리샘플링, 전처리, BERT 등)의 클래스를 초기화하여 한 곳에서 접근 가능하게 함

문법/구조 팁:
- `from_dict` 클래스 메서드는 YAML에서 읽은 dict를 받아서 경로를 dataset 기준으로 확장하고 해당 클래스의 인스턴스를 생성합니다.
- 경로 관련 처리는 프로그램 초기 단계에서만 수행되므로, 이후 파일 존재 여부 등은 각 도구(예: resample.py)에서 체크합니다.
"""

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from style_bert_vits2.logging import logger


class PathConfig:
    """파일/디렉토리 루트 경로를 담는 단순 컨테이너 클래스입니다.

    Args:
        dataset_root (str): 데이터셋이 저장되는 루트 디렉토리 (예: 'Data')
        assets_root (str): 모델 아웃풋(assets)을 저장하는 루트 디렉토리 (예: 'model_assets')

    Note:
        - 초기화 시 문자열을 pathlib.Path로 변환하여 이후에 경로 연산을 안전하게 할 수 있도록 했습니다.
    """

    def __init__(self, dataset_root: str, assets_root: str):
        # Path로 변환하면 / 연산자 등을 사용해 하위 경로를 쉽게 결합할 수 있습니다.
        self.dataset_root = Path(dataset_root)
        self.assets_root = Path(assets_root)


# CUDA 사용 가능 여부를 전역 플래그로 확인합니다.
# 이후 각 구성 클래스에서 디바이스 기본값으로 'cuda'가 주어졌을 때, 이 플래그를 보고
# CUDA가 없으면 자동으로 'cpu'로 변경하는 패턴을 사용합니다.
cuda_available = torch.cuda.is_available()


class Resample_config:
    """리샘플링(Resample) 관련 설정을 보관하는 클래스입니다.

    속성:
        sampling_rate (int): 목표 샘플레이트
        in_dir (Path): 입력(원본) 오디오 디렉토리
        out_dir (Path): 리샘플링 결과를 저장할 디렉토리

    from_dict 메서드:
        - YAML에서 읽은 상대 경로를 dataset_path 기준으로 확장해서 실제 Path로 변환합니다.
        - 경로 유효성 검사는 실제 처리를 수행하는 스크립트(resample.py)에서 수행됩니다.
    """

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate = sampling_rate  # 목표 샘플레이트
        self.in_dir = Path(in_dir)  # 처리할 오디오 디렉토리
        self.out_dir = Path(out_dir)  # 리샘플링 결과 저장 디렉토리

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """딕셔너리에서 인스턴스를 생성할 때 dataset_path 기준으로 경로를 확장합니다."""

        # dataset_path는 보통 `Data/{model_name}` 와 같은 루트 디렉토리입니다.
        data["in_dir"] = dataset_path / data["in_dir"]
        data["out_dir"] = dataset_path / data["out_dir"]

        return cls(**data)


class Preprocess_text_config:
    """텍스트 전처리 관련 설정을 담는 클래스입니다.

    속성 설명:
        transcription_path (Path): 원본 전사(transcription) 파일 경로
        cleaned_path (Path or ''): 전처리된 파일 저장 경로(빈 문자열이면 기본값으로 설정)
        train_path (Path): 생성될 train 리스트 파일 경로
        val_path (Path): 생성될 validation 리스트 파일 경로
        config_path (Path): 업데이트할 config.json 경로
        val_per_lang (int): 화자 단위로 뽑을 validation 수 (변수명은 val_per_lang이나 화자당임)
        max_val_total (int): 전체 validation 수 상한
        clean (bool): 전처리 수행 여부 플래그
    """

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        self.transcription_path = Path(transcription_path)
        self.train_path = Path(train_path)
        # cleaned_path가 비어있거나 None이면 transcription_path에 '.cleaned'를 붙여 기본 경로로 설정
        if cleaned_path == "" or cleaned_path is None:
            self.cleaned_path = self.transcription_path.with_name(
                self.transcription_path.name + ".cleaned"
            )
        else:
            self.cleaned_path = Path(cleaned_path)
        self.val_path = Path(val_path)
        self.config_path = Path(config_path)
        self.val_per_lang = val_per_lang
        self.max_val_total = max_val_total
        self.clean = clean

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """YAML에서 읽은 상대 경로를 dataset_path 기준으로 확장합니다."""

        data["transcription_path"] = dataset_path / data["transcription_path"]
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            # 빈 문자열은 기본값 사용(signify no explicit cleaned path)
            data["cleaned_path"] = ""
        else:
            data["cleaned_path"] = dataset_path / data["cleaned_path"]
        data["train_path"] = dataset_path / data["train_path"]
        data["val_path"] = dataset_path / data["val_path"]
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Bert_gen_config:
    """BERT 기반 벡터 생성 관련 설정입니다.

    - device는 기본적으로 'cuda'지만 시스템에서 CUDA를 사용할 수 없으면 'cpu'로 자동 전환됩니다.
    """

    def __init__(
        self,
        config_path: str,
        num_processes: int = 1,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = Path(config_path)
        self.num_processes = num_processes
        # CUDA가 없을 경우 안전하게 CPU로 fallback
        if not cuda_available:
            device = "cpu"
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Style_gen_config:
    """스타일 벡터(style) 생성 관련 설정.

    device 기본값 또한 CUDA 가능 여부에 따라 자동으로 CPU로 전환됩니다.
    """

    def __init__(
        self,
        config_path: str,
        num_processes: int = 4,
        device: str = "cuda",
    ):
        self.config_path = Path(config_path)
        self.num_processes = num_processes
        if not cuda_available:
            device = "cpu"
        self.device = device

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Train_ms_config:
    """학습(Train) 관련 설정을 담는 클래스입니다.

    주요 속성:
        env (dict): 학습에 필요한 환경변수(MASTER_ADDR 등)를 지정할 수 있으며, run()에서 이들을 os.environ에 주입합니다.
        model_dir (Path): 체크포인트 등 학습 산출물을 저장할 디렉토리 (YAML에 상대경로로 명시되며 dataset_path를 기본으로 결합됩니다.)
        config_path (Path): 학습 관련 세부 설정 파일 경로
        num_workers (int): DataLoader 등의 워커 수
        spec_cache (bool): 멜 스펙트럼 캐시 사용 여부
        keep_ckpts (int): 유지할 체크포인트 수

    Note:
        - model_dir는 dataset_path 기준의 상대 경로로 주어지는 것이 일반적입니다. (예: `Data/{model_name}/models`)
    """

    def __init__(
        self,
        config_path: str,
        env: dict[str, Any],
        # base: Dict[str, any],
        model_dir: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        self.env = env  # 로드할 환경 변수들 (예: MASTER_ADDR, WORLD_SIZE 등)
        # self.base = base  # 하위 모델/베이스 설정 (미사용)
        self.model_dir = Path(
            model_dir
        )  # 학습 아티팩트 저장 디렉토리 (dataset_path 상대 경로인 경우가 많음)
        self.config_path = Path(config_path)  # 학습용 config 파일 경로
        self.num_workers = num_workers  # 데이터 로더 워커 수
        self.spec_cache = spec_cache  # mel 스펙 캐시 사용 여부
        self.keep_ckpts = keep_ckpts  # 보존할 체크포인트 개수

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        # dataset_path와 결합해 config_path를 절대/상대 경로에 맞게 구성
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Webui_config:
    """Web UI 관련 설정(현재 webui.py에서 사용됨).

    - device는 CUDA 사용 가능 여부에 따라 'cpu'로 자동 전환될 수 있습니다.
    """

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        # CUDA가 없으면 자동으로 CPU로 변경
        if not cuda_available:
            device = "cpu"
        self.device = device
        self.model = Path(model)
        self.config_path = Path(config_path)
        self.port: int = port
        self.share: bool = share
        self.debug: bool = debug
        self.language_identification_library: str = language_identification_library

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]
        data["model"] = dataset_path / data["model"]
        return cls(**data)


class Server_config:
    """서버(배포) 관련 설정.

    - device는 'cuda' 기본값이더라도 시스템에서 CUDA를 사용할 수 없으면 'cpu'로 변경됩니다.
    """

    def __init__(
        self,
        port: int = 5000,
        device: str = "cuda",
        limit: int = 100,
        language: str = "JP",
        origins: list[str] = ["*"],
    ):
        self.port: int = port
        if not cuda_available:
            device = "cpu"
        self.device: str = device
        self.language: str = language
        self.limit: int = limit
        self.origins: list[str] = origins

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class Translate_config:
    """번역 API 관련 설정(예: 외부 번역 서비스 키)"""

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class Config:
    """프로젝트의 전체 설정을 로드하고, 하위 구성들(Resample, Preprocess, Train 등)을 초기화합니다.

    동작 요약:
    1. `config_path`가 없으면 `default_config.yml`을 복사하여 기본 config.yml을 생성합니다.
    2. YAML을 파싱하여 `model_name`을 기준으로 `dataset_path`를 결정합니다.
       - YAML에 `dataset_path`가 명시되어 있으면 그 값을 사용하고, 없으면 `path_config.dataset_root / model_name`을 사용합니다.
    3. 각 하위 구성 (`Resample_config`, `Preprocess_text_config`, ...)을 `from_dict`를 통해 생성합니다.

    주의:
    - 이 시점에서는 파일 존재 유무까지 검증하지 않습니다(예: train dataset 폴더의 유효성 검사 등은
      각 스크립트에서 수행). 이 모듈은 설정의 '해석'에 집중합니다.
    """

    def __init__(self, config_path: str, path_config: PathConfig):
        # config.yml이 없으면 기본 설정 파일을 복사해 생성
        if not Path(config_path).exists():
            shutil.copy(src="default_config.yml", dst=config_path)
            logger.info(
                f"A configuration file {config_path} has been generated based on the default configuration file default_config.yml."
            )
            logger.info(
                "Please do not modify default_config.yml. Instead, modify config.yml."
            )
            # sys.exit(0)

        # YAML 파일을 읽어 설정을 파싱
        with open(config_path, encoding="utf-8") as file:
            yaml_config: dict[str, Any] = yaml.safe_load(file.read())
            model_name: str = yaml_config["model_name"]
            self.model_name: str = model_name

            # dataset_path가 설정 파일에 명시적으로 있으면 그것을 사용하고,
            # 없으면 path_config의 dataset_root와 model_name을 결합해 기본값을 만듭니다.
            if "dataset_path" in yaml_config:
                dataset_path = Path(yaml_config["dataset_path"])
            else:
                dataset_path = path_config.dataset_root / model_name

            self.dataset_path = dataset_path
            # path_config(예: configs/paths.yml)에서 정의된 루트 경로들을 저장
            self.dataset_root = path_config.dataset_root
            self.assets_root = path_config.assets_root

            # 모델별 아웃풋 디렉토리 (예: model_assets/{model_name})
            self.out_dir = self.assets_root / model_name

            # 하위 설정을 초기화 (from_dict에서 dataset_path 기준으로 경로를 확장함)
            self.resample_config: Resample_config = Resample_config.from_dict(
                dataset_path, yaml_config["resample"]
            )
            self.preprocess_text_config: Preprocess_text_config = (
                Preprocess_text_config.from_dict(
                    dataset_path, yaml_config["preprocess_text"]
                )
            )
            self.bert_gen_config: Bert_gen_config = Bert_gen_config.from_dict(
                dataset_path, yaml_config["bert_gen"]
            )
            self.style_gen_config: Style_gen_config = Style_gen_config.from_dict(
                dataset_path, yaml_config["style_gen"]
            )
            self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
                dataset_path, yaml_config["train_ms"]
            )
            self.webui_config: Webui_config = Webui_config.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.server_config: Server_config = Server_config.from_dict(
                yaml_config["server"]
            )
            # self.translate_config: Translate_config = Translate_config.from_dict(
            #     yaml_config["translate"]
            # )


# Load and initialize the configuration


def get_path_config() -> PathConfig:
    """`configs/paths.yml` 파일을 읽어 `PathConfig`를 반환합니다.

    동작:
    - `configs/paths.yml`가 존재하지 않으면 `configs/default_paths.yml`를 복사하여 생성합니다.
    - 파일의 내용을 YAML로 파싱해 PathConfig(**dict)로 반환합니다.
    """
    path_config_path = Path("configs/paths.yml")
    if not path_config_path.exists():
        shutil.copy(src="configs/default_paths.yml", dst=path_config_path)
        logger.info(
            f"A configuration file {path_config_path} has been generated based on the default configuration file default_paths.yml."
        )
        logger.info(
            "Please do not modify configs/default_paths.yml. Instead, modify configs/paths.yml."
        )
    with open(path_config_path, encoding="utf-8") as file:
        path_config_dict: dict[str, str] = yaml.safe_load(file.read())
    return PathConfig(**path_config_dict)


def get_config() -> Config:
    """전역 설정(Config) 객체를 반환합니다.

    - 내부적으로 `configs/paths.yml`을 읽어 `PathConfig`를 만들고, `config.yml`을 기반으로 `Config`를 생성합니다.
    - 구버전의 config.yml 포맷 오류(TypeError/KeyError)가 발생하면 default_config.yml로 덮어씌우고 다시 시도합니다.
    """
    path_config = get_path_config()
    try:
        config = Config("config.yml", path_config)
    except (TypeError, KeyError):
        # 이전 버전의 config.yml 형식 문제로 로딩에 실패한 경우 기본 구성으로 교체
        logger.warning("Old config.yml found. Replace it with default_config.yml.")
        shutil.copy(src="default_config.yml", dst="config.yml")
        config = Config("config.yml", path_config)

    return config