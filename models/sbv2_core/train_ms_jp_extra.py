"""
train_ms_jp_extra.py
--------------------
이 파일은 VITS 계열의 음성 합성(텍스트-투-스피치) 모델을 학습하기 위한 스크립트입니다.
주요 역할:
- 분산 학습 환경 초기화 (torch.distributed)
- 데이터 로딩 및 배치 샘플러 설정
- Generator / Discriminator 및 추가 판별자 초기화
- 학습 루프(train_and_evaluate) 및 평가(evaluate) 제공

주석 작성 방식(이 파일 내 주석 컨벤션):
- 모듈 최상단에는 파일 목적과 간단한 설명을 작성합니다.
- 복잡한 블록(분산 설정, 학습 루프 등)에는 블록 수준 설명과 '왜 이렇게 했는가'를 적습니다.
- 함수에는 입출력(입력 파라미터, 반환값)과 주요 부작용(예: GPU로 텐서를 이동)을 설명합니다.

문법 팁:
- `import module as alias` 형태는 긴 모듈명을 축약할 때 사용합니다 (예: `torch.nn.functional as F`).
- from-import 형태(`from pkg import A, B`)는 패키지 내부의 특정 심볼을 직접 가져와 네임스페이스를 깔끔하게 합니다.
- 분산/멀티 GPU 코드에서는 `.cuda(device)` 또는 `.to(device)` 로 명시적으로 장치를 지정하는 것이 안전합니다.

아래는 표준 라이브러리 / 외부 라이브러리 import 섹션입니다. 각 라인은 어떤 역할을 하는지 간단히 주석 처리합니다.

tensorboard --logdir /mnt/d/my_tts_dataset/Saya/models --port 6006


torchrun --nproc_per_node=1 train_ms_jp_extra.py \
  -c configs/config_jp_extra.json \
  -m /mnt/d/my_tts_dataset/mai \
  --skip_default_style
"""

# 표준 라이브러리: 운영체제, 시간, gc(가비지 컬렉션) 등 유틸리티
import argparse  # CLI 인자 파싱
import datetime  # 타임스탬프 생성 등 시간 관련
import gc  # 가비지 수집(메모리 정리) 명시적 호출
import os  # 파일 경로 및 환경변수
import platform  # 플랫폼 정보 확인(Windows vs Linux 등)

# 외부 라이브러리: PyTorch, HF Hub, AMP, DDP, DataLoader, TensorBoard, tqdm 등
import torch
import torch.distributed as dist  # 분산 학습 관련 API
from huggingface_hub import HfApi  # 모델/파일 업로드(옵션)
from torch.cuda.amp import GradScaler, autocast  # 혼합 정밀도 연산(AMP)
from torch.nn import functional as F  # 손실 함수 등 간단한 NN 함수 사용
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP 래퍼
from torch.utils.data import DataLoader  # 데이터 적재
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅
from tqdm import tqdm  # 프로그레스 바
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler  # 길이 기반 분산 샘플러

# 프로젝트 내부 모듈: 데이터 로더, 손실함수, 멜 스펙트로그램 처리, logger, 모델 유틸리티 등
# logging.getLogger("numba").setLevel(logging.WARNING)  # (필요 시) 서브모듈 로그 레벨 조정
import default_style  # 스타일(voice) 관련 초기화/저장 유틸
from config import get_config  # 공통 설정 로드
from data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from losses import WavLMLoss, discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models_jp_extra import (
    DurationDiscriminator,
    MultiPeriodDiscriminator,
    SynthesizerTrn,
    WavLMDiscriminator,
)
from style_bert_vits2.nlp.symbols import SYMBOLS
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# 1. PyTorch / CUDA 전역 설정
# TF32 및 Flash-Attention 사용 설정 (성능 최적화)
# 아래 설정들은 GPU/라이브러리 버전, 하드웨어에 따라 성능과 수치 안정성(정확도)에 영향을 줍니다.
# - TF32: TensorFloat-32는 NVIDIA Ampere 이후의 GPU에서 빠른 행렬 연산을 가능하게 하는 포맷입니다.
#   다만 수치적 안정성이 중요한 경우(특히 매우 작은 손실값을 다루는 경우) 비활성화해야 할 수 있습니다.
# - Flash-SDP / mem-efficient SDP: Self-Dot-Product 연산(일부 어텐션 최적화)에서 성능을 개선합니다.
#   사용 가능한지(또는 안전한지)는 PyTorch 버전과 GPU 드라이버에 따라 다릅니다.
# 참고: 아래 `alow_tf32` 는 모듈 속성 이름이 오타일 가능성이 있으므로, PyTorch가 해당 속성을 인식하지 못하면
# AttributeError가 발생하거나 의도와 다른 동작을 할 수 있습니다. (정확한 속성명은 `allow_tf32` 등일 수 있음)
try:
    # matmul에서 TF32 사용 허용 (행렬곱 성능 개선)
    torch.backends.cuda.matmul.alow_tf32 = True
except Exception:
    # 안전하게 접근하기 위해 예외 처리. 실제 환경에서는 `allow_tf32` 속성이 맞는지 확인 권장
    pass
try:
    torch.backends.cudnn.allow_tf32 = (
        True  # 문제가 발생하면 False로 설정해 TF32를 비활성화해보세요.
    )
except Exception:
    pass

# 멀티스레드 설정: CPU 측 수치 연산에 사용되는 스레드 수를 제한 (컨텍스트에 따라 조정)
torch.set_num_threads(1)

# float32 행렬 곱셈 정밀도 설정: 'medium'은 속도/정밀도 균형 옵션임
torch.set_float32_matmul_precision("medium")

# SDP(Self Dot Product) 커널 및 플래시 SDP 활성화: 어텐션 연산 관련 최적화
try:
    torch.backends.cuda.sdp_kernel("flash")
    torch.backends.cuda.enable_flash_sdp(True)
    # 메모리 효율적 SDP는 PyTorch >= 2.0 에서만 지원될 수 있음
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    # 구버전 PyTorch / 드라이버에서는 해당 옵션이 없을 수 있으므로 무시
    pass


# 전역 설정 로드
config = get_config()
global_step = 0

api = HfApi()

# CLI 인자 파싱
def run():
    """Project entry point for training.

    동작 순서 (요약):
    1. CLI 인자 파싱 (가능하면 config.yml 사용 권장)
    2. 분산 학습 환경 초기화 (env:// 방식, torch.distributed)
    3. 하이퍼파라미터(hps) 및 모델 경로 세팅
    4. 데이터로더 및 배치 샘플러 초기화
    5. 모델/판별자 생성, 옵티마이저/스케줄러 설정 및 체크포인트 로드
    6. 학습 루프 호출

    주요 지역 변수 설명:
    - args: argparse.Namespace, 커맨드라인 인자
    - rank: 분산 프로세스의 전역 순번 (0이 메인 프로세스)
    - local_rank: 프로세스에서 사용할 GPU 장치 번호
    - hps: HyperParameters 객체 (JSON에서 로드됨)

    주의:
    - 가능하면 config.yml 또는 JSON 파일을 사용해 설정을 관리하세요. CLI는 간단한 오버라이드 용도로만 사용하세요.
    """
    # Command line configuration is not recommended unless necessary, use config.yml
    parser = argparse.ArgumentParser()
    # config: 학습에 사용할 하이퍼파라미터 파일 경로 (JSON). 주로 모델/학습 관련 모든 설정이 이 파일에 들어있습니다.
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=config.train_ms_config.config_path,
        help="JSON file for configuration",
    )
    # model: 데이터셋 및 모델을 저장할 루트 경로. 기본값은 config에 설정된 dataset 경로입니다.
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path to the data set folder. Please note that data is no longer stored in the /logs folder by default. If you need to configure via the command line, specify the path relative to the root directory.",
        default=config.dataset_path,
    )
    # assets_root: 추론(inference)에 필요한 모델 자산들이 위치하는 루트 디렉터리
    parser.add_argument(
        "--assets_root",
        type=str,
        help="Root directory of model assets needed for inference.",
        default=config.assets_root,
    )
    # skip_default_style: 기본 스타일(voice) 파일을 자동 저장하지 않음
    parser.add_argument(
        "--skip_default_style",
        action="store_true",
        help="Skip saving default style config and mean vector.",
    )
    # no_progress_bar: 대량 학습 시 tqdm 프로그레스바 비활성화
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        help="Do not show the progress bar while training.",
    )
    # speedup: 로깅과 검증을 비활성화해 학습 속도에 집중 (디버깅/모니터링이 줄어듦)
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Speed up training by disabling logging and evaluation.",
    )
    # HuggingFace hub에 모델을 업로드할 repo id (옵션)
    parser.add_argument(
        "--repo_id",
        help="Huggingface model repo id to backup the model.",
        default=None,
    )
    # 분산 학습에서 커스텀 배치 샘플러를 사용할지 여부
    parser.add_argument(
        "--not_use_custom_batch_sampler",
        help="Don't use custom batch sampler for training, which was used in the version < 2.5",
        action="store_true",
    )
    args = parser.parse_args()

    # 로그 파일 생성
    model_dir = os.path.join(args.model, config.train_ms_config.model_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(os.path.join(args.model, f"train_{timestamp}.log"))

    # Parsing environment variables
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            logger.info(f"Loading configuration from config {env_value!s}")
            os.environ[env_name] = str(env_value)
    # 환경 변수를 로그로 출력(없을 수도 있으므로 안전하게 get 사용)
    logger.info(
        "Loading environment variables \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ.get("MASTER_ADDR", "<not set>"),
            os.environ.get("MASTER_PORT", "<not set>"),
            os.environ.get("WORLD_SIZE", "<not set>"),
            os.environ.get("RANK", "<not set>"),
            os.environ.get("LOCAL_RANK", "<not set>"),
        )
    )

    backend = "nccl"
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    rank = dist.get_rank()
    # LOCAL_RANK가 설정되지 않은 경우 안전하게 기본값 0 사용
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    n_gpus = dist.get_world_size()

    hps = HyperParameters.load_from_json(args.config)
    # This is needed because we have to pass values to `train_and_evaluate()
    hps.model_dir = model_dir
    hps.speedup = args.speedup
    hps.repo_id = args.repo_id

    # CLI에서 지정한 config 경로가 내부 표준 경로와 다르면 파일을 복사합니다.
    # (이렇게 하면 프로젝트 내부에서 항상 동일한 config 경로를 참조할 수 있음)
    if os.path.realpath(args.config) != os.path.realpath(
        config.train_ms_config.config_path
    ):
        with open(args.config, encoding="utf-8") as f:
            data = f.read()
        os.makedirs(os.path.dirname(config.train_ms_config.config_path), exist_ok=True)
        with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
            f.write(data)

    """
    Path constants are a bit complicated...
    TODO: Refactor or rename these?
    (Both `config.yml` and `config.json` are used, which is confusing I think.)

    args.model: For saving all info needed for training.
        default: `Data/{model_name}`.
    hps.model_dir := model_dir: For saving checkpoints (for resuming training).
        default: `Data/{model_name}/models`.
        (Use `hps` since we have to pass `model_dir` to `train_and_evaluate()`.

    args.assets_root: The root directory of model assets needed for inference.
        default: config.assets_root == `model_assets`.

    config.out_dir: The directory for model assets of this model (for inference).
        default: `model_assets/{model_name}`.
    """

    if args.repo_id is not None:
        # Hugging Face Hub에 업로드 시도: repo가 존재하는지 확인하기 위해 먼저 config를 업로드합니다.
        # 업로드가 실패하면 명령행에서 `huggingface-cli login`으로 로그인되어 있는지 확인하세요.
        try:
            api.upload_file(
                path_or_fileobj=args.config,
                path_in_repo=f"Data/{config.model_name}/config.json",
                repo_id=hps.repo_id,
            )
        except Exception as e:
            logger.error(e)
            logger.error(
                f"Failed to upload files to the repo {hps.repo_id}. Please check if the repo exists and you have logged in using `huggingface-cli login`."
            )
            # 실수로 학습을 계속하면 안될 수 있으므로 에러를 재발생시킵니다.
            raise e
        # 데이터 폴더 업로드(체크포인트 복구용). `delete_patterns`는 업로드 시 특정 파일만 유지하게 도움을 줌
        api.upload_folder(
            repo_id=hps.repo_id,
            folder_path=config.dataset_path,
            path_in_repo=f"Data/{config.model_name}",
            delete_patterns="*.pth",  # Only keep the latest checkpoint
            ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
            run_as_future=True,
        )

    # 출력 디렉토리(모델 아웃풋, assets 등) 생성
    os.makedirs(config.out_dir, exist_ok=True)

    # 기본 스타일 저장: 학습에 사용되는 음색(style) 벡터를 자동으로 추출/저장
    if not args.skip_default_style:
        default_style.save_styles_by_dirs(
            os.path.join(args.model, "wavs"),
            config.out_dir,
            config_path=args.config,
            config_output_path=os.path.join(config.out_dir, "config.json"),
        )

    # 시드 고정: 재현성 확보를 위해 CPU/GPU 연산의 시드를 고정합니다.
    # 주의: 분산 환경에서 완전한 재현성을 보장하려면 추가 설정이 필요할 수 있습니다.
    torch.manual_seed(hps.train.seed)
    # 현재 프로세스가 사용할 CUDA 디바이스를 명시적으로 지정합니다. local_rank는 torchrun에서 할당된 GPU 인덱스입니다.
    torch.cuda.set_device(local_rank)

    global global_step
    # TensorBoard writer 초기화 (rank 0 프로세스만 수행)
    # rank 0은 보통 메인 프로세스로 로깅/평가를 담당합니다. speedup 옵션이 켜진 경우 로깅/평가를 비활성화합니다.
    writer = None
    writer_eval = None
    if rank == 0 and not args.speedup:
        # 현재 Git 커밋 해시를 체크해 설정/코드 일관성을 기록
        utils.check_git_hash(model_dir)
        # 학습 로그(학습 중 스칼라/이미지)를 기록할 SummaryWriter
        writer = SummaryWriter(log_dir=model_dir)
        # 평가용 writer (평가 시 스칼라/오디오 저장)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
    # 데이터셋 및 Collate 함수 설정
    # TextAudioSpeakerLoader: (텍스트, 멜 스펙, 오디오, 스피커 등) 을 배치 단위로 읽는 커스텀 로더
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # collate_fn: 배치 내샘플을 하나의 텐서로 만드는 함수, use_jp_extra는 일본어 특화 전처리 옵션
    collate_fn = TextAudioSpeakerCollate(use_jp_extra=True)

    # 샘플러: 배치 내 길이 분포를 조절해 패딩 낭비를 줄이는 커스텀 샘플러를 기본 사용
    # - DistributedBucketSampler: 길이 기반으로 버킷을 나누어 배치들을 구성 (메모리 효율적)
    # - DistributedLengthGroupedSampler: 트랜스포머 라이브러리의 길이 기반 샘플러 (대안)
    if not args.not_use_custom_batch_sampler:
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # num_workers: 워커 수는 시스템 메모리/CPU에 따라 조정 (메모리 부족 시 낮춰보세요)
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            shuffle=False,
            pin_memory=True,  # CUDA 복사 성능 향상
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            # batch_size=hps.train.batch_size,
            persistent_workers=True,
            # prefetch_factor: 워커당 prefetch 크기 (메모리-성능 tradeoff)
            # prefetch_factor=6,
        )
    else:
        # 이전 버전 호환 샘플러 사용 시 (옵션)
        train_sampler = DistributedLengthGroupedSampler(
            dataset=train_dataset,
            batch_size=hps.train.batch_size,
            num_replicas=n_gpus,
            rank=rank,
            lengths=train_dataset.lengths,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            # メモリ消費量を減らそうとnum_workersを1にしてみる
            # num_workers=min(config.train_ms_config.num_workers, os.cpu_count() // 2),
            num_workers=1,
            # shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_size=hps.train.batch_size,
            persistent_workers=True,
            # これもメモリ消費量を減らそうとしてコメントアウト
            # prefetch_factor=6,
        )
        logger.info("Using DistributedLengthGroupedSampler for training.")
        logger.debug(f"len(train_dataset): {len(train_dataset)}")
        logger.debug(f"len(train_loader): {len(train_loader)}")

    eval_dataset = None
    eval_loader = None
    if rank == 0 and not args.speedup:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    if hps.model.use_noise_scaled_mas is True:
        logger.info("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        logger.info("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    if hps.model.use_duration_discriminator is True:
        logger.info("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        net_dur_disc = None
    if hps.model.use_wavlm_discriminator is True:
        net_wd = WavLMDiscriminator(
            hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
        ).cuda(local_rank)
    else:
        net_wd = None
    if hps.model.use_spk_conditioned_encoder is True:
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        logger.info("Using normal encoder for VITS1")

    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        # hps.model 以下のすべての値を引数に渡す
        use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
        use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
        use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
        use_duration_discriminator=hps.model.use_duration_discriminator,
        use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        p_dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_layers_q=hps.model.n_layers_q,
        use_spectral_norm=hps.model.use_spectral_norm,
        gin_channels=hps.model.gin_channels,
        slm=hps.model.slm,
    ).cuda(local_rank)
    if getattr(hps.train, "freeze_JP_bert", False):
        logger.info("Freezing (JP) bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False
    if getattr(hps.train, "freeze_style", False):
        logger.info("Freezing style encoder !!!")
        for param in net_g.enc_p.style_proj.parameters():
            param.requires_grad = False

    if getattr(hps.train, "freeze_decoder", False):
        logger.info("Freezing decoder !!!")
        for param in net_g.dec.parameters():
            param.requires_grad = False

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    if net_wd is not None:
        optim_wd = torch.optim.AdamW(
            net_wd.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_wd = None
    net_g = DDP(
        net_g,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    net_d = DDP(
        net_d,
        device_ids=[local_rank],
        # bucket_cap_mb=512
    )
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            # bucket_cap_mb=512,
        )
    if net_wd is not None:
        net_wd = DDP(
            net_wd,
            device_ids=[local_rank],
            #  bucket_cap_mb=512
        )

    if utils.is_resuming(model_dir):
        if net_dur_disc is not None:
            try:
                _, _, dur_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                    utils.checkpoints.get_latest_checkpoint_path(
                        model_dir, "DUR_*.pth"
                    ),
                    net_dur_disc,
                    optim_dur_disc,
                    skip_optimizer=hps.train.skip_optimizer,
                )
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
            except:
                if not optim_dur_disc.param_groups[0].get("initial_lr"):
                    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
                print("Initialize dur_disc")
        if net_wd is not None:
            try:
                _, optim_wd, wd_resume_lr, epoch_str = (
                    utils.checkpoints.load_checkpoint(
                        utils.checkpoints.get_latest_checkpoint_path(
                            model_dir, "WD_*.pth"
                        ),
                        net_wd,
                        optim_wd,
                        skip_optimizer=hps.train.skip_optimizer,
                    )
                )
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
            except:
                if not optim_wd.param_groups[0].get("initial_lr"):
                    optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
                logger.info("Initialize wavlm")

        try:
            _, optim_g, g_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth"),
                net_g,
                optim_g,
                skip_optimizer=hps.train.skip_optimizer,
            )
            _, optim_d, d_resume_lr, epoch_str = utils.checkpoints.load_checkpoint(
                utils.checkpoints.get_latest_checkpoint_path(model_dir, "D_*.pth"),
                net_d,
                optim_d,
                skip_optimizer=hps.train.skip_optimizer,
            )
            if not optim_g.param_groups[0].get("initial_lr"):
                optim_g.param_groups[0]["initial_lr"] = g_resume_lr
            if not optim_d.param_groups[0].get("initial_lr"):
                optim_d.param_groups[0]["initial_lr"] = d_resume_lr

            epoch_str = max(epoch_str, 1)
            # global_step = (epoch_str - 1) * len(train_loader)
            global_step = int(
                utils.get_steps(
                    utils.checkpoints.get_latest_checkpoint_path(model_dir, "G_*.pth")
                )
            )
            logger.info(
                f"******************Found the model. Current epoch is {epoch_str}, gloabl step is {global_step}*********************"
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
            epoch_str = 1
            global_step = 0
    else:
        try:
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "G_0.safetensors"), net_g
            )
            _ = utils.safetensors.load_safetensors(
                os.path.join(model_dir, "D_0.safetensors"), net_d
            )
            if net_dur_disc is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "DUR_0.safetensors"), net_dur_disc
                )
            if net_wd is not None:
                _ = utils.safetensors.load_safetensors(
                    os.path.join(model_dir, "WD_0.safetensors"), net_wd
                )
            logger.info("Loaded the pretrained models.")
        except Exception as e:
            logger.warning(e)
            logger.warning(
                "It seems that you are not using the pretrained models, so we will train from scratch."
            )
        finally:
            epoch_str = 1
            global_step = 0

    def lr_lambda(epoch):
        """
        Learning rate scheduler for warmup and exponential decay.
        - During the warmup period, the learning rate increases linearly.
        - After the warmup period, the learning rate decreases exponentially.
        """
        if epoch < hps.train.warmup_epochs:
            return float(epoch) / float(max(1, hps.train.warmup_epochs))
        else:
            return hps.train.lr_decay ** (epoch - hps.train.warmup_epochs)

    scheduler_last_epoch = epoch_str - 2
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optim_g, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optim_d, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
    )
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.LambdaLR(
            optim_dur_disc, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
    else:
        scheduler_dur_disc = None
    if net_wd is not None:
        scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
            optim_wd, lr_lambda=lr_lambda, last_epoch=scheduler_last_epoch
        )
        wl = WavLMLoss(
            hps.model.slm.model,
            net_wd,
            hps.data.sampling_rate,
            hps.model.slm.sr,
        ).to(local_rank)
    else:
        scheduler_wd = None
        wl = None
    scaler = GradScaler(enabled=hps.train.bf16_run)
    logger.info("Start training.")

    diff = abs(
        epoch_str * len(train_loader) - (hps.train.epochs + 1) * len(train_loader)
    )
    pbar = None
    if not args.no_progress_bar:
        pbar = tqdm(
            total=global_step + diff,
            initial=global_step,
            smoothing=0.05,
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        )
    initial_step = global_step

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                pbar,
                initial_step,
            )
        else:
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, None],
                None,
                None,
                pbar,
                initial_step,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()
        if net_wd is not None:
            scheduler_wd.step()
        if epoch == hps.train.epochs:
            # Save the final models
            assert optim_g is not None
            utils.checkpoints.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"G_{global_step}.pth"),
            )
            assert optim_d is not None
            utils.checkpoints.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(model_dir, f"D_{global_step}.pth"),
            )
            if net_dur_disc is not None:
                assert optim_dur_disc is not None
                utils.checkpoints.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"DUR_{global_step}.pth"),
                )
            if net_wd is not None:
                assert optim_wd is not None
                utils.checkpoints.save_checkpoint(
                    net_wd,
                    optim_wd,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(model_dir, f"WD_{global_step}.pth"),
                )
            utils.safetensors.save_safetensors(
                net_g,
                epoch,
                os.path.join(
                    config.out_dir,
                    f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                ),
                for_infer=True,
            )
            if hps.repo_id is not None:
                future1 = api.upload_folder(
                    repo_id=hps.repo_id,
                    folder_path=config.dataset_path,
                    path_in_repo=f"Data/{config.model_name}",
                    delete_patterns="*.pth",  # Only keep the latest checkpoint
                    ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
                    run_as_future=True,
                )
                future2 = api.upload_folder(
                    repo_id=hps.repo_id,
                    folder_path=config.out_dir,
                    path_in_repo=f"model_assets/{config.model_name}",
                    run_as_future=True,
                )
                try:
                    future1.result()
                    future2.result()
                except Exception as e:
                    logger.error(e)

    if pbar is not None:
        pbar.close()


def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    pbar: tqdm,
    initial_step: int,
):
    """한 epoch 동안 학습을 수행하고(Generator/Discriminator 업데이트) 필요 시 평가를 수행합니다.

    파라미터 설명:
    - rank, local_rank: 분산환경에서의 전역/로컬 프로세스 인덱스
    - epoch: 현재 epoch 번호
    - hps: HyperParameters 객체 (학습 및 모델 설정 포함)
    - nets: [net_g, net_d, net_dur_disc, net_wd, wl] 모델 리스트 (일부 항목은 None일 수 있음)
    - optims: [optim_g, optim_d, optim_dur_disc, optim_wd]
    - schedulers: [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd]
    - scaler: GradScaler (AMP 사용 시 스케일링을 위해 사용)
    - loaders: [train_loader, eval_loader]
    - logger: 로거 인스턴스
    - writers: (writer, writer_eval) 또는 None
    - pbar: tqdm 인스턴스 (메인 프로세스에서만 사용)
    - initial_step: 학습 시작 시의 global_step (체크포인트에서 복구할 때 사용)

    구현상의 주요 포인트(요약):
    - 입력 배치는 GPU로 전송한 뒤, autocast 컨텍스트 내에서 forward/손실 계산을 수행합니다.
    - Discriminator 먼저 업데이트(진짜/가짜 샘플 판별 손실 역전파), 그 다음 Generator 업데이트를 수행합니다n
    - GradScaler를 사용해 mixed-precision training을 안정적으로 수행합니다.
    - logging은 주기적으로 SummaryWriter에 기록하고, eval 및 체크포인트 저장을 수행합니다.

    Python 문법/패턴 주목할 점:
    - 튜플/리스트 언패킹: `net_g, net_d, ... = nets` 형태로 여러 변수를 한 번에 바인딩합니다.
    - `with autocast(...)` 문: 자동 혼합 정밀도 컨텍스트, 이 안에서 float 연산이 bfloat16 등으로 수행됩니다.
    - `enumerate(train_loader)`는 (index, batch) 쌍을 순회합니다.
    """
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    # 모델을 학습 모드로 전환 (dropout, batchnorm 등의 동작 변경)
    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()
    if net_wd is not None:
        net_wd.train()

    # train_loader는 배치 단위로 다음 튜플을 반환합니다:
    # (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, tone, language, bert, style_vec)
    # - x: 텍스트(토큰 ID) 배치 (LongTensor)
    # - spec: 스펙트로그램(일반적으로 멜/선형)
    # - y: 원시 오디오 파형(또는 오디오 프레임)
    # - speakers/tone/language: 부가 조건(feature)
    # - bert/style_vec: BERT 임베딩 및 스타일 벡터
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        style_vec,
    ) in enumerate(train_loader):
        # MAS (Masked Acoustic Sequence) 노이즈 스케일 업데이트: learning 진행에 따라 노이즈 스케일을 감소시켜 안정화
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            # 음수가 되지 않도록 0.0으로 클램프
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)

        # 데이터/토치 텐서를 GPU로 옮김 (non_blocking=True는 pinned memory 사용 시 성능 향상)
        # 주의: non_blocking=True를 사용하려면 CPU 텐서가 pin_memory 상태여야 합니다.
        x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(
            local_rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(
            local_rank, non_blocking=True
        ), spec_lengths.cuda(local_rank, non_blocking=True)
        y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(
            local_rank, non_blocking=True
        )
        speakers = speakers.cuda(local_rank, non_blocking=True)
        tone = tone.cuda(local_rank, non_blocking=True)
        language = language.cuda(local_rank, non_blocking=True)
        bert = bert.cuda(local_rank, non_blocking=True)
        style_vec = style_vec.cuda(local_rank, non_blocking=True)

        # autocast: 자동 혼합 정밀도(AMP) 컨텍스트. dtype으로 bfloat16 사용 여부는 hps.train.bf16_run으로 제어
        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            # net_g(...)의 반환값은 여러 항목을 포함하는 튜플입니다. 주요 항목 설명:
            # - y_hat: 생성된 오디오(또는 waveform tensor)
            # - l_length: 길이 관련 손실 / 또는 길이 예측 결과
            # - attn: 텍스트-오디오 정렬(attention map)
            # - ids_slice: segment slicing 정보 (어디를 잘라 쓸지에 대한 인덱스)
            # - x_mask, z_mask: 마스크 텐서(패딩 등 처리용)
            # - (z, z_p, m_p, logs_p, m_q, logs_q): 잠재 변수 및 분포 관련 값들
            # - (hidden_x, logw, logw_): encoder/aligner 출력 (duration/discriminator 입력으로 사용)
            # - g: 추가 생성자 내부 정보(예: upsampled feature)
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),  # , logw_sdp),
                g,
            ) = net_g(
                x,
                x_lengths,
                spec,
                spec_lengths,
                speakers,
                tone,
                language,
                bert,
                style_vec,
            )
            # 원래 스펙트로그램(spec)을 멜 스펙트로그램으로 변환
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            # ids_slice를 이용해 미리 정해진 segment(학습 시 사용하는 짧은 길이)로 잘라냄
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            # 생성된 파형(y_hat)을 다시 멜 스펙으로 변환해 mel-based 손실을 계산하기 위함
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # y (원본 파형)도 ids_slice를 사용해 segment 단위로 잘라줍니다.
            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator 판별: 진짜(y) vs 가짜(y_hat.detach())
            # .detach()로 generator의 그래디언트가 discriminator 업데이트 시 전파되지 않도록 함
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # discriminator의 손실을 계산 (AMP 컨텍스트 내에서 계산)
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                # loss_disc_all는 실제로 optimizer.step()에 전달될 전체 discriminator 손실
                loss_disc_all = loss_disc
            # duration discriminator(오디오 길이/시간적 특성 판별자)가 있을 경우 업데이트
            if net_dur_disc is not None:
                # 입력은 generator의 내부 feature (detach로 generator 그래디언트 차단)
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
                with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                    # 현재는 마스크를 고려하지 않고 단순 평균을 사용
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc

                # 옵티마이저 초기화, 역전파, 그래디언트 스케일링/언스케일, 클리핑, 스텝
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                # unscale 이후 clip/inspect 가능
                scaler.unscale_(optim_dur_disc)
                # gradient clipping (값 기반) 및 노름 확인
                grad_norm_dur = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)
            if net_wd is not None:
                # WavLM 기반 판별자 손실 계산 (y와 y_hat은 (batch, 1, time) 형태로 예상)
                # .squeeze(1)로 채널 차원 제거 후 판별자를 적용
                with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                    loss_slm = wl.discriminator(
                        y.detach().squeeze(1), y_hat.detach().squeeze(1)
                    ).mean()

                optim_wd.zero_grad()
                scaler.scale(loss_slm).backward()
                scaler.unscale_(optim_wd)
                # 그래디언트 값 기반 클리핑으로 발산 방지 및 스케일 확인
                grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
                scaler.step(optim_wd)

        # Discriminator 옵티마이저 업데이트: 진짜/가짜 판별 손실로 역전파
        optim_d.zero_grad()
        # GradScaler를 사용해 mixed-precision에서의 역전파 안정성 확보
        scaler.scale(loss_disc_all).backward()
        # optimizer 별로 unscale을 해서 클리핑/디버깅 가능
        scaler.unscale_(optim_d)
        if getattr(hps.train, "bf16_run", False):
            # bf16 실행 중이라면 추가로 글로벌 노름 클리핑 적용
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        # 값 기반의 그래디언트 클리핑 (커스텀 유틸)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            # Generator 업데이트를 위한 손실 계산
            # net_d(y, y_hat)은 discriminator에 대해 (real, fake, feature_real, feature_fake) 등을 반환
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            # duration discriminator가 있으면 generator의 출력에 대해 duration 판별 결과(가짜) 계산
            if net_dur_disc is not None:
                _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            # wavlm 관련 손실(언어모델 기반 혹은 멀티모달 손실)을 사용할 경우 계산
            if net_wd is not None:
                loss_lm = wl(y.detach().squeeze(1), y_hat.squeeze(1)).mean()
                loss_lm_gen = wl.generator(y_hat.squeeze(1))

            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                # Duration loss: 길이 예측치의 합 (여기선 간단히 sum)
                loss_dur = torch.sum(l_length.float())
                # Mel loss: 생성된 멜과 실제 멜의 L1 손실 (스케일로 가중치 조정)
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                # KL loss: 변분 성분(잠재 분포 정규화)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                # Feature matching loss: discriminator의 내부 feature 차이(
                # fmap_r: real feature maps, fmap_g: generated feature maps)
                loss_fm = feature_loss(fmap_r, fmap_g)
                # Generator의 adversarial loss (GAN loss)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # 전체 generator 손실을 합산 (우선순위/가중치는 config에서 제어)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    if net_wd is not None:
                        # wavlm가 있을 경우 duration generator loss와 wavlm 관련 손실을 추가
                        loss_gen_all += loss_dur_gen + loss_lm + loss_lm_gen
                    else:
                        loss_gen_all += loss_dur_gen
        # Generator 파라미터 업데이트
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        # unscale 후 그래디언트 클리핑/검사가 가능
        scaler.unscale_(optim_g)
        # 긴 네트워크의 경우 그래디언트 폭주를 막기 위해 global norm 클리핑 적용
        torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
        # 값 기반 클리핑/로그 목적의 사용자 유틸 호출
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        # optimizer step 수행
        scaler.step(optim_g)
        # scaler 상태(스케일링 계수 등) 업데이트
        scaler.update()

        if rank == 0:
            # 주기적으로 학습 지표를 TensorBoard에 기록
            if global_step % hps.train.log_interval == 0 and not hps.speedup:
                # 현재 learning rate 및 주요 손실들을 모음
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]

                # scalar_dict는 TensorBoard에 기록할 key/value 쌍 모음
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                # dict.update를 사용해 세부 항목을 추가
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/dur": loss_dur,
                        "loss/g/kl": loss_kl,
                    }
                )
                # enumerate와 comprehension으로 리스트형 손실을 key로 풀어 넣음 (예: loss/g/0, loss/g/1, ...)
                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                )

                if net_dur_disc is not None:
                    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})
                    scalar_dict.update(
                        {
                            f"loss/dur_disc_g/{i}": v
                            for i, v in enumerate(losses_dur_disc_g)
                        }
                    )
                    scalar_dict.update(
                        {
                            f"loss/dur_disc_r/{i}": v
                            for i, v in enumerate(losses_dur_disc_r)
                        }
                    )

                    scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
                    scalar_dict.update(
                        {f"loss/g/dur_gen_{i}": v for i, v in enumerate(losses_dur_gen)}
                    )

                if net_wd is not None:
                    scalar_dict.update(
                        {
                            "loss/wd/total": loss_slm,
                            "grad_norm_wd": grad_norm_wd,
                            "loss/g/lm": loss_lm,
                            "loss/g/lm_gen": loss_lm_gen,
                        }
                    )
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # image_dict = {
                #     "slice/mel_org": utils.plot_spectrogram_to_numpy(
                #         y_mel[0].data.cpu().numpy()
                #     ),
                #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                #         y_hat_mel[0].data.cpu().numpy()
                #     ),
                #     "all/mel": utils.plot_spectrogram_to_numpy(
                #         mel[0].data.cpu().numpy()
                #     ),
                #     "all/attn": utils.plot_alignment_to_numpy(
                #         attn[0, 0].data.cpu().numpy()
                #     ),
                # }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    # images=image_dict,
                    scalars=scalar_dict,
                )

            # 주기적 평가 및 체크포인트 저장
            if (
                global_step % hps.train.eval_interval == 0
                and global_step != 0
                and initial_step != global_step
            ):
                # eval_interval에 도달하면 평가 및 체크포인트 저장(단 speedup 모드일 때는 평가 생략)
                if not hps.speedup:
                    evaluate(hps, net_g, eval_loader, writer_eval)

                # 체크포인트 저장(optimizer 상태 포함). 파일명에 global_step을 포함하여 추적 가능하게 함
                utils.checkpoints.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                )
                utils.checkpoints.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth"),
                )
                if net_dur_disc is not None:
                    utils.checkpoints.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"DUR_{global_step}.pth"),
                    )
                if net_wd is not None:
                    utils.checkpoints.save_checkpoint(
                        net_wd,
                        optim_wd,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, f"WD_{global_step}.pth"),
                    )

                # 오래된 체크포인트 정리 (keep_ckpts > 0 일 때만)
                keep_ckpts = config.train_ms_config.keep_ckpts
                if keep_ckpts > 0:
                    utils.checkpoints.clean_checkpoints(
                        model_dir_path=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

                # inference 용으로 safetensors 포맷으로도 저장
                utils.safetensors.save_safetensors(
                    net_g,
                    epoch,
                    os.path.join(
                        config.out_dir,
                        f"{config.model_name}_e{epoch}_s{global_step}.safetensors",
                    ),
                    for_infer=True,
                )
                # Huggingface Hub에 업로드(비동기) - 네트워크/권한 문제참조
                if hps.repo_id is not None:
                    api.upload_folder(
                        repo_id=hps.repo_id,
                        folder_path=config.dataset_path,
                        path_in_repo=f"Data/{config.model_name}",
                        delete_patterns="*.pth",  # Only keep the latest checkpoint
                        ignore_patterns=f"{config.dataset_path}/raw",  # Ignore raw data
                        run_as_future=True,
                    )
                    api.upload_folder(
                        repo_id=hps.repo_id,
                        folder_path=config.out_dir,
                        path_in_repo=f"model_assets/{config.model_name}",
                        run_as_future=True,
                    )

        # global_step는 전체 학습 동안의 step(배치 단위) 카운트
        global_step += 1
        if pbar is not None:
            # pbar description은 현재 epoch 진행률을 간단히 보여주기 위한 문자열
            pbar.set_description(
                f"Epoch {epoch}({100.0 * batch_idx / len(train_loader):.0f}%)/{hps.train.epochs}"
            )
            pbar.update()

    # 에폭 종료 후 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    if pbar is None and rank == 0:
        logger.info(f"====> Epoch: {epoch}, step: {global_step}")


def evaluate(hps, generator, eval_loader, writer_eval):
    """평가 루프: generator를 evaluation 모드로 전환 후 검증 데이터에서 합성 결과(오디오)를 생성하여
    TensorBoard에 오디오/스칼라를 기록합니다.

    주의: 평가 시에는 gradient 계산이 필요 없으므로 `torch.no_grad()` 컨텍스트를 사용합니다.
    """
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print()
    logger.info("Evaluating ...")
    # no_grad: 메모리 사용량 감소 및 계산 속도 향상
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speakers,
            tone,
            language,
            bert,
            style_vec,
        ) in enumerate(eval_loader):
            # GPU로 텐서 이동 (평가 시 devicemapping은 간단히 .cuda() 호출)
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            speakers = speakers.cuda()
            bert = bert.cuda()
            tone = tone.cuda()
            language = language.cuda()
            style_vec = style_vec.cuda()

            # sdp(Flash-SDP) 사용 여부에 따른 추론을 비교하기 위해 두 설정(사용/미사용)으로 infer 수행
            for use_sdp in [True, False]:
                # generator.module.infer: generator가 DDP로 래핑되어 있으므로 module을 통해 실제 모델 접근
                # y=spec을 전달하면 condition된(teacher forced) 합성을 할 수 있음
                y_hat, attn, mask, *_ = generator.module.infer(
                    x,
                    x_lengths,
                    speakers,
                    tone,
                    language,
                    bert,
                    style_vec,
                    y=spec,
                    max_len=1000,
                    sdp_ratio=0.0 if not use_sdp else 1.0,
                )
                # mask의 합으로 실제 샘플 길이를 복원하고 hop_length를 곱해 파형 길이(샘플 수)로 환산
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
                # 以降のログは計算が重い気がするし誰も見てない気がするのでコメントアウト
                # mel = spec_to_mel_torch(
                #     spec,
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # y_hat_mel = mel_spectrogram_torch(
                #     y_hat.squeeze(1).float(),
                #     hps.data.filter_length,
                #     hps.data.n_mel_channels,
                #     hps.data.sampling_rate,
                #     hps.data.hop_length,
                #     hps.data.win_length,
                #     hps.data.mel_fmin,
                #     hps.data.mel_fmax,
                # )
                # image_dict.update(
                #     {
                #         f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             y_hat_mel[0].cpu().numpy()
                #         )
                #     }
                # )
                # image_dict.update(
                #     {
                #         f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                #             mel[0].cpu().numpy()
                #         )
                #     }
                # )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
                            0, :, : y_hat_lengths[0]
                        ]
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    # 평가 결과를 TensorBoard에 기록 (이미지/오디오)
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    # 평가가 끝난 후에는 다시 학습 모드로 전환
    generator.train()


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때 run()을 호출합니다. (모듈로 import 시 실행되지 않음)
    run()
