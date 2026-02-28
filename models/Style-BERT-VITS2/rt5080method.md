# RTX 5080 / WSL2 Style-BERT-VITS2 설치 및 운영 재현 가이드

이 문서는 현재 `pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2`를 WSL2 + NVIDIA GPU 환경에서 다시 설치하고, `app.py`, `server_editor.py`, `transcribe.py`를 실행할 때 필요한 실제 수정 사항과 운영 절차를 재현 가능하게 정리한 문서다.

기준 경로:

- 프로젝트 루트: `/home/ahnhs2k/pytorch-demo`
- 대상 프로젝트: `/home/ahnhs2k/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2`

이 문서는 다음 문제들을 실제로 겪은 뒤 정리한 것이다.

- `uv run`이 활성 `venv`를 무시하고 `.venv`를 새로 만들며 의존성을 다시 해석하는 문제
- `onnxruntime==1.24.2`가 Python 3.10 휠이 없어 설치 실패하는 문제
- `faster-whisper==0.10.1`이 `av==10.0.0`을 끌고 와 Cython 3 / FFmpeg 헤더와 충돌하는 문제
- RTX 5080(`sm_120`)이 구버전 `torch`에서 지원되지 않아 CUDA 전사가 실패하는 문제
- `uv lock`이 PyTorch nightly `cu128`의 `triton==3.6.0+git...` 의존성을 해상하지 못하는 문제
- `pyproject.toml`에 실제 런타임 의존성이 누락되어 `uv run` 시 다수의 `ModuleNotFoundError`가 나는 문제

## 1. 권장 전제 조건

권장 환경은 다음과 같다.

- OS: Windows 11 + WSL2 Ubuntu
- Python: 3.10.x
- CUDA GPU: RTX 5080
- 가상환경: `Style-BERT-VITS2/venv`

중요한 점은 이 프로젝트는 `uv` 기반 메타데이터를 가지고 있지만, RTX 5080 환경에서는 `torch` 계열만 별도로 관리하는 편이 안전하다는 점이다.

운영 원칙:

- 일반 Python 의존성은 `uv`로 관리한다.
- `torch`, `torchaudio`, `torchvision`은 RTX 5080 지원을 위해 `pip install --pre ... --index-url https://download.pytorch.org/whl/nightly/cu128`로 별도 관리한다.
- 따라서 실행 시 `uv run --active --no-sync` 또는 아예 `python` 직접 실행을 써야 한다.

## 2. WSL2 메모리 상향

WSL 기본 메모리로는 모델 병합, 대형 패키지 설치, 캐시 생성 시 불안정할 수 있다. Windows 사용자 홈에 `.wslconfig`를 만들어 메모리를 늘린다.

파일 위치:

- Windows: `C:\Users\<계정명>\.wslconfig`

예시:

```ini
[wsl2]
memory=24GB
swap=16GB
processors=12
```

적용:

```powershell
wsl --shutdown
```

WSL 재진입 후 확인:

```bash
free -h
```

정상 반영 예:

- `Mem: 23Gi`
- `Swap: 16Gi`

## 3. 현재 적용된 핵심 수정

이 프로젝트는 원본 상태로 두면 최신 패키지를 끌어오면서 깨진다. 따라서 아래 수정이 반영되어 있어야 한다.

### 3.1 `requirements.txt`

파일:

- `saya_char_qwen2.5/models/Style-BERT-VITS2/requirements.txt`

현재 핵심 수정:

```txt
Cython<3
setuptools<81
wheel

accelerate
av==12.3.0
cmudict
cn2an
faster-whisper>=1.0.0
```

설명:

- `Cython<3`: 구버전 PyAV 소스 빌드가 Cython 3에서 깨지는 문제를 피한다.
- `setuptools<81`: 일부 레거시 빌드 경고 및 호환성 문제를 줄인다.
- `av==12.3.0`: `av==10.0.0`은 너무 오래되어 최신 FFmpeg 헤더와 충돌했고, 너무 최신 `av`는 반대로 시스템 FFmpeg가 따라오지 못했다. 중간 버전으로 고정한다.
- `faster-whisper>=1.0.0`: `0.10.1`이 `av==10.0.0`을 강하게 끌고 와서 설치가 깨진다.

### 3.2 `pyproject.toml`

파일:

- `saya_char_qwen2.5/models/Style-BERT-VITS2/pyproject.toml`

현재 핵심 수정:

```toml
requires-python = ">=3.10,<3.11"

dependencies = [
    ...
    "accelerate",
    "fastapi",
    "faster-whisper>=1.0.0",
    "gradio>=4.32",
    "matplotlib",
    "onnxruntime<1.24",
    "requests",
    "scikit-learn",
    "scipy",
    "setuptools<81",
    "soundfile",
    "soxr",
    ...
    "transformers<4.55",
    "umap-learn",
    "uvicorn",
]
```

그리고 ONNX 테스트용 환경도 동일하게 상한을 둬야 한다.

```toml
"onnxruntime-directml<1.24; sys_platform == 'win32'"
"onnxruntime-gpu<1.24; sys_platform != 'darwin'"
```

설명:

- `requires-python = ">=3.10,<3.11"`: 현재 RTX 5080 nightly PyTorch 운영은 Python 3.10 기준으로 맞춘다. `uv lock`이 3.11+ 조합까지 해상하려고 들면 nightly `triton` 때문에 실패한다.
- 최신 `onnxruntime==1.24.2`는 Python 3.10용 wheel이 없다.
- Python 3.10 기반 운영을 유지하려면 `onnxruntime<1.24`로 제한해야 한다.
- `transformers<4.55`: 현재 프로젝트의 로컬 BERT `.bin` 파일을 로드할 때 `torch.load` 보안 체크와 충돌하는 더 최신 버전을 피한다.
- `accelerate`, `fastapi`, `uvicorn`, `gradio`, `matplotlib`, `scikit-learn`, `umap-learn`, `soundfile`, `soxr`, `faster-whisper` 등은 실제 실행 경로에서 필요하지만 원래 `pyproject.toml`에 빠져 있었던 항목들이다.

## 4. 초기 설치 절차

`Style-BERT-VITS2` 디렉터리로 이동한다.

```bash
cd ~/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2
```

가상환경이 없다면 먼저 만든다.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -V
```

### 4.1 빌드 툴 선설치

```bash
uv pip install "Cython<3" "setuptools<81" wheel
```

이 단계는 `av` 소스 빌드를 위해 필요하다.

### 4.2 FFmpeg 개발 헤더 설치

PyAV는 시스템 FFmpeg 헤더를 사용해 빌드한다.

```bash
sudo apt update
sudo apt install -y \
  ffmpeg \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libavfilter-dev \
  libswscale-dev \
  libswresample-dev
```

### 4.3 일반 Python 패키지 설치

중요: build isolation을 끄고 설치해야 한다.

```bash
uv pip install --no-build-isolation -r requirements.txt
```

이유:

- build isolation을 켜면 `uv`가 임시 빌드 환경에서 최신 Cython을 다시 끌고 올 수 있다.
- 현재는 로컬 `venv` 안의 `Cython<3`를 실제 빌드에 써야 한다.

### 4.4 RTX 5080용 PyTorch nightly 설치

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

확인:

```bash
python -c "import torch, torchaudio, torchvision; print(torch.__version__); print(torchaudio.__version__); print(torchvision.__version__)"
```

예시:

- `torch 2.12.0.dev...+cu128`
- `torchaudio 2.11.0.dev...+cu128`
- `torchvision 0.26.0.dev...+cu128`

## 5. `torch` 계열 운영 원칙

이 프로젝트는 RTX 5080 때문에 `torch` 계열을 nightly로 써야 한다. 하지만 `uv lock`은 현재 이 nightly 조합의 `triton==3.6.0+git...` 의존성을 제대로 해상하지 못한다.

따라서 다음 원칙으로 운영한다.

- `torch`, `torchaudio`, `torchvision`은 `pyproject.toml`에서 직접 관리하지 않는다.
- 이 세 패키지는 활성 `venv`에서 `pip install --pre ... --index-url ...`로 설치한다.
- 이후 `uv sync`는 `--active --inexact`로 실행하거나, 실행 자체는 `uv run --active --no-sync ...`로 한다.
- `uv`가 환경을 다시 맞추면서 nightly `torch`를 건드리지 않게 하는 것이 핵심이다.

주의:

- `uv lock`에 `torch` nightly를 직접 넣어 관리하려고 하면 `triton` 때문에 해상이 실패한다.
- `uv run --active`를 `--no-sync` 없이 쓰면, 환경을 재정렬하면서 수동 설치한 nightly 조합이 어긋날 수 있다.

## 6. `uv run` 사용 시 주의점

이 프로젝트에서 `uv run`은 그대로 쓰면 자주 실패한다.

### 6.1 왜 실패하는가

예를 들어 `venv`를 활성화해도, 아래처럼 실행하면:

```bash
uv run server_editor.py --inbrowser
```

`uv`는 다음과 같이 동작할 수 있다.

- 활성 `venv`를 무시
- 프로젝트용 `.venv`를 새로 생성
- `pyproject.toml`/`uv.lock` 기반으로 의존성 재해석
- 최신 `onnxruntime` 또는 잠긴 `uv.lock` 버전으로 다시 실패
- 수동 설치한 PyTorch nightly 조합을 망가뜨릴 수 있음

### 6.2 안전한 실행 방식

현재 활성 `venv`를 그대로 사용하고, 재동기화를 건너뛰려면:

```bash
uv run --active --no-sync server_editor.py --inbrowser
```

앱과 전사도 동일하다.

```bash
uv run --active --no-sync app.py
uv run --active --no-sync transcribe.py ...
```

또는 가장 단순하게:

```bash
python server_editor.py --inbrowser
```

운영 안정성 기준으로는 `python` 직접 실행이 가장 예측 가능하다.

## 7. `uv.lock`과 락 파일 갱신

`pyproject.toml`을 고쳐도 `uv.lock`이 옛 버전을 잡고 있으면, `uv run`/`uv sync`는 그 옛 값을 다시 쓴다.

예시:

- `pyproject.toml`은 `onnxruntime<1.24`로 고쳤는데
- `uv.lock`에는 `onnxruntime 1.24.2`가 남아 있을 수 있다

따라서 일반 의존성에 대해서는 락 파일을 갱신해야 한다.

```bash
uv lock
uv sync --active --inexact
```

그 다음 실행:

```bash
uv run --active --no-sync server_editor.py --inbrowser
```

중요:

- `--inexact`를 쓰는 이유는 `uv`가 선언되지 않은 수동 설치 패키지(`torch`, `torchaudio`, `torchvision`)를 지우지 않게 하기 위해서다.
- `torch` nightly는 `uv lock` 대상이 아니므로, `uv lock`/`uv sync` 후에도 필요하면 다시 `pip install --pre ...`로 덮어쓴다.

## 8. 실행 절차

권장 순서:

```bash
cd ~/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2
source venv/bin/activate
uv lock
uv sync --active --inexact
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
python server_editor.py --inbrowser
```

또는:

```bash
cd ~/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2
source venv/bin/activate
uv run --active --no-sync server_editor.py --inbrowser
```

앱 실행:

```bash
uv run --active --no-sync app.py
```

전사 실행:

```bash
uv run --active --no-sync transcribe.py ...
```

초기 실행 시 다음이 자동 생성될 수 있다.

- `configs/paths.yml`

이 경우 로그로 안내가 뜨며 정상이다.

```text
A configuration file configs/paths.yml has been generated based on the default configuration file default_paths.yml.
```

## 9. 자주 발생하는 오류와 대응

### 9.1 `av==10.0.0` 빌드 실패

증상:

- `faster-whisper==0.10.1`이 `av==10.0.0`을 끌고 옴
- Cython 3와 충돌
- 또는 FFmpeg 헤더와 충돌

대응:

- `requirements.txt`에서
  - `faster-whisper==0.10.1` 제거
  - `faster-whisper>=1.0.0`
  - `av==12.3.0`
  - `Cython<3`

### 9.2 `onnxruntime 1.24.2` wheel 없음

증상:

```text
You're using CPython 3.10 (`cp310`), but `onnxruntime` (v1.24.2) only has wheels ... cp311+
```

대응:

- `pyproject.toml`에서 `onnxruntime<1.24`로 제한
- `uv.lock` 갱신 또는 `--no-sync`로 실행

### 9.3 `PyTorch >= 2.4 is required but found 2.3.1`

증상:

- `transformers`가 torch 백엔드를 꺼버림

대응:

- 구버전 `torch` 대신 RTX 5080 지원 nightly `torch`를 설치
- `transformers`는 `4.55` 미만으로 유지

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install "transformers<4.55"
```

### 9.4 `CUDA error: no kernel image is available for execution on the device`

증상:

- RTX 5080에서 Whisper 전사 시 CUDA 커널 실행 실패
- 로그에 `sm_120 is not compatible with the current PyTorch installation`가 함께 뜸

대응:

- RTX 5080 지원 nightly `torch` 계열로 교체
- `uv`가 아니라 `pip`로 직접 설치

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 9.5 `uv lock`이 `triton==3.6.0+git...` 때문에 실패

증상:

```text
Because there is no version of triton==3.6.0+git... and all versions of torch depend on it ...
```

대응:

- `uv`에서 `torch` 계열을 관리하려고 하지 않는다.
- 일반 의존성만 `uv lock`/`uv sync --active --inexact`로 맞춘다.
- `torch`, `torchaudio`, `torchvision`은 `pip install --pre ...`로 별도 유지한다.

## 10. 권장 재현 시나리오

## 10. 권장 재현 시나리오

가장 재현성이 높은 순서는 아래와 같다.

1. WSL 메모리 상향 적용
2. `Style-BERT-VITS2`로 이동
3. `venv` 생성 및 활성화
4. `Cython<3`, `setuptools<81`, `wheel` 선설치
5. 시스템 FFmpeg dev 패키지 설치
6. 수정된 `requirements.txt`로 `--no-build-isolation` 설치
7. `uv lock` / `uv sync --active --inexact`로 일반 의존성 정리
8. `pip install --pre torch torchvision torchaudio --index-url ...`로 nightly 설치
9. `python ...` 또는 `uv run --active --no-sync ...`로 실행

예시 전체:

```bash
cd ~/pytorch-demo/saya_char_qwen2.5/models/Style-BERT-VITS2

python3.10 -m venv venv
source venv/bin/activate

uv pip install "Cython<3" "setuptools<81" wheel

sudo apt update
sudo apt install -y \
  ffmpeg \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libavfilter-dev \
  libswscale-dev \
  libswresample-dev

uv pip install --no-build-isolation -r requirements.txt

uv lock
uv sync --active --inexact

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

python server_editor.py --inbrowser
```

## 11. 운영 원칙

이 프로젝트는 설치가 한 번 맞아도, 이후 `uv run`, `uv sync`, `uv lock`에 따라 다시 깨질 수 있다. 운영 시에는 아래 원칙을 지키는 편이 안전하다.

- 실행은 `python ...` 또는 `uv run --active --no-sync ...` 중심으로 한다.
- `pyproject.toml`을 바꾸면 일반 의존성 기준으로 `uv.lock`을 갱신한다.
- `torch`, `torchaudio`, `torchvision`은 `uv`가 아니라 `pip`로 유지한다.
- 최신 패키지를 자동으로 받게 두지 않는다.
- 특히 `onnxruntime`, `transformers`, `faster-whisper`, `av`는 무핀 운영을 피한다.
- `uv sync` 후에는 필요하면 nightly `torch`를 다시 덮어쓴다.

## 12. 현재 문서 기준 핵심 파일

- `saya_char_qwen2.5/models/Style-BERT-VITS2/requirements.txt`
- `saya_char_qwen2.5/models/Style-BERT-VITS2/pyproject.toml`
- `saya_char_qwen2.5/models/Style-BERT-VITS2/uv.lock`
- `saya_char_qwen2.5/models/Style-BERT-VITS2/server_editor.py`

이 문서는 현재까지 실제 발생한 오류와 수정 사항을 바탕으로 작성되었으며, 재현성 확보가 목적이다. 새로 버전을 올릴 때는 이 네 파일을 먼저 확인해야 한다.
