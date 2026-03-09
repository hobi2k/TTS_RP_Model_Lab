# FlashAttention2 설치 가이드 (WSL, RTX 5080)

이 문서는 `TTS_RP_Model_Lab` 환경에서 `flash-attn`을 설치할 때 필요한 최소 절차와 장애 대응 방법을 정리한다.

## 1) 전제 조건

- WSL2 + NVIDIA 드라이버가 정상 동작해야 한다.
- Python venv 활성화 상태여야 한다.
- PyTorch CUDA 빌드가 먼저 설치되어 있어야 한다. (예: `torch 2.9.1+cu128`)

확인:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 2) CUDA Toolkit 12.8 설치 (Ubuntu 24.04)

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y build-essential ninja-build cuda-toolkit-12-8
```

환경변수:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

확인:

```bash
nvcc --version
```

## 3) flash-attn 설치 (RTX 5080 권장 설정)

중요: RTX 5080은 `sm_120`만 빌드하도록 제한해야 설치 시간이 줄고 실패 확률이 낮아진다.  
`flash-attn==2.8.3` 기준으로는 `TORCH_CUDA_ARCH_LIST`가 아니라 `FLASH_ATTN_CUDA_ARCHS`를 사용한다.

```bash
cd ~/pytorch-demo/TTS_RP_Model_Lab

uv pip uninstall -y flash-attn
rm -rf ~/.cache/uv/sdists-v9/pypi/flash-attn ~/.cache/pip

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=1
export NVCC_THREADS=1

uv pip install --no-build-isolation --no-cache-dir flash-attn
```

설치 검증:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

## 4) 설치 진행 중인지 확인

```bash
ps -ef | rg -i "uv pip install flash-attn|ninja|nvcc|cicc" | rg -v rg
```

컴파일이 정상 진행 중이면 `cicc` 또는 `nvcc` 프로세스가 뜨고 CPU 사용률이 올라간다.

```bash
top -H -p $(pgrep -d',' -f "ninja|nvcc|cicc|uv pip install flash-attn")
```

## 5) 멈춘 것 같을 때

아래 상태면 사실상 멈춘 경우가 많다.

- `ninja`만 남고 `cicc/nvcc`가 사라짐
- CPU 사용률이 계속 0%

복구:

```bash
pkill -f "flash-attn|ninja|nvcc|cicc"
rm -rf /tmp/.tmp*/sdists-v9/pypi/flash-attn
rm -rf ~/.cache/uv/sdists-v9/pypi/flash-attn
```

그 다음 3) 절차로 다시 설치한다.

## 6) 자주 나는 오류와 의미

- `ModuleNotFoundError: No module named 'flash_attn'`
  - 설치가 완료되지 않았거나 실패했다.

- `CUDA error: no kernel image is available for execution on the device`
  - GPU 아키텍처용 커널이 포함되지 않은 빌드다.
  - `FLASH_ATTN_CUDA_ARCHS="120"`으로 다시 빌드한다.

- `FlashAttention2 has been toggled on ... package flash_attn seems to be not installed`
  - 라이브러리 import 실패 상태다. 설치 자체를 먼저 정상화해야 한다.

## 7) 참고

- `flash-attn` 소스 빌드는 오래 걸릴 수 있다.  
  (`sm_80/sm_90/sm_100/sm_120` 전체 빌드 시 1~3시간 이상 가능)
- RTX 5080은 `FLASH_ATTN_CUDA_ARCHS="120"`으로 고정해 빌드 시간을 줄이는 것을 권장한다.
