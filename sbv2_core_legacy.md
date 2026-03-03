# sbv2_core 레거시 학습/사용 가이드

이 문서는 `models/sbv2_core` 경로를 사용하는 레거시 파이프라인만 정리한다.
신규 런타임/워커는 `models/Style-BERT-VITS2` 기준을 사용한다.

## 1) 작업 디렉터리
```bash
cd models/sbv2_core
```

## 2) 의존성 설치

기본(토치 제외):
```bash
pip install -r requirements.txt
```

CPU/레거시 호환(torch 2.1.x 고정):
```bash
pip install -r requirements_cpu.txt
```

RTX 50xx 계열(torch nightly cu128 예시):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

## 3) 초기화(레거시)
```bash
uv run initialize.py
```

## 4) 전처리(레거시)
```bash
uv run --active preprocess_text.py \
  --transcription-path /mnt/d/my_tts_dataset/mai/esd.list \
  --correct_path \
  --use_jp_extra
```

```
uv run --active bert_gen.py -c configs/config_jp_extra.json
```

```
uv run --active style_gen.py -c configs/config_jp_extra.json
```

## 5) 학습(레거시)
JP-Extra 학습 예시:
```bash
torchrun --nproc_per_node=1 train_ms_jp_extra.py \
  -c configs/config_jp_extra.json \
  -m /path/to/tts_dataset/mai \
  --skip_default_style
```

## 6) PyTorch 추론 테스트(레거시)
스크립트: `server_editor.py` 또는 `inference.py`
```bash
uv run run_infer.py
```

## 7) ONNX 변환(레거시)
TTS 본체 ONNX:
```bash
uv run convert_onnx.py --model model_assets/tts/mai/mai_e281_s263000.safetensors
```

BERT ONNX:
```bash
uv run convert_bert_onnx.py --language JP
```

## 8) ONNX 전용 독립 런타임(sbv_runtime)
디렉터리: `models/sbv2_core/sbv_runtime`

설치(토치 없이 ONNX만):
```bash
pip install -r models/sbv2_core/sbv_runtime/requirements_runtime_onnx.txt
```

실행 예시:
```bash
uv run python -m sbv_runtime \
  --model_onnx model_assets/tts/mai/mai_e281_s263000.onnx \
  --config model_assets/tts/mai/config.json \
  --style_vectors model_assets/tts/mai/style_vectors.npy \
  --bert_onnx_dir bert/deberta-v2-large-japanese-char-wwm \
  --text "今日はちょっと寒いな。" \
  --out_wav outputs/mai_test.wav \
  --speaker_name mai \
  --device cpu
```

## 9) 레거시 시스템 연동 흐름
1. `system/llm_engine.py`에서 RP 응답 생성
2. `system/rp_parser.py`로 서술/대사 분리
3. `system/translator.py`로 KO->JA 번역
4. `system/tts_worker_client.py`가 `models/sbv2_core/sbv2_worker.py`와 통신해 WAV 생성
5. 감정 JSON(one-hot) 판정 후 UI 이미지 선택(웹)
6. `ffplay`로 재생(CLI)

필수:
- `ffplay` 설치 필요 (`ffmpeg` 패키지)
- `system/llm_engine.py`, `system/translator.py`, `models/sbv2_core/sbv2_worker.py`의 모델 경로가 현재 로컬 환경과 맞아야 함
- TTS 체크포인트/ONNX: `models/sbv2_core/model_assets/tts/<voice>/`
- TTS 테스트 WAV: `models/sbv2_core/outputs/` 또는 실행 시 지정 경로
