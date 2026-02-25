# TTS 롤플레잉 모델 연구소

TTS 롤플레잉 모델 연구소는 다음 3개 축으로 구성된 실험 저장소입니다.

1. `data`: 멀티턴/싱글턴/번역/GRPO 학습 데이터 생성
2. `models`: RP 모델, 번역 모델, 일본어 TTS 모델 학습 및 추론 테스트
3. `system`: 위 모델들을 연결해 CLI 텍스트+음성 RP 플레이

## 0) 빠른 시작

### 0-1. 환경
- Python: `>=3.10,<3.12`
- 권장: `uv`

```bash
cd saya_char_qwen2.5
uv sync
```

의존성 정책:
- 루트 `pyproject.toml`/`requirements.txt`에서는 `torch`를 기본 강제하지 않습니다.
- GPU(특히 RTX 50xx)는 환경별로 torch 빌드가 달라 별도 설치를 전제로 합니다.

### 0-2. OpenAI API를 쓰는 파이프라인만 추가 준비
- GPT 기반 데이터 생성/파이프라인은 `OPENAI_API_KEY` 필요
- 루트 `.env` 또는 환경변수 설정

예시 `.env`:
```env
OPENAI_API_KEY=...
```

## 1) 디렉터리 구조

```text
saya_char_qwen2.5/
  data/
    singleturn/     # 싱글턴 생성/정규화/변환/번역SFT셋 생성
    original/       # v5 멀티턴 생성 파이프라인
    version_2/      # v6 멀티턴 생성 파이프라인
    version_3/      # v7 멀티턴 생성 파이프라인
    grpog/          # GRPO용 prompt/reference 데이터셋 생성
  models/
    qwen3_core/     # RP/번역 모델 학습 및 추론
    sbv2_core/      # 일본어 TTS 학습/추론/ONNX 변환
    qwen3_reference/ # 모델 구조/동작 학습용 실험 코드
    sbv2_reference/  # 모델 구조/동작 학습용 실험 코드
  system/           # LLM+번역+TTS 통합 런타임 모듈
  main_loop.py      # CLI RP 플레이 엔트리
```

## 2) Data 파이프라인

## 2-1. 싱글턴 RP 데이터 생성

### A) 로컬 모델 기반 생성
스크립트: `data/singleturn/data_generator_local.py`

```bash
uv run data/singleturn/data_generator_local.py \
  --model_path data/generator/Tri-7B \
  --out_path "$DATA_ROOT/singleturn/rp_generated_local.jsonl" \
  --progress_path "$DATA_ROOT/singleturn/rp_generated_local.progress.json" \
  --target_rows 7000
```

주요 옵션:
- `--temperature`, `--top_p`, `--top_k`, `--max_new_tokens`
- `--no_4bit` (기본은 4bit 경로)

### B) GPT 기반 생성
스크립트: `data/singleturn/data_generator_gpt.py`

```bash
uv run data/singleturn/data_generator_gpt.py \
  --model gpt-4.1-mini \
  --target_rows 12000 \
  --out_path "$DATA_ROOT/singleturn/rp_generated.jsonl" \
  --progress_path "$DATA_ROOT/singleturn/rp_generated.progress.json"
```

출력 스키마(핵심):
- `{system, user, assistant_raw, scene}`

## 2-2. 싱글턴 후처리 (messages 포맷 + 정규화)

### A) messages 포맷 변환
스크립트: `data/singleturn/singleturn_message_converter.py`

```bash
uv run data/singleturn/singleturn_message_converter.py \
  --in_path "$DATA_ROOT/singleturn/rp_generated_local.jsonl" \
  --out_path "$DATA_ROOT/singleturn/rp_generated_local_cleaned.jsonl"
```

### B) 대사/치환 정규화
스크립트: `data/singleturn/singleturn_normalize.py`

```bash
uv run data/singleturn/singleturn_normalize.py \
  --in_jsonl "$DATA_ROOT/singleturn/rp_generated_local_cleaned.jsonl" \
  --out_jsonl "$DATA_ROOT/singleturn/rp_singleturn_cleaned.jsonl" \
  --seed 42
```

정규화 규칙(요약):
- `{{user}}` 치환
- `*` 제거
- assistant 화자 라벨 제거
- assistant 대사 큰따옴표 보정

## 2-3. 멀티턴 데이터 생성 

멀티턴은 3개 버전이 공존합니다.

1. `original` = v5
2. `version_2` = v6
3. `version_3` = v7

모든 파이프라인은 공통적으로 아래 특징이 있습니다.
- `scenario_out`와 `multiturn_out`를 분리 저장
- 파일 라인 수 기준 resume 지원
- Qwen 로컬 백엔드와 GPT 백엔드 둘 다 제공

### A) original (v5)

로컬 Qwen:
```bash
uv run data/original/v5_qwen/pipeline.py \
  --model_path data/generator/Tri-7B \
  --scenario_out "$DATA_ROOT/v5/rp_scenario.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v5/rp_datum.jsonl" \
  --fsm_path data/original/v5_qwen/state_fsm.yaml \
  --action_fsm_path data/original/v5_qwen/action_fsm.yaml \
  --turns 6 \
  --use_4bit
```

GPT:
```bash
uv run data/original/v5_gpt/pipeline.py \
  --openai_model gpt-5-mini \
  --scenario_out "$DATA_ROOT/v5/rp_scenario_gpt.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v5/rp_datum_gpt.jsonl" \
  --fsm_path data/original/v5_qwen/state_fsm.yaml \
  --action_fsm_path data/original/v5_qwen/action_fsm.yaml \
  --turns 6
```

### B) version_2 (v6)

로컬 Qwen:
```bash
uv run data/version_2/v6_qwen/pipeline.py \
  --model_path data/generator/Tri-7B \
  --scenario_out "$DATA_ROOT/v6/v2_scenario.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v6/v2_datum.jsonl" \
  --fsm_path data/version_2/v6_qwen/state_fsm.yaml \
  --action_fsm_path data/version_2/v6_qwen/action_fsm.yaml \
  --turns 8 \
  --use_4bit
```

GPT:
```bash
uv run data/version_2/v6_gpt/pipeline.py \
  --openai_model gpt-5-mini \
  --scenario_out "$DATA_ROOT/v6/rp_scenario_gpt.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v6/rp_datum_gpt.jsonl" \
  --fsm_path data/version_2/v6_qwen/state_fsm.yaml \
  --action_fsm_path data/version_2/v6_qwen/action_fsm.yaml \
  --turns 8
```

### C) version_3 (v7)

로컬 Qwen:
```bash
uv run data/version_3/v7_qwen/pipeline.py \
  --model_path data/generator/Tri-7B \
  --scenario_out "$DATA_ROOT/v7/v3_scenario.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v7/v3_datum.jsonl" \
  --fsm_path data/version_3/v7_qwen/state_fsm.yaml \
  --action_fsm_path data/version_3/v7_qwen/action_fsm.yaml \
  --turns 8 \
  --use_4bit
```

GPT:
```bash
uv run data/version_3/v7_gpt/pipeline.py \
  --openai_model gpt-5-mini \
  --scenario_out "$DATA_ROOT/v7/rp_scenario_gpt.jsonl" \
  --samples 5000 \
  --multiturn_out "$DATA_ROOT/v7/rp_datum_gpt.jsonl" \
  --fsm_path data/version_3/v7_qwen/state_fsm.yaml \
  --action_fsm_path data/version_3/v7_qwen/action_fsm.yaml \
  --turns 4
```

## 2-4. 번역 SFT 데이터셋 생성

스크립트: `data/singleturn/jsonl_to_translation_sft.py`

현재 이 파일은 상단 상수(`INPUT_JSONL_PATH`, `OUTPUT_JSONL_PATH`)를 직접 사용합니다.
필요시 파일 상단 경로를 원하는 경로로 수정 후 실행하세요.

```bash
uv run data/singleturn/jsonl_to_translation_sft.py
```

출력 스키마:
- `{instruction, input, output}`

## 2-5. GRPO 학습 데이터셋 생성

스크립트: `data/grpog/build_grpo_dataset.py`

```bash
uv run data/grpog/build_grpo_dataset.py \
  --inputs "$DATA_ROOT/singleturn/rp_singleturn_cleaned.jsonl" "$DATA_ROOT/v7/rp_datum_unite_cleaned.jsonl" \
  --out_train "$DATA_ROOT/grpo/grpo_train.jsonl" \
  --out_eval "$DATA_ROOT/grpo/grpo_eval.jsonl" \
  --eval_ratio 0.05 \
  --seed 42 \
  --max_context_messages 12 \
  --min_prompt_chars 8 \
  --min_reference_chars 4 \
  --tokenizer_path models/qwen3_core/model_assets/qwen3_4b_rp \
  --max_prompt_tokens 1024 \
  --max_reference_tokens 220
```

## 3) Models 파이프라인

## 3-1. Qwen RP 모델

## A) 베이스 모델 다운로드
스크립트: `models/qwen3_core/initialize.py`

```bash
uv run models/qwen3_core/initialize.py
```

현재 `initialize.py`의 `MODELS`에 정의된 모델만 다운로드됩니다.

## B) SFT (QLoRA) Stage 1/2
스크립트: `models/qwen3_core/sft_trainer_qlora.py`

Stage 1 (singleturn):
```bash
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b-instruct \
  --data_path "$DATA_ROOT/singleturn/rp_singleturn_cleaned.jsonl" \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage1 \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 8 \
  --learning_rate 2e-5 \
  --assistant_only_loss
```

Stage 2 (multiturn):
```bash
uv run models/qwen3_core/sft_trainer_qlora.py \
  --model_name models/qwen3_core/model_assets/qwen3-4b-instruct \
  --data_path "$DATA_ROOT/v7/rp_datum_unite_cleaned.jsonl" \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage2 \
  --init_adapter_path models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage1/lora_adapter \
  --load_in_4bit \
  --bf16 \
  --gradient_checkpointing \
  --max_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --assistant_only_loss
```

## C) LoRA 병합
스크립트: `models/qwen3_core/merge.py`

```bash
uv run models/qwen3_core/merge.py \
  --base_model models/qwen3_core/model_assets/qwen3-4b-instruct \
  --adapter_path models/qwen3_core/model_assets/qwen3_4b_rp_lora_stage2/lora_adapter \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp \
  --dtype bf16 \
  --device_map auto \
  --safe_serialization \
  --trust_remote_code
```

## D) GRPO 학습
스크립트: `models/qwen3_core/grpo_trainer.py`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run models/qwen3_core/grpo_trainer.py \
  --model_name models/qwen3_core/model_assets/qwen3_4b_rp \
  --train_data "$DATA_ROOT/grpo/grpo_train.jsonl" \
  --eval_data "$DATA_ROOT/grpo/grpo_eval.jsonl" \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_grpo \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-6 \
  --max_prompt_length 1024 \
  --max_completion_length 220 \
  --num_generations 4 \
  --bf16 \
  --use_lora \
  --load_in_4bit
```

## E) RP 추론 테스트
- `models/qwen3_core/infer_singleturn_merged.py`
- `models/qwen3_core/infer_singleturn_lora.py`

```bash
uv run models/qwen3_core/infer_singleturn_merged.py
```

(대화형 루프, `exit` 입력 시 종료)

## 3-2. 번역 모델 (KO->JA)

## A) 번역 SFT 학습
스크립트: `models/qwen3_core/sft_trainer_translator.py`

```bash
uv run models/qwen3_core/sft_trainer_translator.py \
  --model_name models/qwen3_core/model_assets/qwen3-1.7b-base \
  --data_path "$DATA_ROOT/singleturn/ko-ja_translation_sft.jsonl" \
  --output_dir models/qwen3_core/model_assets/qwen3_1.7_ko2ja_lora \
  --bf16 \
  --gradient_checkpointing
```

## B) 번역 추론 테스트 (LoRA)
스크립트: `models/qwen3_core/infer_translate_lora.py`

```bash
uv run models/qwen3_core/infer_translate_lora.py \
  --base_dir models/qwen3_core/model_assets/qwen3-1.7b-base \
  --lora_dir models/qwen3_core/model_assets/qwen3_1.7_ko2ja_lora/lora_adapter \
  --text "오늘은 좀 피곤해." \
  --instruction "다음 한국어 문장을 자연스러운 일본어로 번역하시오."
```

## 3-3. 일본어 TTS 모델 (Style-Bert-VITS2)

작업 디렉터리는 `models/sbv2_core` 기준입니다.

이 TTS 파이프라인은 `litagin02/Style-Bert-VITS2` 기반으로 구성했습니다.
또한 이 저장소에서는 실사용 편의를 위해 다음을 추가했습니다.
- 프로세스 상주시켜 재로딩 오버헤드를 줄이는 `worker` 런타임 (`sbv2_worker.py`)
- ONNX만으로 실행 가능한 독립형 `sbv_runtime` (`models/sbv2_core/sbv_runtime`)

RTX 5080 환경 메모:
- CUDA/torch 호환성 이슈 때문에 `models/sbv2_core` 내부에서 용도별 가상환경을 분리해 실험했습니다.
- 사용한 구조:
  - `.venv`: 기본 실험 환경
  - `.venv-cpu`: CPU/호환성 검증용
  - `.venv-stylegen`: 스타일 생성/부가 실험용
- 이 전제에 맞춰 `models/sbv2_core/requirements*.txt`와 `models/sbv2_core/pyproject.toml`은 `torch 비강제` 형태로 정리되어 있습니다.

## A) 초기화
```bash
cd models/sbv2_core
uv run initialize.py
```

## B) 전처리
```bash
uv run preprocess_all.py \
  --model_name mai \
  --batch_size 2 \
  --epochs 100 \
  --use_jp_extra
```

## C) 학습
JP-Extra 학습 예시:
```bash
torchrun --nproc_per_node=1 train_ms_jp_extra.py \
  -c configs/config_jp_extra.json \
  -m /path/to/tts_dataset/mai \
  --skip_default_style
```

## D) PyTorch 추론 테스트
스크립트: `run_infer.py`

```bash
uv run run_infer.py
```

## E) ONNX 변환
TTS 본체 ONNX:
```bash
uv run convert_onnx.py --model model_assets/tts/mai/mai_e281_s263000.safetensors
```

BERT ONNX:
```bash
uv run convert_bert_onnx.py --language JP
```

중요 주의사항 (BERT ONNX FP16):
- DeBERTa v2 계열에서 `model_fp16.onnx` 생성 후 타입 불일치(`Cast_output_0`)가 자주 발생합니다.
- 실사용은 `model.onnx`(FP32)로 진행하는 것이 안정적입니다.
- `model_fp16.onnx`가 깨졌다면 삭제하고 FP32만 사용하세요.

## F) ONNX 전용 독립 런타임(sbv_runtime)
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

`--bert_onnx_dir`에는 아래가 같이 있어야 합니다.
- `model.onnx` (또는 정상 동작하는 `model_fp16.onnx`)
- tokenizer 파일들 (`tokenizer.json` 등)

## G) sbv2_core 의존성 설치 가이드 (현재 가상환경 기준)

작업 디렉터리:
```bash
cd models/sbv2_core
```

기본(토치 제외):
```bash
pip install -r requirements.txt
```

CPU/레거시 호환(torch 2.1.x 고정):
```bash
pip install -r requirements_cpu.txt
```

RTX 5080 계열(torch nightly cu128 예시):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

## 4) System 파이프라인 (텍스트+음성 RP CLI)

엔트리: `main_loop.py`

```bash
uv run main_loop.py
```

실행 흐름:
1. `system/llm_engine.py`에서 RP 응답 생성
2. `system/rp_parser.py`로 서술/대사 분리
3. `system/translator.py`로 KO->JA 번역
4. `system/tts_worker_client.py`가 `models/sbv2_core/sbv2_worker.py`와 통신해 WAV 생성
5. `ffplay`로 재생

필수:
- `ffplay` 설치 필요 (`ffmpeg` 패키지)
- `system/llm_engine.py`, `system/translator.py`, `models/sbv2_core/sbv2_worker.py`의 모델 경로가 현재 로컬 환경과 맞아야 함

## 5) 산출물 위치 정리

대표 산출물:
- 싱글턴 생성: `$DATA_ROOT/singleturn/*.jsonl`
- 멀티턴 생성: `$DATA_ROOT/v5/*.jsonl`, `$DATA_ROOT/v6/*.jsonl`, `$DATA_ROOT/v7/*.jsonl`
- GRPO 데이터: `$DATA_ROOT/grpo/*.jsonl`
- RP LoRA/머지 모델: `models/qwen3_core/model_assets/*`
- 번역 LoRA/머지 모델: `models/qwen3_core/model_assets/*`
- TTS 체크포인트/ONNX: `models/sbv2_core/model_assets/tts/<voice>/`
- TTS 테스트 WAV: `models/sbv2_core/outputs/` 또는 실행 시 지정 경로

## 6) 자주 발생하는 문제와 원인

## 6-0. TTS 성능 한계(중요)

현재 구성은 `litagin02/Style-Bert-VITS2` 파이프라인을 기반으로 하지만,
litagin 쪽 pretrained를 그대로 사용하는 방식이 아니기 때문에
동일한 성능(음질/안정성/표현력)이 그대로 나오지 않을 수 있습니다.

## 6-1. `model_fp16.onnx` 로드 실패
에러 예시:
- `Type Error: Type (tensor(float16)) of output arg (...) does not match expected type (tensor(float))`

원인:
- DeBERTa BERT ONNX FP16 변환 시 그래프 타입 규약 깨짐

대응:
- `model_fp16.onnx` 제거
- `model.onnx`(FP32)만 사용

## 6-2. `model_fp16.onnx` 파일이 없다고 뜸
원인:
- 런타임이 FP16 파일을 우선 찾는데 생성 실패/삭제됨

대응:
- BERT ONNX 디렉터리에 `model.onnx`가 있는지 확인
- 런타임 코드/설정이 FP32 fallback 하도록 확인

## 6-3. `word2ph` 길이 오류/토크나이저 불일치
원인:
- 외부 프로젝트로 이동 시 `pyopenjtalk`, `transformers`, tokenizer/BERT 파일 버전 차이

대응:
- `sbv_runtime/requirements_runtime_onnx.txt` 버전 고정 사용
- BERT tokenizer 파일과 ONNX 모델을 항상 같은 소스에서 함께 배포

## 6-4. CUDA provider 미검출
에러/경고 예시:
- `CUDAExecutionProvider is not in available provider names`

원인:
- `onnxruntime-gpu` 미설치 또는 CUDA 런타임 불일치

대응:
- CPU 실행으로 먼저 검증
- CUDA 사용 시 `onnxruntime-gpu` + 드라이버/CUDA 호환성 점검

## 6-5. RTX 50xx + Torch 커널 오류
에러 예시:
- `no kernel image is available for execution on the device`

원인:
- 설치된 torch 빌드가 GPU 아키텍처(sm_120)를 지원하지 않음

대응:
- 사용하려는 torch 빌드를 환경에서 명시 고정
- `uv` 동기화 시 재설치가 일어나지 않도록 lock/의존성 정책 점검

## 7) 권장 운영 순서 (실험용)

1. `data`: singleturn/multiturn/translation/grpo 데이터 준비
2. `models/qwen3_core`: RP SFT -> merge -> (선택) GRPO
3. `models/qwen3_core`: 번역 SFT -> infer 검증
4. `models/sbv2_core`: TTS 학습/추론 -> (선택) ONNX 변환
5. `main_loop.py` 또는 `sbv_runtime`로 통합 테스트
