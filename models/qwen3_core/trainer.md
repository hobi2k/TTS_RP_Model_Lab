# qwen3_core Trainer Notes

이 문서는 `models/qwen3_core` 아래의 `sft_trainer_*` / `grpo_trainer_*` 계열 스크립트의 현재 구조, 구현 방식, 그리고 CausalLM과 VLM 학습 시 차이를 정리한 문서다.

대상 독자:
- 현재 폴더 안의 trainer 스크립트를 유지보수하는 사람
- CausalLM용 스크립트와 VLM용 스크립트를 혼동하지 않고 수정해야 하는 사람
- `text-only RP 데이터`를 VLM에 태울 때 어떤 우회가 필요한지 확인해야 하는 사람

## 1. 파일 분류

현재 기준으로 실사용/참조 가치가 있는 trainer 계열은 아래다.

### SFT 계열

- [sft_trainer_qlora.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora.py)
  - Qwen/Gemma/Rosetta 같은 **CausalLM** 계열 QLoRA SFT
- [sft_trainer_qlora_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_vlm.py)
  - Qwen3.5 계열 **VLM** QLoRA SFT
- [sft_trainer_qlora_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_kanana.py)
  - Kanana 계열 **VLM** QLoRA SFT

### GRPO 계열

- [grpo_trainer.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer.py)
  - **CausalLM**용 GRPO
- [grpo_trainer_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_vlm.py)
  - Qwen3.5 계열 **VLM**용 GRPO
- [grpo_trainer_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_kanana.py)
  - Kanana 계열 **VLM**용 GRPO

### 레거시/보조 파일

- [sft_trainer_hf.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_hf.py)
- [sft_trainer_translator.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_translator.py)
- [grpo_trainer_old.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_old.py)
- [grpo_trainer_colab.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_colab.py)
- [grpo_trainer.md](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer.md)
  - `grpo_trainer.py`의 보상 설계 중심 메모

이 문서는 위 파일들을 상위 관점에서 통합 설명한다.

## 2. 공통 학습 구조

SFT와 GRPO 모두 큰 흐름은 비슷하다.

1. 모델/토크나이저 또는 프로세서 로드
2. 데이터셋을 프로젝트 표준 포맷으로 정규화
3. LoRA/QLoRA 설정 적용
4. Trainer 또는 GRPOTrainer 구성
5. adapter / processor / tokenizer 저장

하지만 구현 세부는 CausalLM과 VLM에서 크게 갈린다.

## 3. CausalLM과 VLM의 핵심 차이

### 3.1 입력 표현 방식

#### CausalLM

- 입력은 결국 `text -> tokenizer -> input_ids`
- 핵심은 `chat_template`과 `assistant_only_loss` 마스킹
- 이미지 관련 텐서가 없다

#### VLM

- 입력은 `text + images -> processor`
- 텍스트 외에 아래 중 일부가 추가될 수 있다
  - `pixel_values`
  - `image_grid_thw`
  - `mm_token_type_ids`
  - 모델별 image meta
- 같은 배치 안에서 이미지가 있는 샘플과 없는 샘플이 섞이면 processor 제약 때문에 우회가 필요할 수 있다

### 3.2 데이터셋 컬럼

#### CausalLM

보통 아래 컬럼만 신경 쓰면 된다.
- `messages`
- `conversations`
- `prompt`
- `reference`

#### VLM

위 컬럼에 더해 아래가 추가된다.
- `image`
- `image_path`
- `images`

그리고 실제 학습에서 중요한 건 **이미지 필드가 존재하느냐**와 **프로세서가 image slot을 요구하느냐**다.

### 3.3 Trainer 계층

#### CausalLM SFT

- `trl.SFTTrainer` 또는 `transformers.Trainer`
- 대부분 텍스트 템플릿만 잘 맞추면 된다

#### VLM SFT

- 보통 `transformers.Trainer + custom collator`
- 이유:
  - processor가 text+image를 동시에 받아야 함
  - assistant-only loss 마스킹도 직접 제어해야 함

#### CausalLM GRPO

- `trl.GRPOTrainer`
- prompt -> N개 completion 생성 -> reward 계산

#### VLM GRPO

- `trl.GRPOTrainer`를 그대로 쓰기 어렵다
- processor adapter, multimodal prompt 처리, image batch 정합, 모델별 state reset이 필요하다
- 따라서 실제로는 custom wrapper가 거의 필수다

## 4. SFT 계열 구조

## 4.1 [sft_trainer_qlora.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora.py)

용도:
- 일반 CausalLM RP SFT

핵심 구조:
- `build_training_example()`
  - 원본 `messages` / `conversations`를 표준 messages로 정규화
- chat template는 tokenizer/모델이 가진 템플릿을 그대로 사용
- LoRA 대상 모듈은 공통 Qwen 계열 projection 모듈
- `assistant_only_loss`는 라벨 마스킹으로 처리

특징:
- 텍스트만 다루므로 collator가 단순하다
- 문제의 대부분은 chat template 경계, label mask, tokenizer special token 정합성이다

대표 리스크:
- assistant 종료 경계가 label에 안 들어가 과생성 발생
- Gemma/Rosetta 병합 후 tokenizer/chat template 불일치

## 4.2 [sft_trainer_qlora_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_vlm.py)

용도:
- Qwen3.5 계열 VLM SFT

핵심 구조:
- `build_example()`
  - 텍스트 메시지를 VLM용 content block 구조로 변환
  - user 첫 메시지에만 image block 삽입
- `QwenVLMCollator`
  - `processor.apply_chat_template(...)`
  - `processor(text=..., images=..., return_tensors="pt")`
  - labels 생성 및 assistant-only loss 마스킹

중요 동작:
- **이미지 없는 샘플도 배치 안에 실제 이미지 샘플이 하나라도 있으면 더미 이미지가 필요하다**
- 이유:
  - 같은 batch에서 일부만 이미지가 있으면 processor 입력 모양을 통일해야 하기 때문
  - 그래서 `has_any_image`가 True인 배치에서는 이미지 없는 샘플에도 더미 이미지를 넣고 image block을 주입한다

중요한 구분:
- `text-only dataset 전체`를 VLM에 태울 때는 **항상 더미 이미지가 필요한 것은 아니다**
- `sft_trainer_qlora_vlm.py`의 현재 구현은:
  - 배치 전체에 이미지가 하나도 없으면 `images=None`
  - 배치 안에 실제 이미지가 하나라도 있으면 나머지 샘플에 더미 이미지 주입

즉, 이 스크립트에서 더미 이미지는 **배치 혼합 문제 해결용**이다.

## 4.3 [sft_trainer_qlora_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_kanana.py)

용도:
- Kanana 계열 VLM SFT

핵심 구조:
- 첫 user content 문자열에 `"<image>\n..."`를 넣는 Kanana 전용 포맷 사용
- `KananaCollator`가 batch를 processor 전용 형식으로 변환

중요 동작:
- **Kanana는 text-only 학습에서도 더미 이미지 주입이 사실상 필수**다
- 이유:
  - Kanana 인코더는 `<image>` 슬롯과 image meta 개수의 정합성을 엄격하게 본다
  - 텍스트 RP 데이터를 태울 때도 image slot을 유지하려면 dummy image가 필요하다

즉, Kanana 계열 VLM은:
- `text-only RP 데이터`
- `실제 이미지 없음`
이어도 **더미 이미지 + image slot 유지**가 기본 전략이다.

## 5. GRPO 계열 구조

## 5.1 [grpo_trainer.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer.py)

용도:
- CausalLM용 RP GRPO

핵심 구조:
- `load_grpo_dataset()`
  - `prompt` 또는 `messages`를 마지막 user 턴까지 trim
  - `reference`를 reward용 정답 텍스트로 유지
- `GRPOTrainer`
  - same prompt에서 `num_generations`개의 completion 생성
  - reward 함수로 상대적 우위를 계산

현재 보상 축:
- `reward_format`
- `reward_role_split`
- `reward_grounded_to_user`
- `reward_character_consistency`
- `reward_reference_alignment`

중요 실행 제약:
- `global eval batch size % num_generations == 0`
- 예: `num_generations=4`이면 `per_device_eval_batch_size`도 4의 배수로 맞춰야 한다

## 5.2 [grpo_trainer_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_vlm.py)

용도:
- Qwen3.5 계열 VLM용 GRPO

핵심 구조:
- `QwenVLMGRPOProcessor`
  - TRL이 기대하는 processor 인터페이스를 Qwen VLM processor에 맞춘 어댑터
- `load_grpo_dataset()`
  - `prompt`, `reference`, `image` 컬럼 정규화
- `QwenVLMGRPOTrainer`
  - Qwen3.5 VLM의 internal multimodal state를 정리하기 위한 wrapper

### text-only 데이터에서의 핵심 정책

이 스크립트는 **text-only GRPO 데이터셋에서는 `image` 컬럼 자체를 제거**한다.

이유:
- TRL은 `image` 컬럼이 존재하기만 해도 멀티모달 generation 경로를 탄다
- Qwen3.5 VLM은 이 경로에서 `pixel_values`, `image_grid_thw`, `rope_deltas` 같은 상태를 사용한다
- text-only인데 멀티모달 경로를 타면 position 계산이 꼬이기 쉽다

즉 현재 구현에서:
- `text-only dataset` -> `image` 컬럼 제거
- `실제 이미지가 있는 dataset` -> `image` 컬럼 유지

### 더미 이미지에 대한 정확한 정리

이 스크립트는 예전에는 text-only 샘플에도 더미 이미지를 붙이는 방향으로 접근했지만, 현재 구현은 그걸 **기본값으로 사용하지 않는다**.

현재 기준:
- **Qwen3.5 VLM GRPO에서는 text-only 데이터에 더미 이미지를 강제하지 않는다**
- 강제하면 TRL의 multimodal path와 모델의 3D position/rope state가 충돌할 수 있다

즉 Qwen3.5 VLM GRPO에서 dummy image는:
- “항상 필요”가 아니라
- “실제 이미지가 있는 멀티모달 학습에서만 이미지 입력으로 사용”하는 방향이 더 안전하다

### 추가 구현 메모

- `QwenVLMGRPOTrainer`는 text-only forward 전 `rope_deltas`를 비운다
- 이유:
  - Qwen3.5 VLM은 이전 multimodal generation의 stale state가 남으면 다음 text-only logprob 계산에서 3D position mismatch가 날 수 있다

### 운영 리스크

- `flash-linear-attention` / `causal-conv1d`가 없으면 torch fallback으로 매우 무거워진다
- 16GB급 GPU에서는 아래를 거의 강제해야 한다
  - `--load_in_4bit`
  - `--bf16`
  - `--per_device_train_batch_size 1`
  - `--max_prompt_length` 축소

## 5.3 [grpo_trainer_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_kanana.py)

용도:
- Kanana 계열 VLM용 GRPO

핵심 구조:
- `KananaGRPOProcessor`
  - TRL의 multimodal processor 호출을 KananaVProcessor API에 맞게 변환
- dataset loader는 **항상 `image="__dummy__"`**를 넣는다
- 실제 이미지 대신 processor가 더미 PIL 이미지를 생성한다

### 왜 Kanana GRPO는 더미 이미지를 항상 주는가

Kanana는 설계상:
- image slot과 image meta 정합성이 엄격하고
- text-only라도 `<image>`를 전제로 한 포맷을 유지하는 쪽이 안정적이었다

그래서 현재 구현은:
- text-only RP GRPO 데이터라도
- TRL 멀티모달 경로를 강제로 타게 만들고
- processor가 dummy image를 공급한다

즉 Kanana 계열에서는:
- **더미 이미지가 구현 전략의 핵심**
- Qwen3.5 VLM과 다르게 “멀티모달 포맷 유지” 쪽으로 문제를 푼다

## 6. CausalLM vs VLM: 실제 구현 차이 요약

## 6.1 SFT에서의 차이

### CausalLM SFT

- tokenizer 기반
- `messages -> text`
- label mask만 맞추면 됨

### VLM SFT

- processor 기반
- `messages -> content blocks`
- 이미지가 있으면 image block + image tensor
- 모델/processor에 따라 text-only에서도 dummy image가 필요할 수 있음

정확한 현재 정책:
- Qwen3.5 VLM SFT:
  - 혼합 batch에서만 dummy image 사용
- Kanana VLM SFT:
  - text-only에서도 dummy image 사용

## 6.2 GRPO에서의 차이

### CausalLM GRPO

- prompt와 completion만 관리하면 됨
- reward 함수가 거의 전부다

### VLM GRPO

- prompt 외에 image field 존재 여부가 trainer 경로를 바꾼다
- TRL 내부 multimodal path 제약을 알아야 한다
- processor adapter가 거의 필수다
- 모델별 internal state까지 신경 써야 한다

정확한 현재 정책:
- Qwen3.5 VLM GRPO:
  - text-only 데이터면 `image` 컬럼 제거
  - 실제 이미지가 있을 때만 multimodal path
- Kanana VLM GRPO:
  - text-only라도 `image="__dummy__"` 유지
  - dummy image로 multimodal path 강제

## 7. 더미 이미지 사용 규칙 정리

이 항목이 가장 자주 헷갈리는 부분이다.

### “VLM은 더미 이미지를 줘야 한다”는 말의 정확한 뜻

항상 참이 아니다. 모델과 trainer 구현에 따라 다르다.

#### Qwen3.5 VLM SFT

- 혼합 batch 정합성을 위해 필요할 수 있음
- 모든 text-only 샘플에 무조건 강제하지는 않음

#### Qwen3.5 VLM GRPO

- 현재 구현에서는 text-only 데이터에 더미 이미지를 기본 전략으로 쓰지 않음
- 오히려 TRL multimodal path 충돌을 피하기 위해 `image` 컬럼 자체를 제거

#### Kanana VLM SFT / GRPO

- text-only에서도 더미 이미지 전략이 핵심
- processor가 image slot 정합성을 요구하기 때문

결론:
- `VLM == 무조건 dummy image`는 틀린 일반화다
- **Qwen3.5와 Kanana는 전략이 다르다**

## 8. 자주 터지는 문제와 원인

### 8.1 `global eval batch size must be divisible by num_generations`

원인:
- GRPO eval batch 구성 규칙 위반

해결:
- `per_device_eval_batch_size * world_size`를 `num_generations`의 배수로 맞춘다

### 8.2 `weave` import 에러

원인:
- 현재 `trl`이 import-time에 `weave`를 요구하는 경로가 있음

현재 처리:
- [grpo_trainer_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_vlm.py) 안에 stub 주입

### 8.3 Qwen3.5 VLM `compute_3d_position_ids` / `rope_deltas` 관련 에러

원인:
- multimodal state와 text-only forward가 섞임

현재 처리:
- text-only dataset에서는 `image` 컬럼 제거
- trainer wrapper에서 stale `rope_deltas` 초기화

### 8.4 OOM

특히 VLM GRPO에서 자주 난다.

원인:
- fast path 미설치
- batch size 과대
- 4bit/bf16 미사용

실무 권장:
- `--load_in_4bit`
- `--bf16`
- `--per_device_train_batch_size 1`
- `--max_prompt_length` 축소

## 9. 수정 시 원칙

### CausalLM trainer를 수정할 때

- 먼저 `chat_template` 경계와 label mask를 본다
- tokenizer save/load 경로가 merge와 일관적인지 본다

### VLM trainer를 수정할 때

- 먼저 이 4개를 본다
  - image slot 표현 방식
  - processor가 기대하는 입력 형식
  - text-only에서 image 컬럼을 유지할지 제거할지
  - 모델 내부 multimodal state가 다음 step에 남는지

특히 VLM에서 수정은 아래 질문부터 해야 한다.

1. 이 모델은 text-only에서도 image slot이 필요한가?
2. 같은 batch에 image/no-image가 섞일 때 processor가 허용하는가?
3. TRL이 `image` 컬럼 존재만으로 multimodal path를 타는가?
4. 모델이 이전 multimodal step state를 내부에 보존하는가?

이 4개를 확인하지 않고 손대면 대부분 다시 깨진다.

## 10. 권장 사용법

### CausalLM RP SFT

- 먼저 [sft_trainer_qlora.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora.py)

### CausalLM RP GRPO

- 먼저 [grpo_trainer.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer.py)

### Qwen3.5 계열 VLM SFT

- [sft_trainer_qlora_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_vlm.py)

### Qwen3.5 계열 VLM GRPO

- [grpo_trainer_vlm.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_vlm.py)
- 단, text-only dataset인지 실제 image dataset인지 먼저 명확히 해야 한다

### Kanana 계열 VLM SFT/GRPO

- [sft_trainer_qlora_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/sft_trainer_qlora_kanana.py)
- [grpo_trainer_kanana.py](TTS_RP_Model_Lab/models/qwen3_core/grpo_trainer_kanana.py)
- dummy image 전략을 기본으로 생각해야 한다

## 11. 문서 범위

이 문서는 2026-03-12 시점의 실제 코드 기준이다.

특히 아래 두 항목은 바뀔 수 있다.
- Qwen3.5 VLM GRPO의 text-only 처리 정책
- VLM fast path 설치 여부에 따른 메모리 권장값

따라서 trainer를 수정한 뒤에는 반드시 이 문서도 함께 갱신해야 한다.
