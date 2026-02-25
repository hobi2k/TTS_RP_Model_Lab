# qwen3_core GRPO Trainer Notes

이 문서는 `models/qwen3_core/grpo_trainer.py`의 현재 학습 방식과 보상함수 기준을 설명한다.

## 1. 파일명 관련

- 요청 파일명은 `trpo_trainer.md`였지만, 실제 구현 스크립트는 `grpo_trainer.py`다.
- 알고리즘은 TRPO가 아니라 GRPO를 사용한다.

## 2. 학습 입력 형식

`grpo_trainer.py`는 다음 컬럼을 입력으로 받는다.

- 우선 사용: `prompt`
- 대체 사용: `prompt_messages` (내부에서 `prompt`로 변환)

핵심 의도:
- GRPO는 SFT처럼 정답 라벨을 직접 맞추는 방식이 아니다.
- 같은 `prompt`에 대해 여러 응답을 생성하고, reward를 통해 상대적으로 더 나은 응답 분포로 업데이트한다.

## 2.1 전제조건: Chat Template 경계 수정

GRPO 전에 SFT 단계 chat template 경계를 반드시 점검해야 한다.

핵심 전제:
- `assistant_only_loss=True`를 쓸 경우, assistant 응답 종료 경계가 학습 가능한 형태여야 한다.
- 종료 경계를 학습하지 못하면 추론 시 과생성(`clipped_ratio` 고정)으로 이어질 수 있다.

수정 대상:
- `models/qwen3_core/model_assets/qwen3-4b-instruct/chat_template.jinja`
- 실제 학습 경로의 `chat_template.jinja`

검증 방법:
- `sft_trainer_qlora_test.py --debug_label_mask --debug_only`
- assistant 라벨 구간이 기대한 경계까지 포함되는지 확인

## 3. 현재 보상함수 구성

현재 보상함수는 아래 5개만 사용한다.

1. `reward_format`
2. `reward_role_split`
3. `reward_grounded_to_user`
4. `reward_character_consistency`
5. `reward_reference_alignment`

`reward_length`와 `reward_repetition`은 제거된 상태다.

### 3.1 가중치

현재 가중치:

- `reward_format`: `0.28`
- `reward_role_split`: `0.18`
- `reward_grounded_to_user`: `0.14`
- `reward_character_consistency`: `0.16`
- `reward_reference_alignment`: `0.24`

가중치 순서와 함수 순서는 코드에서 동일해야 한다.

### 3.2 `reward_format`

목표:
- 출력을 `서술 1블록 + 큰따옴표 대사 1블록` 형태로 안정화.

핵심 기준:
- 비어 있지 않은 줄 기준 2줄 구조
- 닫힌 큰따옴표 대사 1쌍 존재
- 2번째 줄이 대사 블록이면 최고점

### 3.3 `reward_role_split`

목표:
- 역할 혼동(assistant가 user 대행/메타 role 노출) 억제.

핵심 기준:
- `SYSTEM:`, `USER:`, `ASSISTANT:` 등의 role marker 노출 감점
- 플레이어 이름 + 조사 패턴 감점

### 3.4 `reward_grounded_to_user`

목표:
- 마지막 user 발화와의 의미 정합성 유지.

핵심 기준:
- user 발화와 completion 임베딩 유사도 점수화
- 질문 입력인데 유사도가 매우 낮으면 강감점

복창 억제:
- user/completion 3-gram 중복률(`_ngram_overlap_ratio`)이 높을 때 약감점
  - `> 0.55`: `-0.05`
  - `> 0.70`: `-0.10`

주의:
- grounded는 의미 연결을 위한 보상이며, 복창 방지 패널티는 약하게만 적용한다.

### 3.5 `reward_character_consistency`

목표:
- 시나리오북 캐릭터 정체성과 말투 일치 유지.

핵심 기준:
- 서술 줄의 주인공 이름 일치
- 대사(`quote_line`)의 종결어미 기준 존댓말/반말 일치

말투 판별 방식:
- 문장 중간 단어가 아니라 문장 끝 어절 종결형만 검사
- 존댓말 판별은 `습니다/습니까/세요/해요/예요/게요/...` 계열 종결어미 포함

### 3.6 `reward_reference_alignment`

목표:
- 같은 프롬프트의 reference 응답과 의미적으로 정렬된 출력 유도.

핵심 기준:
- completion/reference 임베딩 유사도 점수화
- reference가 없으면 중립 점수 처리

## 4. 임베딩 보상 정책

임베딩 모델:
- 기본값: `data/embedding/BGE-m3-ko`

중요 정책:
- 임베딩 초기화 실패 시 폴백(F1/Jaccard 등) 사용하지 않는다.
- 초기화 실패는 학습 즉시 실패(hard fail)로 처리한다.

의도:
- 보상 정의를 고정해 실험 재현성과 해석 가능성을 유지한다.

## 5. 왜 이 구성이 필요한가

현재 구성은 다음 균형을 목표로 한다.

- 형식 안정성: `reward_format`
- 역할 안정성: `reward_role_split`
- 대화 반응성: `reward_grounded_to_user`
- 캐릭터 일관성: `reward_character_consistency`
- 참조 정렬: `reward_reference_alignment`

즉, "형식만 맞는 답변"이나 "reference만 베끼는 답변" 한쪽으로 치우치지 않도록 설계했다.

## 6. 실행 예시

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run models/qwen3_core/grpo_trainer.py \
  --model_name models/qwen3_core/model_assets/qwen3_4b_rp \
  --train_data /mnt/d/rp_data/grpo/grpo_train.jsonl \
  --eval_data /mnt/d/rp_data/grpo/grpo_eval.jsonl \
  --output_dir models/qwen3_core/model_assets/qwen3_4b_rp_grpo \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 2e-6 \
  --max_prompt_length 1024 \
  --max_completion_length 220 \
  --num_generations 4 \
  --per_device_eval_batch_size 4 \
  --gradient_checkpointing \
  --bf16 \
  --use_lora \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant \
  --bnb_4bit_compute_dtype bfloat16 \
  --reward_embedding_model data/embedding/BGE-m3-ko \
  --reward_embedding_batch_size 16
```
