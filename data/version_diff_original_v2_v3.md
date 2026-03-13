# `data/original`, `data/version_2`, `data/version_3` 기능 차이 정리

이 문서는 `TTS_RP_Model_Lab/data` 아래의 세 버전:

- `original` = `v5`
- `version_2` = `v6`
- `version_3` = `v7`

의 **코드 기준 기능 차이**만 정리한다.  
추상적 해석은 제외하고, 실제로 달라진 **파일 구조 / 기본 파라미터 / FSM 상태 / 전이 규칙 / 파이프라인 동작**만 기록한다.

---

## 1. 디렉터리/파일 구조 차이

### `original`
- 디렉터리:
  - `original/v5_qwen`
  - `original/v5_gpt`
- 주요 파일:
  - `generator.py`
  - `pipeline.py`
  - `fsm_engine.py`
  - `prompts.py`
  - `state_fsm.yaml`
  - `action_fsm.yaml`
- 없음:
  - `backend.py`
  - `v6_base_fsm.yaml`

### `version_2`
- 디렉터리:
  - `version_2/v6_qwen`
  - `version_2/v6_gpt`
- 추가 파일:
  - `version_2/v6_qwen/backend.py`
  - `version_2/v6_gpt/backend.py`
  - `version_2/v6_base_fsm.yaml`
- 구조 차이:
  - `backend.py`가 생겨 모델 호출 백엔드가 별도 파일로 분리됨
  - `v6_base_fsm.yaml`가 존재하지만 README 기준 실제 실행 경로에서 직접 로드되지는 않음

### `version_3`
- 디렉터리:
  - `version_3/v7_qwen`
  - `version_3/v7_gpt`
- 유지되는 구조:
  - `backend.py` 분리 구조 유지
- 제거된 보조 파일:
  - `v6_base_fsm.yaml` 없음

---

## 2. 파이프라인 CLI 기본값 차이

기준 파일:
- `original/v5_qwen/pipeline.py`
- `version_2/v6_qwen/pipeline.py`
- `version_3/v7_qwen/pipeline.py`

### `--turns` 기본값
- `original/v5`: `8`
- `version_2/v6`: `8`
- `version_3/v7`: `3`

즉 `version_3`만 기본 멀티턴 길이가 줄어 있다.

### 기본 action FSM 경로
- `original/v5`: `data/original/v5_qwen/action_fsm.yaml`
- `version_2/v6`: `data/version_2/v6_qwen/action_fsm.yaml`
- `version_3/v7`: `data/version_3/v7_qwen/action_fsm.yaml`

### 시나리오북 생성 후 후처리

#### `original/v5`
- `generate_scenario_book(...)` 호출 후
  - `normalize_scenario_text(text)`
  - `is_valid_scenario_book(text)` 검사
- 말투 강제/재작성 루프 없음

#### `version_2/v6`
- `generate_scenario_book(...)` 호출 시
  - `chosen_speech_style` 인자 추가
- 이후 추가 처리:
  - `enforce_base_speech_style_line(text, chosen_speech_style)`
  - `has_speech_style_mismatch(text)` 검사
  - 최근 protagonist 발화 재작성 최대 2회

#### `version_3/v7`
- `version_2`의 말투 보정 로직 유지
- 추가 차이:
  - `invalid_reason` import
  - 시나리오북 검증 실패 시 `invalid_reason(text)`를 로그로 출력

---

## 3. 생성 그래프 노드 차이

기준 문서와 코드:
- `original/README.md`
- `version_2/README.md`
- `version_3/README.md`
- 각 버전 `generator.py`

### `original/v5`
- 기본 노드 흐름:
  1. `INIT`
  2. `GEN_USER` 또는 `GEN_USER_SEXUAL`
  3. `DETECT`
  4. `GEN_ASSISTANT` 또는 `GEN_CRISIS` 또는 `GEN_SEXUAL`
  5. `EVAL_INTERNAL`
  6. `EVAL_JSON`
  7. `FSM_STEP`
  8. `NEXT`

### `version_2/v6`
- 노드 추가:
  - `EVAL_DIALOGUE_QUALITY`
- 흐름:
  1. `INIT`
  2. `GEN_USER` 또는 `GEN_USER_SEXUAL`
  3. `DETECT`
  4. `GEN_ASSISTANT` 또는 `GEN_SEXUAL` 또는 `GEN_CRISIS`
  5. `EVAL_DIALOGUE_QUALITY`
  6. `EVAL_INTERNAL`
  7. `EVAL_JSON`
  8. `FSM_STEP`
  9. `NEXT`

### `version_3/v7`
- `EVAL_DIALOGUE_QUALITY` 유지
- 그래프 구조는 `version_2`와 동일

### `EVAL_DIALOGUE_QUALITY` 동작 차이
- `original/v5`
  - 실패 시 quality rewrite로 보내는 경로가 있음
- `version_2/v6`
  - README와 코드 기준 `warn` 중심 동작
  - hard-fail보다는 경고 기록 후 다음 단계 진행
- `version_3/v7`
  - `version_2`와 동일하게 경고 중심

---

## 4. pacing 관련 상수 차이

기준 파일:
- `original/v5_qwen/generator.py`
- `version_2/v6_qwen/generator.py`
- `version_3/v7_qwen/generator.py`

### 성행위 최소 유지 턴
- `original/v5`: `MIN_SEXUAL_TURNS = 4`
- `version_2/v6`: `MIN_SEXUAL_TURNS = 4`
- `version_3/v7`: `MIN_SEXUAL_TURNS = 2`

### crisis 최소 유지 턴
- `original/v5`: 별도 `MIN_CRISIS_TURNS` 상수 없음
- `version_2/v6`: `MIN_CRISIS_TURNS = 2`
- `version_3/v7`: `MIN_CRISIS_TURNS = 1`

즉:
- `original`에는 crisis 최소 유지 상수가 없음
- `version_2`부터 crisis 턴 잠금이 들어감
- `version_3`는 sexual/crisis 최소 유지 턴 수를 줄임

---

## 5. `VNState` 필드 차이

기준 파일:
- `version_2/v6_qwen/generator.py`
- `version_3/v7_qwen/generator.py`

### `original/v5`
- state dict를 직접 다루는 구조
- typed field 선언이 v6/v7처럼 정리되어 있지 않음

### `version_2/v6`, `version_3/v7`
- 공통적으로 다음 상태 필드가 명시됨
  - `crisis_lock`
  - `crisis_turns`
  - `sexual_turns`
  - `sexual_lock`
  - `aftermath_turns`
  - `user_idle_streak`
  - `stall_count`
  - `action_state_hist`

즉 `version_2`부터 crisis/sexual lock 관리 필드가 명시적 상태로 올라왔다.

---

## 6. `action_fsm.yaml` 차이

### 6-1. `original(v5)` -> `version_2(v6)`

기준 파일:
- `original/v5_qwen/action_fsm.yaml`
- `version_2/v6_qwen/action_fsm.yaml`

#### 추가된 상태
`version_2`에서 아래 상태가 추가됨:
- `CRISIS_1`
- `CRISIS_2`
- `AFTERMATH_CRISIS_1`

#### 추가된 전이
- `IDLE -> CRISIS_1` if `mental_instability >= 2`
- `EVENT -> CRISIS_1` if `mental_instability >= 2`
- `CONFLICT -> CRISIS_1` if `mental_instability >= 2`
- `AFTERMATH -> CRISIS_1` if `mental_instability >= 2`
- `CRISIS_1 -> CRISIS_2`
- `CRISIS_2 -> AFTERMATH_CRISIS_1`
- `AFTERMATH_CRISIS_1 -> EVENT` if `mental_instability <= 2`

#### 기존 전이 threshold 변경
- `IDLE -> CONFLICT`
  - `original`: `threat >= 2`
  - `version_2`: `threat >= 1`
- `EVENT -> CONFLICT`
  - `original`: `threat >= 2`
  - `version_2`: `threat >= 1`
- `CONFLICT -> AFTERMATH`
  - `original`: `event >= 2`
  - `version_2`: `event >= 1`
- `AFTERMATH -> IDLE`
  - `original`: `pressure == 0 and threat == 0`
  - `version_2`: `pressure <= 1 and threat <= 1`
- `AFTERMATH_SEX_2 -> IDLE`
  - `original`: `pressure == 0 and threat == 0`
  - `version_2`: `pressure <= 1 and threat <= 1`

#### state_flags 추가
`version_2`에서 crisis 상태 힌트가 추가됨:
- `CRISIS_1: 정신 붕괴 초입`
- `CRISIS_2: 정신 붕괴 심화`
- `AFTERMATH_CRISIS_1: 붕괴 여파 정리`

### 6-2. `version_2(v6)` -> `version_3(v7)`

기준 파일:
- `version_2/v6_qwen/action_fsm.yaml`
- `version_3/v7_qwen/action_fsm.yaml`

#### FSM 메타 정보 변경
- `name`
  - `v6`: `qwen_vn_action_fsm_v1`
  - `v7`: `qwen_vn_action_fsm_v2`
- `initial_state`
  - `v6`: `IDLE`
  - `v7`: `EVENT`

#### 주요 전이 변경
- `IDLE`
  - `v6`: `EVENT`로 가려면 `event >= 1`
  - `v7`: `EVENT`로 가려면 `event >= 0`
- `EVENT -> RESOLUTION`
  - `v6`: `resolve == 1`
  - `v7`: `event >= 1`
- `CONFLICT -> RESOLUTION`
  - `v6`: `resolve == 1`
  - `v7`: `resolve >= 1`
- `RESOLUTION -> AFTERMATH`
  - `v6`: `event >= 1`
  - `v7`: `event >= 0`
- `AFTERMATH`
  - `v6`: `IDLE`로 복귀
  - `v7`: `EVENT`로 복귀
- `AFTERMATH_CRISIS_1`
  - `v6`: `EVENT` if `mental_instability <= 2`
  - `v7`: `EVENT` if `event >= 0`
- `AFTERMATH_SEX_2`
  - `v6`: `IDLE` if `pressure <= 1 and threat <= 1`
  - `v7`: `EVENT` if `event >= 0`

#### 제거된 전이
- `v6`의 `RESOLUTION -> SEXUAL_1` 전이가 `v7`에는 없음

#### 상태 설명/힌트 텍스트 변경
- `v7`은 state `desc`와 `action_hint`를 더 짧은 텍스트로 교체
- 기능적으로 중요한 변화는 아니고, 프롬프트에 넣는 힌트 문구 차이만 있음

---

## 7. `state_fsm.yaml` 차이

### 7-1. `original(v5)` -> `version_2(v6)`

기준 파일:
- `original/v5_qwen/state_fsm.yaml`
- `version_2/v6_qwen/state_fsm.yaml`

#### 상태 구성
- 상태 집합은 사실상 동일 계열:
  - `INTRO`, `INTERACTION`, `FLIRT`, `TRAINING`, `GROWTH`, `TRAGEDY`, `PSYCHO`, `BOND`, `DEPENDENCY`, `FRACTURE`, `CRISIS`, `AFTERMATH`

#### 전이 threshold 변경
- `INTERACTION -> BOND`
  - `original`: `intimacy >= 2`
  - `version_2`: `intimacy >= 1`
- `FLIRT -> BOND`
  - `original`: `intimacy >= 2`
  - `version_2`: `intimacy >= 1`
- `TRAINING -> BOND`
  - `original`: `intimacy >= 2`
  - `version_2`: `intimacy >= 1`
- `GROWTH -> BOND`
  - `original`: `intimacy >= 2`
  - `version_2`: `intimacy >= 1`
- `BOND -> INTERACTION`
  - `original`: `pressure >= 2`
  - `version_2`: `pressure >= 3`
- `FRACTURE -> CRISIS`
  - `original`: `mental_instability >= 3`
  - `version_2`: `mental_instability >= 2`
- `FRACTURE -> DEPENDENCY`
  - `original`: `intimacy >= 2 and threat == 0`
  - `version_2`: `intimacy >= 1 and threat <= 1`

#### 기능적 의미
- `version_2`는 `BOND` 진입과 `CRISIS` 진입 threshold가 더 낮아졌다.
- `BOND -> INTERACTION` 이탈은 더 어렵게 바뀌었다.

### 7-2. `version_2(v6)` -> `version_3(v7)`

기준 파일:
- `version_2/v6_qwen/state_fsm.yaml`
- `version_3/v7_qwen/state_fsm.yaml`

#### FSM 메타 정보 변경
- `name`
  - `v6`: `qwen_vn_state_fsm_v2`
  - `v7`: `qwen_vn_state_fsm_v3`
- `initial_state`
  - `v6`: `INTRO`
  - `v7`: `INTERACTION`

#### 상태 집합 변경
`v6`에 있었고 `v7`에 없는 상태:
- `INTRO`
- `GROWTH`

`v7`에 새로 추가된 상태:
- `ADULT`

#### 장르별 진입 전이 단순화
`v7`의 `INTERACTION`에서는 다음 전이가 모두 `probe >= 0` 조건으로 연결됨:
- `연애물 -> FLIRT`
- `육성물 -> TRAINING`
- `성인물 -> ADULT`
- `비극 -> TRAGEDY`
- `심리 시뮬레이션 -> PSYCHO`

즉 `v6`처럼 intimacy/threat/mental_instability 같은 개별 신호 threshold를 더 보지 않고, 장르 기반 분기가 바로 들어간다.

#### 후반 상태 전이 단순화
- `TRAGEDY -> AFTERMATH`
  - `v7` 추가
- `PSYCHO -> AFTERMATH`
  - `v7` 추가
- `BOND -> AFTERMATH`
  - `v7` 추가
- `DEPENDENCY -> AFTERMATH`
  - `v7` 추가
- `FRACTURE -> AFTERMATH`
  - `v7` 추가
- `CRISIS -> AFTERMATH`
  - `v6`: `resolve == 1`
  - `v7`: `resolve >= 0`

#### 기능적 의미
- `v7`은 관계 FSM 상태 수를 줄였고, 장르 핵심 상태 진입을 단순화했다.
- `AFTERMATH`로 가는 경로가 크게 늘었다.

---

## 8. 프롬프트 구성 차이

기준 파일:
- `original/v5_qwen/prompts.py`
- `version_2/v6_qwen/prompts.py`
- `version_3/v7_qwen/prompts.py`

### `original(v5)` -> `version_2(v6)`

#### 시스템 메시지 구성 분리
- `original`
  - `generator.py` 내부에 `BASE_SYSTEM_POLICY`와 system message 조립 로직이 있었음
- `version_2`
  - `prompts.py`로 `BASE_SYSTEM_POLICY`와 `build_system_messages()`가 이동

#### `build_user_prompt()` 규칙 강화
`version_2`에서 아래 항목이 명시적으로 추가/강화됨:
- 각 줄 70자 제한
- 첫 줄은 player 서술 1문장, 둘째 줄은 player 대사 1문장
- 전턴 `{protagonist}` 출력/대화 이력 복사 금지
- 직전 턴과 다른 발화 방식 강제
- 직전 player 대사의 핵심 구절 8자 이상 재사용 금지
- 상대가 거절/유보했으면 같은 요구 반복 금지
- 상대가 수락했으면 같은 동의 재질문 금지
- 새 정보 1개 + 새 결정 1개 + 다음 행동 1개 강제

즉 `version_2`부터 player 프롬프트 제약이 더 상세해졌다.

### `version_2(v6)` -> `version_3(v7)`
- `prompts.py` 구조는 크게 동일
- 핵심 차이는 prompt 함수보다 FSM/초기 상태/턴 수 쪽에서 발생

---

## 9. `original`, `version_2`, `version_3`를 한 번에 요약하면

### `original(v5)`
- `backend.py` 없음
- `EVAL_DIALOGUE_QUALITY`는 존재
- `MIN_SEXUAL_TURNS = 4`
- `MIN_CRISIS_TURNS` 없음
- action FSM에 crisis 상태 없음
- pipeline 시나리오북 말투 보정 없음

### `version_2(v6)`
- `backend.py` 추가
- `v6_base_fsm.yaml` 추가
- action FSM에 `CRISIS_1`, `CRISIS_2`, `AFTERMATH_CRISIS_1` 추가
- `MIN_CRISIS_TURNS = 2` 추가
- 시나리오북 말투 보정/재작성 추가
- prompt 규칙이 더 상세해짐

### `version_3(v7)`
- `backend.py` 유지
- 기본 `turns = 3`
- `MIN_SEXUAL_TURNS = 2`
- `MIN_CRISIS_TURNS = 1`
- state FSM에서 `INTRO`, `GROWTH` 제거, `ADULT` 추가
- action FSM `initial_state = EVENT`
- state FSM `initial_state = INTERACTION`
- `AFTERMATH -> EVENT`, `AFTERMATH_SEX_2 -> EVENT` 같은 빠른 재진입 전이 사용
- 장르별 상태 진입 조건을 단순화

