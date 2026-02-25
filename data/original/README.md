# data/original 상세 동작 문서

이 문서는 `data/original`의 v5 파이프라인에서 FSM이 실제로 생성을 어떻게 제어하는지, 모델에 무엇이 입력되고 무엇이 출력되는지를 코드 기준으로 정리한다.

대상 코드
- `data/original/v5_qwen/generator.py`
- `data/original/v5_qwen/prompts.py`
- `data/original/v5_qwen/fsm_engine.py`
- `data/original/v5_qwen/state_fsm.yaml`
- `data/original/v5_qwen/action_fsm.yaml`
- `data/original/v5_gpt/*`는 백엔드만 다르고 구조는 동일하다.

## 1. 실행 단위와 큰 흐름

배치 실행
- `pipeline.py`가 시나리오북을 생성한 뒤 `run_scenario(...)`를 호출한다.

단일 시나리오 실행
- `run_scenario(...)`는 아래를 초기화한다.
1. 관계 FSM 엔진: `QwenFSMEngine(state_fsm.yaml, system_lore)`
2. 행동 FSM 엔진: `QwenFSMEngine(action_fsm.yaml, system_lore)`
3. 임베딩 메모리: `EmbeddingMemory(...)`
4. LangGraph: `build_graph(...)`

턴 루프 기본 경로
1. `INIT`
2. `GEN_USER` 또는 `GEN_USER_SEXUAL`
3. `DETECT`
4. `GEN_ASSISTANT` 또는 `GEN_CRISIS` 또는 `GEN_SEXUAL`
5. `EVAL_INTERNAL`
6. `EVAL_JSON`
7. `FSM_STEP`
8. `NEXT`
9. 종료 조건 전까지 2로 반복

핵심
- FSM은 프롬프트 힌트 수준이 아니라, 분기 노드 선택과 다음 턴 상태 강제 갱신까지 포함해 전개를 제어한다.

## 2. 모델에 실제로 들어가는 입력

### 2.1 공통 메시지 구조

각 생성 노드에서 모델 호출은 기본적으로 아래 형태다.

1. `state["messages"]` 기존 대화
2. 현재 노드용 `task` 프롬프트를 `{"role":"user","content": task}`로 추가
3. `generate_text(...)` 호출

`state["messages"]`에는 초기부터 system 2개가 들어간다.
- system 1: 시나리오북 원문
- system 2: 역할 분리/형식 강제 정책 문자열

즉 모델은 매 턴 다음을 본다.
- 시나리오북
- 고정 정책
- 직전까지의 실제 대화
- 현재 노드 전용 지시 프롬프트

### 2.2 프롬프트에 들어가는 FSM 관련 값

#### A. `GEN_USER` 프롬프트 (`build_user_prompt`)

입력 필드
- `fsm_state`: 관계 FSM 현재 상태 문자열
- `action_state`: 행동 FSM 현재 상태 문자열
- `relation_status`: 관계 상태 라벨
- `relation_intent`: 관계 상승/악화 의도
- `genre`, `history`, `last_assistant`, 이름 정보

프롬프트에는 명시적으로 다음 블록이 들어간다.
- `[현재 국면] {fsm_state}`
- `[행위/사건 상태] {action_state}`
- 상태별 전개 규칙 목록(IDLE/EVENT/CONFLICT/RESOLUTION/AFTERMATH/SEXUAL)

#### B. `GEN_ASSISTANT` 프롬프트 (`build_assistant_turn_prompt`)

입력 필드
- `fsm_state`
- `action_state`
- `flags`에서 해석한 `allow_sexual`
- `relation_status`
- `signals` 수치
  - `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`
- `genre`, `history`, `user_text`

프롬프트에는 명시적으로 다음이 들어간다.
- `국면: {fsm_state}`
- `행위/사건 상태: {action_state}`
- `성행위 허용: {allow_sexual}`
- 수치 신호들

#### C. `GEN_SEXUAL` 프롬프트 (`build_sexual_prompt`)

일반 프롬프트와 차이
- `action_state`별 `stage_rules_map` 힌트가 추가된다.
  - `SEXUAL_1`~`SEXUAL_4`
  - `AFTERMATH_SEX_1`~`AFTERMATH_SEX_2`

즉 sexual은 일반 상태보다 더 구체적인 단계 힌트 텍스트를 모델에 전달한다.

#### D. `GEN_CRISIS` 프롬프트 (`build_crisis_prompt`)

일반 프롬프트와 차이
- 위기 국면 전용 제약이 들어간다.
- 감각 붕괴, 혼란, 회피 등 crisis 성격의 지시를 별도로 준다.

## 3. 모델 출력과 후처리

### 3.1 기본 출력 형식 요구

대부분 노드에서 출력은 2줄을 목표로 한다.
1. 서술 1줄
2. 큰따옴표 대사 1줄

생성 직후 수행
- 라벨 제거, 따옴표 정규화
- 역할 혼동 검사
- 반복 검사(임베딩 메모리 기반)
- 필요 시 재작성 노드(`GEN_ASSISTANT_REWRITE*`)로 보정

### 3.2 임베딩 메모리 사용

`EmbeddingMemory`는 반복 억제용이다.
- kind별 벡터 저장(`narration`, `dialogue`, `assistant`, `user` 등)
- `is_repetitive(...)`로 유사도 임계치 검사
- 반복 판정 시 해당 노드 재시도

중요
- 외부 지식 검색형 RAG가 아니라, 반복/정체 제어용 메모리다.

## 4. FSM이 전개를 제어하는 실제 방식

## 4.1 관계 FSM(`state_fsm.yaml`)

입력 신호
- `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `genre`

출력
- 상태 이름(`INTRO`, `INTERACTION`, `BOND`, `CRISIS` 등)
- `state_flags` 기반 플래그
  - `allow_sexual`
  - `relation_status`

보강 로직
- `fsm_engine.py`가 시나리오북에서 전역 `allow_sexual`을 파싱해 상태 플래그와 병합한다.
- `relation_status`는 신호값으로도 재추정한다.

## 4.2 행동 FSM(`action_fsm.yaml`)

입력 신호
- `event`, `threat`, `pressure`, `resolve`, `sexual_ready`

출력
- 베이스 행동 상태
  - `IDLE`, `EVENT`, `CONFLICT`, `RESOLUTION`, `AFTERMATH`
  - `SEXUAL_1`~`SEXUAL_4`
  - `AFTERMATH_SEX_1`~`AFTERMATH_SEX_2`

## 4.3 `FSM_STEP`에서의 강제 조정 규칙

`EVAL_JSON`으로 받은 신호를 넣어 두 FSM을 `step(...)`한 뒤, `fsm_step_node`가 최종 `action_state`를 재보정한다.

적용되는 강제 규칙
1. sexual 진행 잠금
   - 상수: `MIN_SEXUAL_TURNS = 4`
   - sexual 진입 후 최소 턴을 유지하도록 lock/counter 사용
2. sexual aftermath 전환
   - `SEXUAL_n` 진행 후 `AFTERMATH_SEX_1` -> `AFTERMATH_SEX_2` -> `IDLE`
3. 준비 안 된 sexual 강등
   - `sexual_ready=False`인데 sexual 상태면 `EVENT`로 강등
4. idle 정체 해소
   - `user_idle_streak >= 2`이고 `IDLE`이면 `EVENT`로 승격
5. 다양화
   - `_pick_action_state(...)`로 최근 이력, 장르 가중치를 반영해 일반 상태를 다양화

결론
- FSM 결과를 그대로 쓰지 않고, 서사 정체/조건 위반을 막는 룰로 한 번 더 강제한다.

## 5. sexual과 crisis 진입 조건, 유지 방식

## 5.1 sexual 진입

`DETECT`에서 `sexual_ready` 계산
- `allow_sexual` 해석 결과가 참
- `sexual_condition_rel == relation_status`
- 필요 조건 충족 시 true

`after_detect` 분기
- `action_state`가 `SEXUAL_*`면 `GEN_SEXUAL`로 이동

유지 방식
- `FSM_STEP`의 `sexual_turns`, `sexual_lock`으로 최소 턴 유지
- 이후 aftermath 상태로 단계적 전환

## 5.2 crisis 진입

`after_detect` 분기
- `fsm_rel.get_state() == "CRISIS"`면 `GEN_CRISIS`로 이동

유지 방식
- original(v5)에는 v7처럼 crisis 턴 고정 카운터가 별도로 없다.
- 관계 FSM이 `CRISIS`를 유지하는 동안 매 턴 `GEN_CRISIS` 경로를 탄다.
- `resolve` 등 신호가 바뀌어 관계 FSM이 이탈하면 일반 경로로 복귀한다.

## 5.3 일반 상태와 차이

sexual/crisis의 강한 제어
- 전용 생성 노드(`GEN_SEXUAL`, `GEN_CRISIS`)
- 전용 프롬프트 제약
- sexual의 경우 단계별 힌트 맵 + 턴 잠금

일반 상태의 제어
- `action_state` 라벨 + 공통 규칙 기반
- `_pick_action_state` 다양화
- lane 검사/반복 검사/재시도

즉 sexual/crisis는 고해상도 제어, 일반 상태는 상대적으로 저해상도 제어다.

## 6. 신호(`signals`) 수치가 어떻게 만들어지는가

`EVAL_JSON` 단계
- LLM 평가기로 JSON 생성
- 키: `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `event`

후처리 보정
1. 범위 clamp
2. 장르별 drift 가중치 적용
3. 턴당 변화량 제한(대체로 ±1)

이렇게 만든 신호가 다음 `FSM_STEP` 입력이 되어 다음 턴 전개를 바꾼다.

## 7. "숫자 파라미터가 아닌데 어떻게 제어되나"에 대한 답

제어는 하나로 하지 않는다. 아래 4개를 동시에 건다.
1. 프롬프트 조건 주입(`fsm_state`, `action_state`, `flags`, `signals`)
2. 노드 분기 강제(`GEN_ASSISTANT`/`GEN_SEXUAL`/`GEN_CRISIS`)
3. 출력 검증과 재시도(lane/역할/반복/형식)
4. 턴 경계 상태 강제 갱신(`FSM_STEP`)

그래서 단순 라벨 전달보다 훨씬 강한 구조적 제어가 된다.

## 8. 빠른 디버깅 체크포인트

로그에서 아래를 보면 현재 제어가 어디서 걸리는지 바로 확인할 수 있다.
- `DETECT`: `sexual_ready` 계산 결과
- `FSM_STEP pre/post`: 전이 전후 `action_state`
- `GEN_* invalid=...`: 어떤 검증에서 재시도되는지
- `EVAL_JSON`: 신호값이 실제로 어떻게 바뀌는지


## 9. 왜 LangGraph, FSM, 레인 검증을 함께 쓰는가

세 구성요소는 역할이 겹치지 않는다.

1. LangGraph
- 노드 실행 순서와 분기를 관리한다.
- 어떤 노드가 언제 실행되는지 구조를 고정한다.

2. FSM
- 상태를 누적하고 전이 규칙을 강제한다.
- 최소 유지 턴, 단계적 전환, 금지 조건을 관리한다.

3. 레인 검증과 리트라이
- 출력 문면 품질을 사후 보정한다.
- 역할 혼동, 형식 위반, 반복 같은 실패를 걸러낸다.

왜 셋 다 필요한가
- LangGraph만 있으면 순서만 있고 상태 누적 제약이 약하다.
- FSM만 있으면 상태는 정해지지만 실제 출력 품질 실패를 막지 못한다.
- 검증만 있으면 사후 수정은 가능하지만 전개 방향 일관성이 약하다.

결론
- LangGraph는 실행 골격을 만든다.
- FSM은 전개 방향과 상태 제약을 건다.
- 레인 검증은 실제 출력 오류를 막는다.
- 셋을 같이 써야 전개 일관성과 출력 품질을 동시에 확보할 수 있다.
