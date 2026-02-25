# data/version_3 상세 동작 문서

이 문서는 `data/version_3`의 v7 파이프라인에서 FSM이 실제로 생성을 어떻게 제어하는지, 모델 입력과 출력이 무엇인지, v6 대비 무엇이 달라졌는지를 코드 기준으로 정리한다.

대상 코드
- `data/version_3/v7_qwen/generator.py`
- `data/version_3/v7_qwen/prompts.py`
- `data/version_3/v7_qwen/fsm_engine.py`
- `data/version_3/v7_qwen/state_fsm.yaml`
- `data/version_3/v7_qwen/action_fsm.yaml`
- `data/version_3/v7_gpt/*`는 모델 백엔드만 다르고 흐름은 동일하다.

## 1. 실행 단위와 큰 흐름

배치 실행
- `pipeline.py`가 시나리오북을 생성한 뒤 `run_scenario(...)`를 호출한다.

단일 시나리오 실행
- `run_scenario(...)` 초기화
1. 관계 FSM 엔진 `QwenFSMEngine(state_fsm.yaml, system_lore)`
2. 행동 FSM 엔진 `QwenFSMEngine(action_fsm.yaml, system_lore)`
3. 임베딩 메모리 `EmbeddingMemory(...)`
4. LangGraph `build_graph(...)`

턴 루프 노드
1. `INIT`
2. `GEN_USER` 또는 `GEN_USER_SEXUAL`
3. `DETECT`
4. `GEN_ASSISTANT` 또는 `GEN_SEXUAL` 또는 `GEN_CRISIS`
5. `EVAL_DIALOGUE_QUALITY`
6. `EVAL_INTERNAL`
7. `EVAL_JSON`
8. `FSM_STEP`
9. `NEXT`
10. 종료 조건 충족 시 `DONE`

핵심
- v7도 FSM은 프롬프트 힌트 + 분기 제어 + 턴 경계 강제 전환으로 동작한다.

## 2. 모델에 실제로 들어가는 입력

### 2.1 공통 메시지 구조

각 생성 노드는 아래를 모델에 전달한다.
1. `state["messages"]` 누적 대화
2. 노드별 `task`를 user 메시지로 추가
3. `generate_text(...)` 호출

초기 system 메시지 2개
- system 1: 시나리오북
- system 2: 고정 정책 `BASE_SYSTEM_POLICY`

즉 매 턴 입력
- 시나리오북
- 고정 정책
- 대화 이력
- 현재 노드 지시

### 2.2 프롬프트에 들어가는 FSM 관련 값

A. `GEN_USER`
- `fsm_state`, `action_state`, `relation_status`, `relation_intent`, `genre`, `history`, `last_assistant`
- v7은 여기에 추가 강제 문구를 덧붙여 반복 억제를 강화한다.

B. `GEN_ASSISTANT`
- `fsm_state`, `action_state`, `flags`, `signals`, `relation_status`, `genre`, `history`, `user_text`
- `signals`는 `mental_instability`, `intimacy`, `threat`, `pressure`, `probe` 등을 포함한다.

C. `GEN_SEXUAL`
- `action_state`별 `stage_rules_map` 힌트를 주입한다.
- sexual 단계별로 필요한 전개 요소를 텍스트로 명시한다.

D. `GEN_CRISIS`
- crisis 단계별 힌트를 주입한다.
- 위기 반응을 행동과 대사로 드러내도록 강제한다.

중요
- YAML의 `desc` 문장은 엔진이 전이 계산에 참고하지 않으며 프롬프트에도 직접 주입되지 않는다.
- 실제 제어는 상태 라벨, 단계 힌트, 분기, 검증, 강제 전환 조합으로 이뤄진다.

## 3. 모델 출력과 후처리

기본 출력 목표
- 2줄 형식
1. 서술 1줄
2. 큰따옴표 대사 1줄

생성 직후 수행
- 메타 제거, 라벨 제거, 참조 정규화
- 역할/형식 검증
- 반복/에코 검증
- 실패 시 rewrite 경로 재시도

품질 평가 노드
- `EVAL_DIALOGUE_QUALITY`는 v7도 warn 중심이다.
- 실패를 즉시 hard-fail하지 않고 경고를 남긴다.

## 4. FSM이 전개를 제어하는 방식

### 4.1 관계 FSM `state_fsm.yaml`

입력
- `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `genre`

출력
- 상태 `INTERACTION`, `FLIRT`, `TRAINING`, `ADULT`, `TRAGEDY`, `PSYCHO`, `BOND`, `DEPENDENCY`, `FRACTURE`, `CRISIS`, `AFTERMATH`
- 상태 플래그 `allow_sexual`, `relation_status`, `sexual_gate`

엔진 병합
- `get_flags()`가 상태 플래그 + 시나리오북 전역 플래그를 합친다.
- 시나리오북 `관계 상태` 파싱값도 병합한다.

### 4.2 행동 FSM `action_fsm.yaml`

입력
- `event`, `threat`, `pressure`, `mental_instability`, `resolve`, `sexual_ready`

출력
- 일반: `IDLE`, `EVENT`, `CONFLICT`, `RESOLUTION`, `AFTERMATH`
- 위기: `CRISIS_1`, `CRISIS_2`, `AFTERMATH_CRISIS_1`
- 성행위: `SEXUAL_1`부터 `SEXUAL_4`, `AFTERMATH_SEX_1`, `AFTERMATH_SEX_2`

### 4.3 `FSM_STEP` 강제 조정 규칙

`FSM_STEP`는 두 FSM 전이 후 강제 로직을 적용한다.

핵심 규칙
1. sexual 최소 유지
- 상수 `MIN_SEXUAL_TURNS = 2`
- v6보다 빠르게 sexual에서 여파로 넘어가도록 설계된다.

2. crisis 최소 유지
- 상수 `MIN_CRISIS_TURNS = 1`
- 위기 전개를 짧게 유지하고 빠르게 여파로 이동한다.

3. aftermath 전환
- sexual은 `AFTERMATH_SEX_1` -> `AFTERMATH_SEX_2` -> `IDLE` 또는 `EVENT`
- crisis는 `AFTERMATH_CRISIS_1` 뒤 일반 전개로 복귀한다.

4. 일반 상태 선택
- v7은 일반 분기에서 `_pick_action_state` 다양화를 거의 쓰지 않고 `base_action_state`를 우선 채택한다.
- 이로 인해 v6보다 상태 선택이 더 직선적이다.

5. 정체 해소
- `stall_count`, `user_idle_streak` 누적 시 상태 전진을 강제한다.

## 5. sexual/crisis 진입과 유지

### 5.1 sexual

`DETECT`에서 계산
- `allow_sexual`
- `sexual_condition_rel == relation_status`

v7 특수 처리
- 장르가 `성인물`이면 `allow_sexual=True`, `sexual_ready=True`를 강하게 보정한다.

분기
- `after_detect`에서 sexual 계열이면 `GEN_SEXUAL`

유지
- `MIN_SEXUAL_TURNS=2`와 lock으로 짧게 유지 후 여파로 전환

### 5.2 crisis

분기
- `after_detect`에서 crisis 계열 또는 `crisis_lock>0`이면 `GEN_CRISIS`

유지
- `MIN_CRISIS_TURNS=1`로 최소 유지 후 빠르게 여파 전환

## 6. 신호 `signals` 생성 방식

`EVAL_JSON`
- LLM이 JSON 신호를 생성한다.
- 키: `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `event`

후처리
1. 범위 clamp
2. `relation_intent` 반영
3. 장르별 drift
4. 턴당 증감 제한
5. 정체 시 최소 변화 강제

이 신호가 다음 `FSM_STEP` 입력이 된다.

## 7. v7에서 장르가 상태에 개입하는 방식

`INIT`에서 장르에 따라 관계 FSM 상태를 강제로 시작한다.
- 연애물 `FLIRT`
- 육성물 `TRAINING`
- 비극 `TRAGEDY`
- 심리 시뮬레이션 `PSYCHO`
- 성인물 `ADULT`

효과
- 초반 턴에서 장르 핵심 상태를 빠르게 밟도록 유도한다.

## 8. v6 대비 v7 핵심 차이

1. 전개 속도
- v7은 `MIN_SEXUAL_TURNS`와 `MIN_CRISIS_TURNS`를 줄여 단기 전개를 강화했다.

2. 초기 장르 진입
- v7은 `INIT`에서 장르 기반 관계 상태 강제 설정이 들어간다.

3. 일반 상태 선택
- v6은 `_pick_action_state` 다양화 비중이 더 크다.
- v7은 base 상태 우선으로 직진성이 높다.

4. 성인물 보정
- v7은 성인물에서 sexual gate를 더 공격적으로 열어 빠른 진입을 유도한다.

## 9. `desc`와 제어의 관계

결론
- `state_fsm.yaml`과 `action_fsm.yaml`의 `desc` 문장은 직접 프롬프트로 들어가지 않는다.
- 제어의 실질은 아래다.
1. 상태 라벨과 신호의 프롬프트 주입
2. sexual/crisis 단계 힌트 텍스트
3. 노드 분기 강제
4. `FSM_STEP`의 lock, 턴 카운터, 강제 전환
5. validator, rewrite, retry

## 10. 빠른 디버깅 체크포인트

로그에서 우선 확인할 항목
- `INIT`: `forced_fsm_state`, 이름 파싱, 장르 파싱
- `DETECT`: `sexual_ready`, 관계 상태 보정
- `FSM_STEP pre/post`: `base`, `action_state`, lock, turns
- `GEN_* invalid=...`: 실패 원인
- `EVAL_DIALOGUE_QUALITY warn=...`: 품질 경고
- `EVAL_JSON`: 신호 변화

## 11. 왜 LangGraph, FSM, 레인 검증을 함께 쓰는가

v7에서도 제어를 안정적으로 유지하려면 세 층이 동시에 필요하다.

1. LangGraph
- 노드 실행 순서와 분기 조건을 고정한다.
- fast progression 구조에서도 실행 흐름을 안정화한다.

2. FSM
- 상태 전이와 턴 유지 규칙을 관리한다.
- v7의 빠른 sexual과 crisis 전개에서도 최소 유지와 여파 전환을 강제한다.

3. 레인 검증과 리트라이
- 출력 문면의 역할 혼동, 형식 위반, 반복을 잡는다.
- warn 정책과 rewrite 경로로 품질을 보정한다.

왜 셋 다 필요한가
- LangGraph만 있으면 흐름은 있으나 상태 제약이 약하다.
- FSM만 있으면 상태는 유지되지만 실제 출력 품질 실패를 막지 못한다.
- 검증만 있으면 사후 보정은 가능하지만 전개 방향 일관성이 약해진다.

결론
- LangGraph는 실행 구조를 제어한다.
- FSM은 전개 상태를 제어한다.
- 레인 검증은 출력 품질을 제어한다.
- 세 계층을 함께 써야 짧은 턴에서도 일관성과 품질을 동시에 확보할 수 있다.
