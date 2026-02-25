# data/version_2 상세 동작 문서

이 문서는 `data/version_2`의 v6 파이프라인에서 FSM이 실제로 생성을 어떻게 제어하는지, 모델에 무엇이 입력되고 무엇이 출력되는지를 코드 기준으로 정리한다.

대상 코드
- `data/version_2/v6_qwen/generator.py`
- `data/version_2/v6_qwen/prompts.py`
- `data/version_2/v6_qwen/fsm_engine.py`
- `data/version_2/v6_qwen/state_fsm.yaml`
- `data/version_2/v6_qwen/action_fsm.yaml`
- `data/version_2/v6_gpt/*`는 모델 백엔드만 다르고 흐름은 동일하다.

## 1. 실행 단위와 큰 흐름

배치 실행
- `pipeline.py`가 시나리오북을 만든 뒤 `run_scenario(...)`를 호출한다.

단일 시나리오 실행
- `run_scenario(...)`가 아래를 초기화한다.
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
- FSM은 프롬프트 힌트만 주는 게 아니라, 노드 분기와 턴 경계 상태 강제 갱신으로 전개를 직접 제어한다.

## 2. 모델에 실제로 들어가는 입력

### 2.1 공통 메시지 구조

각 생성 노드는 아래 입력을 사용한다.
1. `state["messages"]` 기존 대화
2. 현재 노드용 `task`를 `{"role":"user","content": task}`로 추가
3. `generate_text(...)` 호출

`state["messages"]`에는 초기부터 system 메시지 2개가 들어간다.
- system 1: 시나리오북 원문
- system 2: 역할/형식 고정 정책 `BASE_SYSTEM_POLICY`

즉 모델은 매 턴 아래를 본다.
- 시나리오북
- 고정 정책
- 누적 대화 이력
- 현재 노드 전용 프롬프트

### 2.2 프롬프트에 들어가는 FSM 관련 값

A. `GEN_USER` 프롬프트
- 입력: `fsm_state`, `action_state`, `relation_status`, `relation_intent`, `genre`, `history`, `last_assistant`
- 프롬프트에 `[현재 국면]`, `[행위/사건 상태]`가 들어간다.

B. `GEN_ASSISTANT` 프롬프트
- 입력: `fsm_state`, `action_state`, `flags`, `signals`, `relation_status`, `genre`, `history`, `user_text`
- `signals` 주요 키: `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`

C. `GEN_SEXUAL` 프롬프트
- `action_state`별 `stage_rules_map` 문장을 추가 주입한다.
- `SEXUAL_1`부터 `AFTERMATH_SEX_2`까지 단계별 힌트가 명시된다.

D. `GEN_CRISIS` 프롬프트
- `action_state`별 위기 단계 힌트를 주입한다.
- `CRISIS_1`, `CRISIS_2`, `AFTERMATH_CRISIS_1` 전용 지시가 들어간다.

중요
- 일반 상태도 라벨만 쓰는 것이 아니라, 프롬프트 텍스트로 상태를 명시한다.
- 다만 `state_fsm.yaml`, `action_fsm.yaml`의 `desc`는 엔진에서 프롬프트로 직접 주입하지 않는다.

## 3. 모델 출력과 후처리

기본 출력 목표
- 2줄 형식
1. 서술 1줄
2. 큰따옴표 대사 1줄

생성 직후 수행
- 라벨 제거, 메타 제거, 참조 정규화
- 역할 충돌 검사
- 형식 검사
- 반복 검사(임베딩 메모리와 직전 턴 비교)
- 필요 시 rewrite 노드로 보정

품질 평가 노드
- `EVAL_DIALOGUE_QUALITY`는 v6에서 hard-fail이 아니라 warn 중심으로 동작한다.
- 로그에 `warn=...`를 남기고 다음 단계로 진행한다.

## 4. FSM이 전개를 제어하는 실제 방식

### 4.1 관계 FSM `state_fsm.yaml`

입력 신호
- `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `genre`

출력
- 상태 `INTRO`, `INTERACTION`, `FLIRT`, `TRAINING`, `GROWTH`, `TRAGEDY`, `PSYCHO`, `BOND`, `DEPENDENCY`, `FRACTURE`, `CRISIS`, `AFTERMATH`
- 상태 플래그 `allow_sexual`, `relation_status`, `sexual_gate` 등

엔진 병합 규칙
- `get_flags()`가 상태 플래그 + 시나리오북 전역 플래그를 병합한다.
- 시나리오북 `관계 상태` 파싱값도 `flags["relation_status"]`로 들어간다.

### 4.2 행동 FSM `action_fsm.yaml`

입력 신호
- `event`, `threat`, `pressure`, `mental_instability`, `resolve`, `sexual_ready`

출력 상태
- 일반: `IDLE`, `EVENT`, `CONFLICT`, `RESOLUTION`, `AFTERMATH`
- 위기: `CRISIS_1`, `CRISIS_2`, `AFTERMATH_CRISIS_1`
- 성행위: `SEXUAL_1`부터 `SEXUAL_4`, `AFTERMATH_SEX_1`, `AFTERMATH_SEX_2`

### 4.3 `FSM_STEP` 강제 조정 규칙

`FSM_STEP`는 `fsm_rel.step(...)`, `fsm_act.step(...)` 뒤에 후처리 규칙을 적용한다.

핵심 규칙
1. sexual 최소 유지
- 상수 `MIN_SEXUAL_TURNS = 4`
- `sexual_lock`, `sexual_turns`로 최소 턴을 유지하고 여파로 전환한다.

2. crisis 최소 유지
- 상수 `MIN_CRISIS_TURNS = 2`
- `crisis_lock`, `crisis_turns`로 위기 상태를 최소 턴 유지한다.

3. aftermath 전환
- sexual은 `AFTERMATH_SEX_1` -> `AFTERMATH_SEX_2` -> `IDLE`
- crisis는 `AFTERMATH_CRISIS_1` 이후 `EVENT` 또는 `CONFLICT`로 복귀한다.

4. 일반 상태 다양화
- `_pick_action_state(...)`가 최근 이력과 장르 가중치로 상태를 다양화한다.

5. 정체 해소 강제
- `stall_count`, `user_idle_streak` 누적 시 상태를 강제로 전진시킨다.
- `IDLE` 고착이면 `EVENT`로 승격한다.

## 5. sexual/crisis 진입과 유지

### 5.1 sexual

`DETECT`에서 계산
- `allow_sexual` 플래그
- `sexual_condition_rel == relation_status`
- 위 조건이 맞으면 `sexual_ready=True`

분기
- `after_detect`에서 `action_state`가 sexual 계열이면 `GEN_SEXUAL`

유지
- `MIN_SEXUAL_TURNS`와 lock으로 유지
- 이후 `AFTERMATH_SEX_*`로 강제 전환

### 5.2 crisis

분기
- `after_detect`에서 `action_state`가 crisis 계열이거나 `crisis_lock > 0`이면 `GEN_CRISIS`

유지
- `MIN_CRISIS_TURNS=2`, `crisis_lock`으로 최소 턴 유지
- `AFTERMATH_CRISIS_1`로 전환 후 일반 상태로 복귀

## 6. 신호 `signals` 생성 방식

`EVAL_JSON` 단계
- LLM 평가기로 JSON을 생성한다.
- 키: `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `event`

후처리
1. 범위 clamp
2. `relation_intent` 반영
3. 장르별 drift 확률 적용
4. 턴당 증감 제한
5. 정체 시 최소 변화 강제

이 신호가 다음 `FSM_STEP` 입력이 된다.

## 7. `desc`가 제어에 미치는 영향

결론
- YAML의 `desc`는 엔진이 상태 설명으로 보관하지만, 직접 프롬프트에 넣지 않는다.
- 실제 제어는 아래 조합으로 이뤄진다.
1. 상태 라벨과 신호를 프롬프트에 명시
2. sexual/crisis 전용 단계 힌트 텍스트
3. 노드 분기 강제
4. `FSM_STEP` 강제 전환
5. validator/rewrite/retry

즉 `desc` 문장을 직접 쓰지 않아도 제어는 되지만, 일반 상태의 세밀한 행동 제어는 sexual/crisis 대비 약하다.

## 8. version_2 특이점

- `v6_base_fsm.yaml`는 `version_2` 실행 경로에서 직접 로드하지 않는다.
- `EVAL_DIALOGUE_QUALITY`는 경고 중심 정책으로 동작한다.
- 일반 상태에서 `_pick_action_state(...)`로 장르별 다양화가 들어간다.

## 9. 빠른 디버깅 체크포인트

로그에서 아래를 우선 확인한다.
- `DETECT`: `sexual_ready`, 관계 파싱
- `FSM_STEP pre/post`: `base`, `action_state`, lock, turns
- `GEN_* invalid=...`: 어떤 검증에서 재시도되는지
- `EVAL_DIALOGUE_QUALITY warn=...`: 품질 경고 패턴
- `EVAL_JSON`: 신호가 실제로 어떻게 변하는지

## 10. 왜 LangGraph, FSM, 레인 검증을 함께 쓰는가

세 구성요소는 서로 다른 문제를 해결한다.

1. LangGraph
- 노드 순서와 분기를 정의한다.
- 생성, 평가, 전이, 다음 턴 준비 흐름을 고정한다.

2. FSM
- 상태 누적과 전이 규칙을 강제한다.
- sexual과 crisis의 최소 유지 턴, 여파 전환, 일반 상태 보정을 담당한다.

3. 레인 검증과 리트라이
- 출력의 역할, 형식, 반복 품질을 검증한다.
- 실패 시 rewrite 또는 재생성으로 보정한다.

왜 셋 다 필요한가
- LangGraph만으로는 상태 제약과 턴 누적 규칙이 약하다.
- FSM만으로는 실제 문면의 역할 혼동이나 형식 위반을 막지 못한다.
- 검증만으로는 전개 방향과 국면 일관성을 안정적으로 유지하기 어렵다.

결론
- LangGraph는 실행 파이프라인을 제어한다.
- FSM은 상태 기반 전개를 제어한다.
- 레인 검증은 출력 오류를 제어한다.
- 셋을 결합해야 반복 억제, 상태 일관성, 형식 준수를 동시에 달성할 수 있다.
