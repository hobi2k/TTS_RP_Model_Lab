# FSM State와 Action State 정리

이 문서는 `data/original` 계열 파이프라인에서 사용하는 `fsm_state`와 `action_state`가 무엇인지, 어디서 만들어지고 어디에 주입되는지, 그리고 서로가 실제 생성 흐름에 어떤 영향을 주는지를 정리한다.

기준 코드는 아래 두 곳이다.

- `data/original/README.md`
- `data/original/v5_qwen/generator.py`

## 1. 한 줄 요약

- `fsm_state`: 현재 관계/국면이 어느 단계에 있는지를 나타내는 상위 상태
- `action_state`: 이번 턴에서 어떤 사건/행동 전개를 진행할지를 나타내는 하위 상태

쉽게 말하면:

- `fsm_state`는 "지금 서사가 어떤 국면인가"
- `action_state`는 "이번 턴에 어떤 종류의 사건을 진행할 것인가"

를 담당한다.

## 2. 두 상태의 역할 차이

### `fsm_state`

`fsm_state`는 관계 FSM의 현재 상태다.  
이 값은 서사의 큰 방향, 분위기, 긴장도, 위기 여부 같은 상위 레벨 흐름을 표현한다.

이 상태는 보통 이런 질문에 답한다.

- 지금 두 인물의 관계가 안정적인가
- 긴장이나 위기가 커졌는가
- 현재 장면이 평상 국면인지 위기 국면인지

즉 `fsm_state`는 "상위 서사 레일"에 가깝다.

### `action_state`

`action_state`는 행동 FSM의 현재 상태다.  
이 값은 이번 턴에서 실제로 어떤 종류의 사건이 진행되고 있는지를 나타낸다.

예를 들면:

- `IDLE`
- `EVENT`
- `CONFLICT`
- `RESOLUTION`
- `AFTERMATH`
- `SEXUAL_1 ~ SEXUAL_4`
- `AFTERMATH_SEX_1 ~ AFTERMATH_SEX_2`

즉 `action_state`는 "현재 턴의 사건 진행 단계"에 가깝다.

## 3. 어디서 만들어지나

두 상태는 모두 `fsm_step_node`에서 갱신된다.

핵심 구조는 다음과 같다.

1. 앞 단계에서 `EVAL_JSON`이 생성된다.
2. 여기서 `mental_instability`, `intimacy`, `threat`, `pressure`, `probe`, `resolve`, `event`, `sexual_ready` 같은 신호를 읽는다.
3. 관계 FSM과 행동 FSM에 각각 이 신호를 넣어 `step()`을 호출한다.
4. 관계 FSM의 상태는 `fsm_state`가 된다.
5. 행동 FSM의 상태는 `base_action_state`가 되며, 이후 추가 규칙으로 한 번 더 보정된 뒤 최종 `action_state`가 된다.

즉 둘 다 같은 턴 평가 결과를 재료로 하지만, 서로 다른 목적의 FSM을 돈다.

## 4. 어떤 신호를 보나

### `fsm_state`를 만드는 관계 FSM

관계 FSM은 주로 아래 신호를 본다.

- `mental_instability`
- `intimacy`
- `threat`
- `pressure`
- `probe`
- `resolve`
- `genre`

이 FSM은 "관계 긴장", "심리 상태", "국면 전환" 같은 큰 흐름을 잡는 데 집중한다.

### `action_state`를 만드는 행동 FSM

행동 FSM은 주로 아래 신호를 본다.

- `event`
- `threat`
- `pressure`
- `resolve`
- `sexual_ready`

이 FSM은 "이번 턴이 사건 턴인지", "갈등 턴인지", "해소 턴인지", "sexual 단계로 진입 가능한지"를 판단하는 데 집중한다.

즉 같은 입력을 공유해도, 해석 목적이 다르다.

## 5. 생성 흐름에서 어떻게 호출되나

`fsm_state`와 `action_state`는 단순 저장용 상태가 아니다.  
실제로 프롬프트 조립과 생성 노드 분기에서 계속 사용된다.

### 프롬프트에 직접 주입

`generator.py`의 `gen_user_node`와 대응되는 assistant 생성 경로에서는 현재 상태를 읽어서 프롬프트에 넣는다.

주입되는 형태는 대략 다음과 같다.

- `[현재 국면] {fsm_state}`
- `[행위/사건 상태] {action_state}`
- `국면: {fsm_state}`
- `행위/사건 상태: {action_state}`

즉 모델은 매 턴 생성 시점마다:

- 지금 서사의 큰 국면이 무엇인지
- 이번 턴의 사건 단계가 무엇인지

를 명시적으로 전달받는다.

## 6. `fsm_state`는 구체적으로 무엇에 영향을 주나

### 1. 상위 국면을 프롬프트에 고정한다

`fsm_state`는 프롬프트에서 현재 장면의 서사적 톤을 고정하는 역할을 한다.  
예를 들어 위기 국면이면 모델은 평온한 일상 대화를 생성하는 대신 더 압박감 있고 긴장된 방향으로 유도된다.

### 2. 생성 결과의 적합성을 검사하는 기준이 된다

`generator.py`의 `lane_satisfied(fsm_state, text, embed_memory)` 같은 검사 로직은 `fsm_state`를 기준으로 생성된 텍스트가 현재 국면에 맞는지 확인한다.

즉 `fsm_state`는 단순 참고값이 아니라, "이 출력이 지금 국면에 맞는가"를 판단하는 기준으로도 쓰인다.

### 3. 위기 전용 생성 경로를 열 수 있다

관계 FSM이 특정 위기 상태, 예를 들어 `CRISIS`에 들어가면 생성 파이프라인은 위기 전용 경로를 선택할 수 있다.

즉 `fsm_state`는:

- 프롬프트의 분위기
- 출력 검증 기준
- 생성 노드 분기

세 가지에 모두 영향을 준다.

## 7. `action_state`는 구체적으로 무엇에 영향을 주나

### 1. 이번 턴의 사건 타입을 고정한다

`action_state`는 현재 턴이 다음 중 무엇인지 정한다.

- 대기/정체 턴
- 이벤트 발생 턴
- 갈등 진행 턴
- 해소 턴
- 후일담 턴
- sexual 단계 턴

즉 모델은 지금 "사건을 전개해야 하는지", "정리해야 하는지", "강도를 높여야 하는지"를 `action_state`를 보고 판단한다.

### 2. sexual 프롬프트 단계 힌트를 바꾼다

`action_state`가 `SEXUAL_*` 계열이면 `stage_rules_map` 같은 단계별 규칙이 프롬프트에 추가된다.

이건 sexual 경로에서 특히 중요하다.  
동일한 sexual 턴이어도 `SEXUAL_1`과 `SEXUAL_4`는 전개 강도와 허용되는 진행 방식이 다르기 때문이다.

### 3. sexual 전용 생성 노드를 선택하게 만든다

`action_state`가 `SEXUAL_*`면 sexual 생성 경로로,  
`AFTERMATH_SEX_*`면 그 후속 정리 경로로 갈 수 있다.

즉 `action_state`는 생성 노드 선택에 직접 사용된다.

### 4. 행동 상태 이력에 누적된다

`action_state_hist`에 누적되어 이후 "같은 전개 반복 방지", "단계 유지", "후속 전환" 같은 규칙에 재사용된다.

즉 `action_state`는 한 턴짜리 상태가 아니라 다음 턴에도 영향을 남긴다.

## 8. 둘은 서로 직접 바꾸나

여기서 중요한 점이 있다.

`fsm_state`와 `action_state`는 보통 "서로를 직접 변경하는 상태 변수"는 아니다.  
즉 관계 FSM 내부에서 행동 FSM 상태를 덮어쓰거나, 반대로 행동 FSM 내부에서 관계 FSM 상태를 직접 바꾸는 구조는 아니다.

대신 실제 영향 방식은 다음과 같다.

### 1. 같은 입력 신호를 공유한다

두 FSM은 같은 `EVAL_JSON`에서 파생된 신호를 읽는다.  
그래서 한 턴에서 위협, 압박, 친밀도 변화가 크면 두 FSM이 동시에 영향을 받는다.

예:

- 위협이 커짐
  - 관계 FSM은 위기 국면으로 기울 수 있음
  - 행동 FSM은 갈등/사건 턴으로 기울 수 있음

### 2. 생성 단계에서 함께 결합된다

프롬프트에는 `fsm_state`와 `action_state`가 동시에 들어간다.  
즉 모델은 "지금 국면은 위기인데, 이번 턴 사건은 해소 단계다" 같은 복합 정보를 함께 받는다.

그래서 실제 출력은 둘의 조합으로 결정된다.

### 3. 상위 국면이 행동 선택의 적절성을 간접 제약한다

`fsm_state`가 위기 국면이면, `action_state`가 같은 `EVENT`라도 생성 결과는 더 날카롭고 긴장된 사건 전개가 된다.

반대로 `fsm_state`가 안정 국면이면 같은 `EVENT`라도 더 가벼운 사건 전개로 갈 수 있다.

즉 `fsm_state`는 `action_state`의 해석 맥락을 바꾼다.

### 4. 행동 상태가 관계 상태를 간접적으로 강화한다

`action_state`가 갈등, sexual progression, aftermath 등을 반복적으로 밟으면, 다음 턴 `EVAL_JSON`의 신호가 달라지고 결국 관계 FSM도 다른 방향으로 이동할 가능성이 커진다.

즉 직접 대입은 아니지만:

- `action_state`가 출력과 서사를 바꾸고
- 그 출력이 다음 턴 신호를 바꾸고
- 그 신호가 다시 `fsm_state`를 움직인다

는 식의 간접 상호작용이 있다.

## 9. 특히 중요한 차이: `action_state`는 후처리된다

`fsm_state`는 관계 FSM에서 나온 상태가 비교적 직접적으로 쓰인다.  
하지만 `action_state`는 그렇지 않다.

`fsm_step_node`에서는 행동 FSM의 원래 상태(`base_action_state`)를 받은 뒤, 추가 규칙으로 최종 `action_state`를 다시 조정한다.

대표 규칙:

1. sexual 단계 최소 턴 유지
2. sexual aftermath 단계 전환
3. `sexual_ready=False`면 강등
4. 너무 오래 `IDLE`이면 정체 해소
5. 최근 이력을 보고 다양화

즉 최종 `action_state`는:

- "행동 FSM의 원래 판단"
- "서사 품질을 위한 후처리"

가 합쳐진 결과다.

이 점이 `fsm_state`와 가장 큰 차이다.

## 10. 왜 두 상태를 굳이 분리하나

둘을 하나로 합치면 문제가 생긴다.

예를 들어:

- 관계는 위기 국면인데 이번 턴은 해소 행동을 해야 할 수 있고
- 관계는 안정 국면인데 이번 턴엔 갑작스러운 이벤트가 일어날 수 있다

즉:

- 상위 국면
- 현재 사건 단계

는 같은 정보가 아니다.

그래서 분리하면 다음이 가능해진다.

- 같은 `action_state=EVENT`라도 `fsm_state`에 따라 톤을 다르게 생성
- 같은 `fsm_state=CRISIS`라도 `action_state`에 따라 갈등/해소/후일담을 구분
- sexual 경로를 세밀한 단계로 제어하면서도 상위 서사 국면은 별도로 유지

## 11. 실제 턴 흐름 예시

한 턴의 큰 흐름은 보통 다음처럼 이해하면 된다.

1. 이전 대화와 현재 출력 후보를 바탕으로 `EVAL_JSON` 신호 생성
2. 관계 FSM이 신호를 받아 `fsm_state` 갱신
3. 행동 FSM이 신호를 받아 `base_action_state` 갱신
4. 후처리 규칙으로 최종 `action_state` 확정
5. `fsm_state`와 `action_state`를 프롬프트에 주입
6. 상태 조합에 따라 일반/위기/sexual 생성 경로 선택
7. 생성 결과를 검사하고 이력 업데이트
8. 다음 턴에서 다시 같은 과정 반복

## 12. 실전 해석법

코드를 읽을 때는 아래처럼 해석하면 가장 이해가 쉽다.

- `fsm_state`
  - 지금 관계와 서사의 상위 국면
  - 분위기, 위기, 긴장도, 진행 레일

- `action_state`
  - 이번 턴의 구체적인 사건 단계
  - 이벤트, 갈등, 해소, aftermath, sexual progression

둘의 관계는:

- 같은 신호를 공유하지만 역할은 다르고
- 직접 값을 덮어쓰기보다는
- 프롬프트와 분기에서 함께 결합되어 최종 출력을 만든다

## 13. 최종 정리

가장 짧게 정리하면 이렇다.

- `fsm_state`는 "지금 어떤 국면인가"를 정한다.
- `action_state`는 "이번 턴에 무엇을 할 것인가"를 정한다.
- 둘은 같은 신호를 공유하지만 다른 목적의 FSM을 돈다.
- `fsm_state`는 상위 국면, 위기 여부, 출력 적합성 검사에 강하게 작용한다.
- `action_state`는 현재 사건 단계, sexual 단계, 생성 노드 선택에 강하게 작용한다.
- 둘은 직접 서로를 바꾸기보다, 같은 생성 프롬프트와 분기 로직 안에서 결합되어 실제 텍스트 전개를 만든다.
