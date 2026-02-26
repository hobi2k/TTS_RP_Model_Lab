# FastAPI RESTful API + Demo Architecture Guide

이 문서는 `system/webapi` 아키텍처 문서다.

대상 코드:
- `system/webapi/app.py`
- `system/webapi/schemas.py`
- `system/webapi/services.py`
- `system/webapi/demo.py`

---

## 1. 먼저 이해해야 할 FastAPI 핵심 개념

FastAPI는 크게 네 가지로 이해하면 된다.

1. **Path Operation(라우트 함수)**
- `@app.get`, `@app.post` 데코레이터로 URL과 메서드를 정의한다.
- 함수는 HTTP 요청을 받아 비즈니스 로직을 실행하고 응답을 반환한다.

2. **Pydantic 모델(요청/응답 스키마)**
- 요청 바디 검증과 응답 형식 보장을 담당한다.
- 잘못된 입력을 초기에 차단하고, API 계약을 명확히 만든다.

3. **의존 서비스 계층 분리**
- 라우트 안에서 모델 로딩, 상태 관리, 파이프라인 조합까지 직접 처리하면 유지보수가 어렵다.
- 그래서 라우트는 얇게 유지하고, 실제 처리는 별도 서비스 클래스(`RuntimeServices`)가 담당한다.

4. **OpenAPI 자동 문서화**
- 스키마를 잘 설계하면 `/docs`에 자동으로 API 명세가 생성된다.
- 협업과 테스트 효율이 크게 올라간다.

---

## 2. RESTful 관점에서 이 프로젝트를 어떻게 나눴는가

이 프로젝트는 "기능 단위 endpoint"와 "통합 endpoint"를 함께 제공하는 방식으로 설계되었다.

- 기능 단위 endpoint
  - `/api/chat`: RP 응답 생성만
  - `/api/parse`: 파싱만
  - `/api/translate`: 번역만
  - `/api/tts`: 음성합성만

- 통합 endpoint
  - `/api/turn` (`/api/main-loop` 동일):
    - 입력 → LLM 생성 → 파싱 → 번역 → TTS → 감정판정 → 메모리 업데이트

이렇게 나누는 이유는 단순하다.
- 디버깅/테스트는 단일 기능 endpoint가 유리하다.
- 실제 제품 동작은 통합 endpoint가 유리하다.

즉, "개발 효율"과 "운영 동선"을 동시에 맞춘 구조다.

---

## 3. 계층 설계: 왜 `app.py`와 `services.py`를 분리했는가

### 3.1 `app.py`의 책임

`app.py`는 HTTP 레이어만 담당한다.

- 라우트 선언
- 요청/응답 스키마 연결
- 예외를 HTTP 에러로 변환
- CORS/리다이렉트/데모 마운트 같은 웹 설정

중요한 점은, `app.py`가 "모델 내부 로직"을 거의 모른다는 것이다.
이 원칙을 지켜야 API 구조를 오래 유지할 수 있다.

### 3.2 `services.py`의 책임

`RuntimeServices`는 도메인 실행 엔진이다.

- LLM/Translator/TTS 지연 로딩
- 프롬프트 조립
- short-term history + long-term memory retrieval 주입
- 응답 파싱/번역/TTS 순차 실행
- 감정 JSON 판정
- 상태 업데이트

즉, 라우트 함수는 "입력 전달 + 결과 반환"만 하고,
실제 업무 로직은 전부 `RuntimeServices`에 집중한다.

---

## 4. 요청/응답 설계 방법 (`schemas.py`)

이 프로젝트 스키마 설계의 핵심 원칙은 두 가지다.

1. **요청에 경계값을 둔다**
- 예: `text`는 `min_length=1`
- 예: `temperature`, `top_p`, `top_k` 범위 제한

2. **응답을 고정 구조로 만든다**
- 예: `TurnResponse`는 항상 `rp_text`, `narration`, `dialogue_ko`, `dialogue_ja`, `wav_path`, `emotion` 필드를 제공
- 클라이언트가 분기 로직을 단순하게 작성할 수 있다.

`EmotionState`를 별도 모델로 둔 것도 같은 이유다.
- one-hot 규약을 API 레벨에서 드러내기 쉽다.

---

## 5. 실제 메인 파이프라인 구현 흐름 (`RuntimeServices.turn`)

`turn()`은 사실상 main_loop 서버 버전이다. 실행 순서는 아래와 같다.

1. `self._turn_lock` 획득
- history/memory/DB 업데이트 충돌을 방지하기 위해 턴 단위 직렬화

2. 컴포넌트 보장
- `_ensure_llm()`
- `_ensure_translator()`
- `_ensure_tts()`
- `_ensure_mainloop_components()`

3. 프롬프트 메시지 구성
- 캐릭터 system 프롬프트
- memory system 메시지(장기기억 retrieval 결과)
- 최근 history
- 현재 user 입력

4. LLM 생성
- `raw_text`

5. RP 파싱
- narration / dialogue 분리

6. 번역기 입력 정규화
- 대사는 "한 줄"로 정규화해서 번역기로 전달

7. 번역 + TTS
- `dialogue_ko -> dialogue_ja -> wav_path`

8. 감정 판정
- 1차: LLM JSON one-hot
- 실패 시: 키워드 fallback

9. 상태 업데이트
- short-term history append
- memory chain update

10. `TurnResponse` 형태 반환

이 순서를 고정한 이유는, 출력 품질보다 먼저 **재현성**을 지키기 위해서다.
(같은 입력에서 단계가 바뀌면 결과와 디버깅이 흔들린다.)

---

## 6. 동시성과 안정성: 실제 운영에서 중요한 부분

웹 환경에서는 같은 API가 동시에 들어온다. 이때 깨지기 쉬운 지점은 상태 공유다.

이 프로젝트가 택한 방법:

- `RuntimeServices` 단일 인스턴스 사용
- `_load_lock`
  - 모델 중복 로딩 방지
- `_turn_lock`
  - turn 파이프라인 직렬화
- SQLite는 `check_same_thread=False`
  - 단, 무제한 병렬 접근이 아니라 상위 lock 직렬화 전제를 둠

즉, "DB 옵션으로 병렬 처리"가 아니라 "상위 파이프라인 직렬 제어"로 안정성을 확보했다.

---

## 7. 에러 처리 전략

라우트 함수는 try/except로 서비스 예외를 잡아 `HTTPException(500)`로 변환한다.

예:
- `/api/main-loop` 실패 시
  - `detail="main-loop failed: ..."`

`demo.py`는 이 500을 다시 잡아서 화면용 오류 프레임으로 바꾼다.
- 사용자에게는 VN 대화창 안에서 오류를 표시
- 개발자는 서버 로그에서 원인 확인

즉, API와 UI가 각각 자기 레이어에서 에러를 처리한다.

---

## 8. Gradio 데모를 "REST 소비자"로 붙인 이유

`/demo`는 내부 함수 직접호출이 아니라 `httpx`로 REST endpoint를 호출한다.

이렇게 구현한 이유:
- 데모와 실제 API를 분리된 계약으로 유지
- 데모가 곧 API 통합 테스트 역할
- 나중에 프론트엔드를 Gradio에서 React/Unity로 바꿔도 API는 그대로 재사용 가능

즉, 데모를 백엔드 내부 함수 호출기로 만들지 않고, **클라이언트처럼** 동작시켜 구조를 검증한다.

---

## 9. 메인루프 시연 UX 구현 포인트 (`demo.py`)

`mainloop_turn()`은 generator로 구현되어 단계적 렌더링을 한다.

1. 첫 프레임
- narration 표시
- 대사는 비움

2. 타자 효과 프레임
- `dialogue_ko`를 글자 단위로 점진 출력

3. 최종 프레임
- 전체 텍스트 확정
- 오디오 경로 반영(autoplay)
- transcript/history 갱신

감정 이미지도 이 시점에 적용된다.
- angry/sad/happy/neutral one-hot에 맞는 이미지 선택

---

## 10. 환경변수 설계

주요 환경변수:

- `WEBAPI_ENABLE_GRADIO` (기본 1)
  - 1: `/demo` 활성
  - 0: REST 전용

- `WEBAPI_BASE_URL`
  - demo가 호출할 API base URL

- `QWEN_MODEL_DIR`
  - RP LLM 경로 오버라이드

- `TRANS_MODEL_DIR`
  - 번역 모델 경로 오버라이드

경로를 환경변수로 열어둔 이유:
- 로컬/서버/배포 환경 차이에 대응
- 코드 수정 없이 모델 경로 교체 가능

---

## 11. 현재 구조 구현 방법

아래 순서로 구현하면 같은 아키텍처를 구현할 수 있다.

1. `schemas.py`부터 설계
- request/response 계약 확정

2. `services.py` 작성
- 모델 로딩, 파이프라인, 상태 관리, lock

3. `app.py` 작성
- 엔드포인트를 서비스 메서드에 매핑
- 예외를 HTTPException으로 통일

4. API 단위 테스트
- `/api/chat`, `/api/translate`, `/api/tts`를 먼저 검증

5. `/api/turn` 통합 검증
- end-to-end 결과 확인

6. `demo.py` 연결
- REST 호출 기반으로 UI를 붙임

이 순서가 중요한 이유:
- UI부터 만들면 에러 원인 분리가 어려워진다.
- API 계약부터 고정해야 디버깅 비용이 줄어든다.

---

## 12. 확장 시 권장 방향

1. 세션 분리
- 현재 기본 session_id는 단일값 기반
- 사용자별 세션 키를 API에 도입하면 멀티유저 메모리 분리 가능

2. 인증/권한
- 운영 환경이라면 API Key/JWT 추가

3. 비동기 작업 큐
- TTS 장시간 작업을 background queue로 분리 가능

4. 관측성 강화
- 요청 ID, 단계별 latency, 실패 원인 로깅 표준화

5. 계약 고정
- OpenAPI를 기준으로 클라이언트/서버 버전 관리

---

## 13. 핵심 요약

이 구조의 본질은 다음 한 줄이다.

- **FastAPI는 계약과 라우팅을 담당하고, RuntimeServices가 실제 파이프라인을 담당하며, Gradio는 REST 클라이언트로 동작한다.**

이 원칙을 유지하면,
- 디버깅이 쉬워지고
- 교체(모델/UI)가 쉬워지고
- 운영 안정성이 올라간다.
