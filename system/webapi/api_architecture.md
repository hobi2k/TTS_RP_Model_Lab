# Qwen WebAPI Architecture

이 문서는 현재 `system/webapi` 폴더의 실제 코드 기준 아키텍처 문서다.

대상 파일:
- `system/webapi/app.py`
- `system/webapi/schemas.py`
- `system/webapi/services.py`
- `system/webapi/demo.py`

관련 코어 모듈:
- `system/llm_engine.py`
- `system/langgraph_pipeline.py`
- `system/memory_chain.py`
- `system/prompt_compiler.py`
- `system/rp_parser.py`
- `system/translator.py`
- `system/tts_worker_client.py`

---

## 1. 목적

`system/webapi`는 Qwen 계열 텍스트 RP 모델을 REST API와 Gradio UI로 노출하는 서버 계층이다.

현재 목표는 세 가지다.

1. 텍스트 RP 턴을 안정적으로 실행
2. history + long-term memory를 포함한 메인 파이프라인 제공
3. 디버그 가능한 단일 기능 API와 실제 시연용 통합 API를 동시에 제공

---

## 2. 구성 요약

## 2.1 `app.py`

FastAPI 엔트리포인트다.

역할:
- CORS 설정
- REST 라우트 선언
- 예외를 `HTTPException(500)`로 변환
- `/demo` Gradio 마운트

## 2.2 `schemas.py`

Pydantic 요청/응답 스키마 정의다.

핵심 모델:
- `ChatRequest`, `ChatResponse`
- `TranslateRequest`, `TranslateResponse`
- `ParseRequest`, `ParseResponse`
- `TTSRequest`, `TTSResponse`
- `TurnRequest`, `TurnResponse`
- `EmotionState`

## 2.3 `services.py`

실제 런타임 컨테이너다.

핵심 클래스:
- `RuntimeServices`

역할:
- 모델 지연 로딩
- prompt/history/memory 조립
- LangGraph 파이프라인 실행
- 단일 기능 메서드 제공

## 2.4 `demo.py`

Gradio 기반 VN 스타일 데모 UI다.

역할:
- `/api/main-loop` 호출
- 타자 효과 출력
- 감정 one-hot 기반 표정 이미지 선택
- transcript/debug 정보 표시

---

## 3. API 구성

현재 endpoint는 두 층으로 나뉜다.

### 3.1 단일 기능 endpoint

- `POST /api/chat`
  - RP 원문 생성만 수행

- `POST /api/parse`
  - RP 원문 파싱만 수행

- `POST /api/translate`
  - 한국어 대사를 일본어로 번역

- `POST /api/tts`
  - 일본어 대사를 음성으로 합성

### 3.2 통합 endpoint

- `POST /api/turn`
- `POST /api/main-loop`

두 endpoint는 현재 같은 파이프라인을 호출한다.

실행 순서:
- input
- prompt assembly
- memory injection
- LLM generation
- RP parse
- translate
- emotion
- TTS
- history update
- memory update

---

## 4. 런타임 서비스 구조

`RuntimeServices`는 무거운 객체를 lazy-load한다.

주요 내부 필드:
- `_llm`
- `_translator`
- `_tts`
- `_prompt_compiler`
- `_memory_chain`
- `_turn_graph`
- `_history`

동시성 제어:
- `_load_lock`
  - 중복 로딩 방지
- `_turn_lock`
  - 턴 실행 직렬화

`_history`는 `deque(maxlen=6)`이다.
즉 user/assistant 2개 메시지 기준 최근 3턴만 유지한다.

---

## 5. 메인 파이프라인

`services.turn()`은 직접 모든 단계를 구현하지 않고 `LangGraph`를 호출한다.

실제 그래프는 `system/langgraph_pipeline.py`의 `build_turn_graph()`에서 만든다.

노드 순서:
1. `prepare_prompt`
2. `generate_parse`
3. `translate`
4. `emotion`
5. `tts`
6. `commit`

### 5.1 `prepare_prompt`

메시지 배열을 조립한다.

순서:
1. 캐릭터 system prompt
2. memory system message
3. recent history
4. current user message

### 5.2 `generate_parse`

`QwenEngine`로 prompt를 생성하고, RP 텍스트를 뽑은 뒤 `RPParser`로 구조화한다.

### 5.3 `translate`

정규화된 한국어 대사를 일본어로 번역한다.

### 5.4 `emotion`

우선 `llm_engine.infer_emotion_json()`을 시도하고, 실패하면 키워드 기반 fallback으로 one-hot 감정을 만든다.

### 5.5 `tts`

일본어 대사가 있으면 SBV2 워커로 합성한다.

### 5.6 `commit`

여기서 상태가 실제로 저장된다.

- history append
- memory chain update

즉 메모리 체인은 턴이 끝난 뒤 commit 시점에 갱신된다.

---

## 6. 메모리 체인

`webapi`는 현재 `SummaryMemoryChain`를 사용한다.

구성:
- SQLite 저장
- 후보 기억 추출
- 점수화
- slot 승격
- retrieval
- system message 주입

프롬프트 주입 위치:
- `prepare_prompt` 노드에서 `prompt_compiler.compile()` 뒤에 memory system message 추가

업데이트 위치:
- `commit` 노드에서 `memory_chain.update(...)`

즉 현재 `webapi` 경로는
- short-term history
- long-term memory

둘 다 사용한다.

---

## 7. Qwen 엔진 특성

현재 `QwenEngine`는 프로젝트 설정에 따라 vLLM 또는 Hugging Face 경로를 사용할 수 있다.

이 `webapi` 경로는 "vLLM을 쓸 수 있는 텍스트 causal LM 경로"라는 점이 `webapi_vlm`과 가장 큰 차이다.

정리:
- 입력: text-only
- 엔진: `QwenEngine`
- backend: vLLM/HF 가능

---

## 8. 응답 계약

통합 응답은 `TurnResponse`를 따른다.

필드:
- `rp_text`
- `narration`
- `dialogue_ko`
- `dialogue_ja`
- `wav_path`
- `emotion`

`emotion`은 one-hot 구조다.

```json
{
  "neutral": 1,
  "sad": 0,
  "happy": 0,
  "angry": 0
}
```

이 구조는 UI 쪽 표정 렌더링에 바로 사용된다.

---

## 9. Gradio 데모 구조

`demo.py`는 내부 함수를 직접 호출하지 않고 `httpx`로 REST endpoint를 호출한다.

즉 이 데모는 단순 view가 아니라 API 소비자다.

장점:
- 실제 운영 경로와 동일한 계약 검증
- UI와 백엔드 분리 유지
- 이후 프론트엔드 교체 시 API 재사용 가능

### 메인 UX 흐름

`mainloop_turn()`은 generator 기반이다.

순서:
1. `/api/main-loop` 호출
2. narration만 먼저 렌더
3. dialogue_ko 타자 효과 출력
4. 최종 프레임에서 wav/autoplay/history 반영

감정 상태는 `_normalize_emotion()` -> `_pick_char_b64()`로 이어져 표정 이미지를 선택한다.

---

## 10. 환경변수

주요 환경변수:

- `WEBAPI_ENABLE_GRADIO`
  - `1`이면 `/demo` 활성

- `WEBAPI_BASE_URL`
  - demo가 REST를 호출할 base URL

- `QWEN_MODEL_DIR`
  - 텍스트 RP 모델 경로

- `TRANS_MODEL_DIR`
  - 번역 모델 경로

---

## 11. 안정성 설계 포인트

이 경로는 상태를 공유하므로 동시 실행보다 일관성을 우선한다.

핵심 원칙:
- 모델 로딩은 lazy
- 턴 실행은 직렬화
- SQLite는 `check_same_thread=False`지만 실질 동시 접근은 `_turn_lock`로 막음

즉 "빠른 병렬 응답"보다 "history/memory 불일치 방지"가 더 우선인 구조다.

---

## 12. 요약

`system/webapi`는 현재 다음 조합으로 동작한다.

- Qwen 텍스트 LLM
- LangGraph 턴 파이프라인
- SummaryMemoryChain
- RP 파싱
- 번역
- 감정 판정
- TTS
- REST API + Gradio

즉 텍스트 RP 메인 경로의 기준 구현이다.

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
