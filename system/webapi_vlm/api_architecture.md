# VLM WebAPI Architecture

이 문서는 현재 `system/webapi_vlm` 폴더의 실제 코드 기준 아키텍처 문서다.

대상 파일:
- `system/webapi_vlm/app.py`
- `system/webapi_vlm/schemas.py`
- `system/webapi_vlm/services.py`
- `system/webapi_vlm/demo.py`

관련 코어 모듈:
- `system/vlm_engine.py`
- `system/vlm_langgraph_pipeline.py`
- `system/memory_chain.py`
- `system/prompt_compiler.py`
- `system/rp_parser.py`
- `system/translator.py`
- `system/tts_worker_client.py`

---

## 1. 목적

`system/webapi_vlm`은 Kanana 기반 VLM RP 모델을 별도 REST API와 Gradio UI로 노출하는 서버 계층이다.

이 경로는 `system/webapi`와 분리된 독립 런타임이다.

현재 목적:
1. 이미지 없는 일반 RP 대화 지원
2. 이미지 포함 멀티모달 RP 대화 지원
3. LangGraph + memory chain 기반 턴 처리
4. Qwen/vLLM 경로와 상태를 섞지 않는 독립 서버 제공

---

## 2. 구성 요약

## 2.1 `app.py`

VLM 전용 FastAPI 엔트리포인트다.

역할:
- CORS 설정
- REST 라우트 선언
- 예외를 `HTTPException(500)`로 변환
- `/demo` Gradio 마운트

실행 예:

```bash
uv run uvicorn system.webapi_vlm.app:app --host 0.0.0.0 --port 8001
```

## 2.2 `schemas.py`

VLM 전용 요청/응답 스키마 정의다.

텍스트 경로와의 차이:
- `ChatRequest.image_path`
- `TurnRequest.image_path`

즉 VLM은 image upload 경로를 API 계약에 포함한다.

## 2.3 `services.py`

VLM 런타임 컨테이너다.

핵심 객체:
- `_vlm`
- `_translator`
- `_tts`
- `_prompt_compiler`
- `_memory_chain`
- `_turn_graph`
- `_history`

## 2.4 `demo.py`

VLM 전용 Gradio UI다.

특징:
- VN 스타일 stage
- 이미지 업로드 입력 포함
- 감정 one-hot 기반 표정 이미지 선택
- debug/transcript 패널 제공

---

## 3. API 구성

### 3.1 단일 기능 endpoint

- `POST /api/chat`
  - 이미지 선택적 포함
  - VLM 응답 원문 생성만 수행

- `POST /api/parse`
  - RP 원문 파싱

- `POST /api/translate`
  - 한국어 대사 -> 일본어 번역

- `POST /api/tts`
  - 일본어 대사 -> WAV

### 3.2 통합 endpoint

- `POST /api/turn`
- `POST /api/main-loop`

두 endpoint는 동일한 LangGraph 기반 턴 파이프라인을 탄다.

---

## 4. 런타임 구조

`RuntimeServices`는 모델/번역기/TTS를 lazy-load한다.

핵심 락:
- `_load_lock`
- `_turn_lock`

history:
- `deque(maxlen=6)`

즉 최근 3턴만 short-term memory로 유지한다.

장기 기억은 별도 `SummaryMemoryChain`가 담당한다.

---

## 5. VLM 메인 파이프라인

`services.turn()`은 `build_vlm_turn_graph()`로 생성된 LangGraph를 호출한다.

그래프 노드:
1. `prepare_prompt`
2. `route_has_image`
3. `generate_text_only` 또는 `generate_with_image`
4. `translate`
5. `emotion`
6. `tts`
7. `commit`

이 경로는 텍스트-only와 이미지 포함 멀티모달을 같은 그래프 안에서 분기한다.

---

## 6. 이미지 유무 분기

핵심 분기 조건:
- `image_path`가 비어 있으면 `generate_text_only`
- `image_path`가 있으면 `generate_with_image`

### 6.1 이미지 없는 경우

일반 RP 대화처럼 동작한다.

프롬프트 순서:
1. 기본 RP system
2. memory system message
3. recent history
4. current user message

### 6.2 이미지 있는 경우

이미지 해석 포함 RP 대화로 동작한다.

프롬프트 순서:
1. 기본 RP system
2. memory system message
3. image guidance system
4. recent history
5. current user message

image guidance는 이미지 해석 규칙을 system prompt로 넣는다.

규칙 요약:
- 이미지에서 보이는 장면/분위기/인물 상태를 먼저 파악
- 이미지에 없는 사실은 단정하지 않음
- 이미지 묘사를 나열하지 않고 RP 반응 속에 녹임

---

## 7. 메모리 체인

현재 `webapi_vlm`도 `SummaryMemoryChain`를 사용한다.

즉 현재는 `qwen` 경로와 대칭이다.

### 7.1 주입 위치

`prepare_prompt`에서 `memory_chain.build_memory_system_message(current_user_text=...)`를 호출해 system message를 추가한다.

### 7.2 업데이트 위치

`commit`에서:
- history append
- `memory_chain.update(user_text=..., assistant_text=...)`

### 7.3 chat endpoint도 메모리 사용

`services.chat()`도:
- memory retrieval 주입
- 응답 후 memory update

를 수행한다.

즉 `turn()`만이 아니라 `chat()`도 장기 기억을 탄다.

### 7.4 디버그

실제 메모리 주입 확인용 로그를 켤 수 있다.

```bash
export VLM_MEMORY_DEBUG=1
uv run uvicorn system.webapi_vlm.app:app --host 0.0.0.0 --port 8001
```

로그 예:
- `[vlm-memory] chat inject: ...`
- `[vlm-memory] turn inject: ...`

이 로그가 뜨면 retrieval message가 실제 prompt에 들어간 것이다.

---

## 8. VLM 엔진 구조

VLM 런타임은 `system/vlm_engine.py`의 `KananaVLMEngine`를 사용한다.

현재 동작:
- `AutoProcessor`
- `AutoModelForVision2Seq`
- image file load 또는 dummy image
- Kanana conv 포맷으로 prompt 구성
- 전체 디코드 후 마지막 assistant 구간 추출

중요한 점:
- 현재 이 경로는 vLLM을 사용하지 않는다.

---

## 9. vLLM과의 관계

현재 프로젝트에 설치된 vLLM은 `0.14.0`이다.

하지만 `saya_vlm_3b`는 로컬 config 기준:
- `architectures = ["KananaVForConditionalGeneration"]`
- `model_type = "kanana-1.5-v"`
- `auto_map` 기반 custom code 모델

즉 vLLM이 기본 지원하는 표준 멀티모달 모델이 아니다.

현재 결론:
- `webapi_vlm`은 vLLM 경로를 사용하지 않음
- 단순 backend 교체로 vLLM에 바로 꽂을 수 있는 상태도 아님
- 별도 vLLM custom model integration이 필요할 가능성이 높음

즉 `webapi_vlm`은 현재 Hugging Face 멀티모달 런타임이다.

---

## 10. 감정 상태 매핑

감정 생성:
- `vlm_langgraph_pipeline.py`의 `node_emotion`
- 우선 `llm_engine.infer_emotion_json()`
- 실패 시 keyword fallback

응답 구조:

```json
{
  "neutral": 1,
  "sad": 0,
  "happy": 0,
  "angry": 0
}
```

UI 반영:
- `_normalize_emotion()`
- `_pick_char_b64()`

즉 backend의 one-hot 감정을 받아 표정 이미지 선택에 사용한다.

---

## 11. Gradio 데모 구조

`demo.py`는 REST 소비자다.

즉 내부 서비스를 직접 호출하지 않고 `httpx`로 `/api/main-loop`를 호출한다.

구성 요소:
- stage 배경
- 캐릭터 전신 이미지
- 대사 패널
- 텍스트 입력
- 이미지 업로드
- voice 선택
- 실행 버튼
- transcript/debug 패널

메인 UX:
1. 사용자 입력 + 선택적 이미지 업로드
2. `/api/main-loop` 호출
3. narration 먼저 표시
4. dialogue_ko 타자 효과
5. 최종 프레임에서 표정/오디오/history 반영

---

## 12. 환경변수

주요 환경변수:

- `WEBAPI_VLM_ENABLE_GRADIO`
  - `/demo` 활성 여부

- `WEBAPI_VLM_BASE_URL`
  - demo가 호출할 REST base URL

- `KANANA_VLM_MODEL_DIR`
  - VLM 모델 경로

- `KANANA_VLM_LOAD_IN_4BIT`
  - 4bit 로딩 여부

- `KANANA_VLM_TRUST_REMOTE_CODE`
  - custom code 허용 여부

- `KANANA_VLM_ATTN_IMPL`
  - attention implementation

- `KANANA_VLM_MAX_LENGTH`
- `KANANA_VLM_DUMMY_IMAGE_SIZE`

- `TRANS_MODEL_DIR`
  - 번역 모델 경로

- `VLM_MEMORY_DEBUG`
  - memory retrieval injection 로그 출력

---

## 13. 안정성 설계

현재 구조는 Qwen 경로와 섞이지 않도록 별도 FastAPI 앱/서비스를 둔다.

핵심 원칙:
- Qwen 런타임과 VLM 런타임 분리
- VLM 내부 상태는 `_turn_lock`으로 직렬화
- history/memory는 동일 서비스 인스턴스 내부에서만 관리

즉 single-process 내부 일관성을 우선하는 구조다.

---

## 14. 요약

`system/webapi_vlm`은 현재 다음 조합으로 동작한다.

- Kanana VLM 엔진
- 이미지 유무 분기 LangGraph
- SummaryMemoryChain
- RP 파싱
- 번역
- 감정 판정
- TTS
- REST API + Gradio

현재는 `qwen webapi`와 구조적으로 대칭이지만,
엔진 backend는 vLLM이 아니라 Hugging Face 멀티모달 경로라는 점이 가장 큰 차이다.
