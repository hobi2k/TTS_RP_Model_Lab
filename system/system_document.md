# System Folder Document

이 문서는 `TTS_RP_Model_Lab/system` 폴더 전체의 현재 구현 상태를 기준으로 구조, 역할, 호출 흐름, 파일별 기능을 정리한 문서다.

기준 날짜:
- 2026-03-10

대상 범위:
- `system/*.py`
- `system/utils/*.py`
- `system/webapi/*`
- `system/webapi_vlm/*`

제외:
- `__pycache__`
- 생성된 `.pyc`

---

## 1. 폴더 전체 목적

`system` 폴더는 이 프로젝트의 "대화 런타임 계층"이다.

역할은 크게 네 가지다.

1. 텍스트 RP 모델과 VLM RP 모델을 감싸는 추론 엔진 제공
2. RP 원문을 파싱하고 번역/TTS/감정 판정을 연결하는 턴 파이프라인 제공
3. 장기 기억(memory chain)과 단기 history를 관리하는 상태 계층 제공
4. FastAPI + Gradio 데모 형태로 외부에 API/UI를 노출

현재 구조는 두 개의 런타임 경로로 나뉜다.

- `webapi`
  - Qwen 계열 텍스트 RP 경로
  - `QwenEngine` 사용
  - LangGraph 사용
  - SummaryMemoryChain 사용
  - vLLM 사용 가능 경로

- `webapi_vlm`
  - Kanana 기반 VLM RP 경로
  - `KananaVLMEngine` 사용
  - VLM 전용 LangGraph 사용
  - SummaryMemoryChain 사용
  - 이미지 입력 지원
  - 현재는 Hugging Face `AutoModelForVision2Seq` 경로 사용, vLLM은 사용하지 않음

---

## 2. 최상위 구조 요약

### 2.1 공통 코어 모듈

- `llm_engine.py`
  - Qwen 기반 텍스트 LLM 엔진
- `vlm_engine.py`
  - Kanana 기반 VLM 엔진
- `prompt_compiler.py`
  - 캐릭터 system prompt 생성
- `rp_parser.py`
  - RP 원문을 narration/action/dialogue로 구조화
- `translator.py`
  - 한국어 대사를 일본어로 번역
- `tts_worker_client.py`
  - SBV2 워커 기반 TTS 호출
- `memory_chain.py`
  - SQLite 기반 장기 기억 관리

### 2.2 파이프라인 모듈

- `langgraph_pipeline.py`
  - Qwen 텍스트 경로용 LangGraph 턴 파이프라인
- `vlm_langgraph_pipeline.py`
  - VLM 경로용 LangGraph 턴 파이프라인

### 2.3 유틸리티

- `utils/text_split.py`
  - 문장 분리 유틸리티

### 2.4 API/UI 계층

- `webapi/`
  - Qwen 텍스트 경로용 REST API + Gradio UI
- `webapi_vlm/`
  - VLM 경로용 REST API + Gradio UI

---

## 3. 공통 처리 흐름

대체로 모든 경로는 아래 단계 중 일부 또는 전부를 공유한다.

1. 사용자 입력 수신
2. system prompt 구성
3. memory system message retrieval
4. recent history 결합
5. 모델 응답 생성
6. RP 파싱
7. 한국어 대사 정규화
8. 일본어 번역
9. 감정 one-hot 판정
10. TTS 합성
11. history 저장
12. memory update

Qwen 경로와 VLM 경로의 차이는 "모델 입력 형태"와 "생성 노드"에 있다.

- Qwen:
  - `messages -> build_prompt() -> generate()`
- VLM:
  - `messages -> generate_from_messages(image_path=...)`
  - 이미지 유무에 따라 그래프 분기

---

## 4. 파일별 상세 설명

## 4.1 `llm_engine.py`

### 목적

Qwen 계열 causal LM을 감싸는 텍스트 추론 엔진이다.

### 핵심 구성

- `GenerationConfig`
  - 생성 파라미터 묶음
  - `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty` 등 포함

- `QwenEngine`
  - 모델 로딩
  - prompt 생성
  - 텍스트 생성
  - 감정 JSON 판정

### 주요 책임

1. base model 경로 또는 ID 로딩
2. backend 선택
   - vLLM
   - Hugging Face Transformers
3. OpenAI-style message list를 텍스트 prompt로 변환
4. RP 응답 생성
5. 감정 JSON one-hot 추정

### 현재 성격

이 파일은 `system/webapi` 경로의 핵심 엔진이다.

---

## 4.2 `vlm_engine.py`

### 목적

Kanana 기반 VLM 모델을 감싸는 멀티모달 추론 엔진이다.

### 핵심 구성

- `VLMGenerationConfig`
  - VLM 생성 파라미터
- `KananaVLMEngine`
  - processor/model 로딩
  - 이미지 포함 conv 생성
  - assistant 응답 생성
  - 감정 JSON 판정

### 주요 책임

1. `AutoProcessor` / `AutoModelForVision2Seq` 로딩
2. OpenAI-style messages를 Kanana conv 포맷으로 변환
3. 이미지 파일 로드 또는 dummy image 대체
4. `generate_from_messages()`를 통해 이미지 포함 생성
5. 디코드 결과에서 마지막 assistant 구간만 추출
6. 감정 JSON one-hot 추정

### 구현 포인트

- 현재 `build_prompt()`는 제거되었고, `generate_from_messages()`가 `_build_conv()`를 직접 호출한다.
- 이미지가 없을 때도 모델 입력 형식을 맞추기 위해 dummy image를 사용할 수 있다.
- 현재 VLM 경로는 vLLM이 아니라 Hugging Face 경로를 사용한다.

---

## 4.3 `prompt_compiler.py`

### 목적

캐릭터 RP system prompt를 조립하는 모듈이다.

### 핵심 구성

- `CharacterProfile`
  - 캐릭터 기본 정보
- `PromptCompiler`
  - profile 기반 system message 생성

### 현재 사용 방식

- `webapi`
  - `CharacterProfile(name="사야")`
- `webapi_vlm`
  - 동일하게 `CharacterProfile(name="사야")`

즉 현재는 최소 profile만 유지하고, 불필요한 persona/speaking_style 필드는 제거된 상태를 전제로 한다.

---

## 4.4 `rp_parser.py`

### 목적

RP 원문 응답을 구조화 블록으로 분리하는 파서다.

### 핵심 구성

- `RPBlock`
  - `narration`
  - `action`
  - `dialogue_en`

- `RPParser`
  - 원문 텍스트를 `RPBlock`으로 파싱

### 역할

모델이 생성한 응답을:
- 서술
- 행동
- 대사

로 나누고, 이후 번역/TTS에서는 주로 대사만 사용한다.

---

## 4.5 `translator.py`

### 목적

한국어 대사를 일본어로 번역하는 번역기 래퍼다.

### 핵심 구성

- `KoJaTranslator`

### 역할

- 입력: `dialogue_ko`
- 출력: `dialogue_ja`

### 사용 위치

- `webapi`
- `webapi_vlm`

양쪽 모두 동일한 번역기를 사용한다.

---

## 4.6 `tts_worker_client.py`

### 목적

Style-BERT-VITS2 워커를 호출하는 TTS 클라이언트다.

### 핵심 구성

- `SBV2WorkerClient`

### 역할

- 일본어 텍스트를 워커로 보냄
- `style_index`, `style_weight`, `speaker_name`를 함께 넘김
- 결과 WAV 경로를 반환

---

## 4.7 `memory_chain.py`

### 목적

SQLite 기반 장기 기억 관리 모듈이다.

### 핵심 구성

- `_utc_now`
- `_safe_float`
- `_tokenize_ko_en`
- `SummaryMemoryConfig`
- `SummaryMemoryChain`

### 구현 기능

1. 대화 턴 저장
   - `chat_turns`

2. 후보 기억 추출
   - LLM을 사용해 structured candidate 추출
   - 현재는 `llm_engine.generate_from_messages()`가 있으면 그 경로를 우선 사용

3. 후보 점수화
   - confidence
   - future impact
   - emotion intensity
   - recurrence
   - novelty

4. 장기 기억 슬롯 승격
   - `memory_slots`

5. retrieval
   - 가능하면 sqlite-vec 기반 vector search
   - 아니면 lexical fallback

6. system message 생성
   - 장기기억 슬롯 + 최근 대화 스냅샷을 한 system message로 반환

### 현재 중요 포인트

이 모듈은 더 이상 Qwen 전용이 아니다.

현재는:
- `QwenEngine`
- `KananaVLMEngine`

둘 다 사용할 수 있게 일반화되어 있다.

즉 `webapi`와 `webapi_vlm` 모두 같은 메모리 체인을 공유 가능하다.

---

## 4.8 `langgraph_pipeline.py`

### 목적

Qwen 텍스트 경로용 턴 파이프라인을 LangGraph로 구성한다.

### 상태 구조

- `TurnState`
  - 입력 텍스트
  - style/speaker
  - prompt/messages
  - rp_text / narration / dialogue
  - 번역/TTS/감정 결과

### 노드 순서

1. `prepare_prompt`
2. `generate_parse`
3. `translate`
4. `emotion`
5. `tts`
6. `commit`

### commit 역할

- history append
- memory update

즉 Qwen 메인 대화 경로는 LangGraph와 memory chain을 같이 사용한다.

---

## 4.9 `vlm_langgraph_pipeline.py`

### 목적

VLM 전용 턴 파이프라인을 LangGraph로 구성한다.

### 상태 구조

- `VLMTurnState`
  - `text_ko`
  - `image_path`
  - style/speaker
  - messages
  - rp_text / narration / dialogue
  - 번역/TTS/감정 결과

### 핵심 차이

이미지 유무에 따라 생성 노드가 분기된다.

### 노드 흐름

1. `prepare_prompt`
2. `route_has_image`
3. `extract_visual_facts` 또는 `generate_text_only`
4. `generate_with_image` 또는 `translate`
4. `translate`
5. `emotion`
6. `tts`
7. `commit`

### 이미지 guidance 규칙

이미지가 있을 때는 2단 구조로 처리한다.

1. `extract_visual_facts`
- 현재 이미지에서 보이는 사실만 추출
- 읽을 수 있는 글자가 있으면 OCR 시도
- 출력 형식은 RP가 아니라 구조화된 visual facts 텍스트
- 이전 이미지 OCR 결과나 memory를 섞지 않도록 강한 제약을 둔다

2. `generate_with_image`
- 기본 RP system
- memory system message
- history
- current user
- image-aware RP guidance
- 직전에 추출된 visual facts

즉 이미지를 직접 보고 RP를 한 번에 만드는 게 아니라,
먼저 시각 사실을 추출한 뒤 그 결과를 바탕으로 RP를 생성한다.

시각 사실 추출 단계의 핵심 규칙:
- 현재 장면의 1차 근거는 현재 이미지
- 읽을 수 있는 글자는 가능한 한 읽는다
- 확신이 낮으면 추측이라고 명시한다
- 이전 턴 이미지 단서는 자동 재사용하지 않는다

RP 생성 단계의 핵심 규칙:
- visual facts를 현재 장면의 우선 근거로 사용
- memory는 관계/배경/누적 단서 보강용
- 최종 출력은 반드시 서술 1블록 + 대사 1블록

### commit 역할

- history append
- memory update

즉 현재 VLM 경로도 LangGraph + memory chain을 사용한다.

---

## 4.10 `utils/text_split.py`

### 목적

텍스트를 문장 단위로 나누는 유틸리티다.

### 역할

현재 `system` 런타임의 핵심 축은 아니지만, 번역/TTS 또는 후처리 유틸로 재사용 가능한 보조 모듈이다.

---

## 5. `webapi` 폴더 설명

### 구성

- `app.py`
  - FastAPI 진입점
- `schemas.py`
  - Pydantic 요청/응답 스키마
- `services.py`
  - RuntimeServices
- `demo.py`
  - Gradio 시연 UI
- `assets/`
  - 배경/표정/캐릭터 리소스

### 현재 역할

Qwen 텍스트 RP 데모 전용 API/UI다.

### 특징

- LangGraph 사용
- SummaryMemoryChain 사용
- 번역/TTS/감정 판정 통합
- Gradio는 REST 소비자 역할

---

## 6. `webapi_vlm` 폴더 설명

### 구성

- `app.py`
  - VLM 전용 FastAPI 진입점
- `schemas.py`
  - VLM 요청/응답 스키마
- `services.py`
  - VLM RuntimeServices
- `demo.py`
  - VLM 전용 Gradio UI

### 현재 역할

Kanana 기반 VLM RP 데모 전용 API/UI다.

### 특징

- 이미지 업로드 입력 지원
- 이미지 유무에 따라 LangGraph 분기
- SummaryMemoryChain 사용
- 번역/TTS/감정 판정 통합
- 현재 vLLM 미사용

---

## 7. Qwen 경로와 VLM 경로의 차이

## 7.1 공통점

- prompt compiler 사용
- RP parser 사용
- translator 사용
- TTS worker client 사용
- history 사용
- memory chain 사용
- LangGraph 사용
- 감정 one-hot -> UI 표정 매핑 사용

## 7.2 차이점

### Qwen 경로

- 엔진: `QwenEngine`
- 입력: 텍스트만
- prompt 생성: `build_prompt(messages)`
- backend: vLLM 또는 HF 경로 가능

### VLM 경로

- 엔진: `KananaVLMEngine`
- 입력: 텍스트 + 선택적 이미지
- prompt 생성: Kanana conv 포맷
- 이미지 있을 때 system guidance 추가
- backend: 현재 HF `AutoModelForVision2Seq`

---

## 8. 감정 상태 매핑 구조

### 생성 단계

- `llm_engine.infer_emotion_json(...)`
- 실패 시 keyword fallback

### 응답 형식

- `neutral`
- `sad`
- `happy`
- `angry`

각 값은 one-hot `0/1`

### UI 단계

Gradio 데모는 응답의 감정 상태를 받아 캐릭터 이미지 선택에 사용한다.

- angry -> annoyed
- sad -> crying
- happy -> smile
- default -> neutral

즉 "감정 판정"과 "표정 렌더링"은 분리돼 있다.

---

## 9. 상태 저장 구조

### 9.1 short-term history

- `deque(maxlen=6)`
- user/assistant 두 메시지가 1턴
- 최근 3턴 유지

### 9.2 long-term memory

- SQLite DB
- 후보 추출
- slot 승격
- retrieval 후 system message 주입

### 9.3 TTS 결과

- wav 경로만 반환
- 실제 오디오 파일 관리는 TTS 워커 또는 outputs 디렉터리 외부 정책에 따름

---

## 10. 동시성 모델

각 `RuntimeServices`는 대체로 다음 전략을 쓴다.

- `_load_lock`
  - 무거운 모델/번역기/TTS 중복 로딩 방지
- `_turn_lock`
  - history/memory/DB 업데이트 직렬화

즉,
- 읽기보다
- "턴 단위 일관성"

을 우선하는 구조다.

---

## 11. 현재 구현 제약

1. `webapi_vlm`은 현재 vLLM을 사용하지 않는다.
2. `saya_vlm_3b`는 custom architecture이므로 vLLM 즉시 호환 모델이 아니다.
3. VLM 경로의 이미지 이해 품질은 모델 자체와 prompt guidance 품질에 크게 좌우된다.
4. memory chain은 공용화됐지만, 후보 추출 품질은 사용하는 엔진 출력 안정성에 의존한다.

---

## 12. 실제 실행 진입점

### Qwen REST + Demo

```bash
uv run uvicorn system.webapi.app:app --host 0.0.0.0 --port 8000
```

### VLM REST + Demo

```bash
uv run uvicorn system.webapi_vlm.app:app --host 0.0.0.0 --port 8001
```

### VLM memory debug 로그 활성화

```bash
export VLM_MEMORY_DEBUG=1
uv run uvicorn system.webapi_vlm.app:app --host 0.0.0.0 --port 8001
```

---

## 13. 파일별 빠른 인덱스

- 텍스트 LLM 엔진: `system/llm_engine.py`
- VLM 엔진: `system/vlm_engine.py`
- 장기기억: `system/memory_chain.py`
- 텍스트 LangGraph: `system/langgraph_pipeline.py`
- VLM LangGraph: `system/vlm_langgraph_pipeline.py`
- 프롬프트 생성: `system/prompt_compiler.py`
- RP 파싱: `system/rp_parser.py`
- 번역: `system/translator.py`
- TTS: `system/tts_worker_client.py`
- Qwen API: `system/webapi/app.py`
- Qwen 서비스: `system/webapi/services.py`
- Qwen 데모: `system/webapi/demo.py`
- VLM API: `system/webapi_vlm/app.py`
- VLM 서비스: `system/webapi_vlm/services.py`
- VLM 데모: `system/webapi_vlm/demo.py`

---

## 14. 요약

현재 `system` 폴더는

- Qwen 텍스트 RP 런타임
- Kanana VLM RP 런타임
- 공통 번역/TTS/파싱/메모리
- 두 개의 독립 REST/Gradio 진입점

으로 구성된 대화 시스템 레이어다.

핵심 차이는 엔진과 입력 형태이고,
핵심 공통점은 LangGraph + memory chain + RP 후처리 구조다.
