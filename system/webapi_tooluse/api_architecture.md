# Tool-Use WebAPI Architecture

이 문서는 `system/webapi_tooluse`의 실제 코드 기준 아키텍처 문서다.

대상 파일:
- `system/webapi_tooluse/app.py`
- `system/webapi_tooluse/schemas.py`
- `system/webapi_tooluse/services.py`

관련 코어 모듈:
- `system/tool_use_pipeline.py`
- `system/llm_engine.py`
- `system/vlm_engine.py`
- `system/memory_chain.py`
- `system/prompt_compiler.py`
- `system/rp_parser.py`
- `system/translator.py`
- `system/tts_worker_client.py`

## 1. 목적

`system/webapi_tooluse`는 기존 고정 파이프라인과 분리된 tool-use 전용 REST 서버다.

핵심 차이:
- 먼저 planner가 필요한 도구를 선택한다.
- 선택된 도구만 실행한다.
- 도구 결과를 근거로 최종 RP 응답을 다시 생성한다.

## 2. 도구 목록

현재 허용 도구:
- `memory_lookup`
- `recent_history_lookup`
- `image_analyze`

`memory_lookup`는 장기기억과 최근 대화 스냅샷을 조회한다.
`recent_history_lookup`는 직전 user/assistant 문면을 다시 확인한다.
`image_analyze`는 현재 턴 이미지가 있을 때만 VLM으로 시각 정보/OCR을 추출한다.

## 3. 런타임 구조

`RuntimeServices`는 아래 리소스를 lazy-load한다.

- planner/composer용 `QwenEngine`
- 이미지 분석용 `KananaVLMEngine`
- `SummaryMemoryChain`
- `PromptCompiler`
- `RPParser`
- translator / TTS client
- tool-use LangGraph

기본적으로 tool-use 경로는 `LLM_BACKEND=hf`, `LLM_STRICT_VLLM=0` 정책을 사용한다.
현재 프로젝트 환경에서 `vllm`이 빠져 있어도 planner/composer가 바로 죽지 않게 하기 위한 선택이다.

## 4. 그래프 구조

실제 그래프는 `system/tool_use_pipeline.py`의 `build_tool_use_turn_graph()`가 만든다.

노드 순서:
1. `plan_tools`
2. 조건 분기
3. `execute_tools`
4. `compose`
5. `translate`
6. `emotion`
7. `tts`
8. `commit`

### 4.1 `plan_tools`

현재 user 입력, 이미지 존재 여부, history 존재 여부를 보고 planner가 JSON 형태의 tool call 목록을 만든다.

### 4.2 `execute_tools`

planner가 고른 도구만 실행한다.
각 도구 결과는 `tool_trace`에 기록되고, 이후 composer가 읽을 `tool_context` 문자열로 합쳐진다.

### 4.3 `compose`

`PromptCompiler`의 기본 RP system prompt 위에 tool 결과를 system message로 얹고,
history와 current user를 함께 넣어 최종 RP 응답을 생성한다.

### 4.4 `commit`

최종 RP 응답을 history에 저장하고 memory chain을 갱신한다.

## 5. API

- `POST /api/tool-plan`
  - planner가 선택한 도구 호출 목록만 반환
- `POST /api/turn`
  - tool-use 메인 턴 실행
- `POST /api/main-loop`
  - `/api/turn`과 같은 동작을 데모 호환 이름으로 노출

## 6. 응답 특징

통합 응답은 기존 `TurnResponse` 계열 필드에 더해 `tool_trace`를 포함한다.

`tool_trace`는 다음 용도에 쓴다.
- planner가 무엇을 골랐는지 확인
- 어떤 도구 결과가 최종 응답에 영향을 줬는지 추적
- prompt/기억/이미지 분석 오염 디버깅
