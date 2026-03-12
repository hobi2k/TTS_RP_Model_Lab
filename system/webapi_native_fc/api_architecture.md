# Native Function Calling WebAPI Architecture

이 문서는 `saya_rp_4b_v3`의 로컬 chat template가 제공하는 네이티브 function calling 포맷을 직접 사용하는 전용 API 구조를 설명한다.

대상 파일:
- `system/native_fc_engine.py`
- `system/native_fc_pipeline.py`
- `system/webapi_native_fc/app.py`
- `system/webapi_native_fc/schemas.py`
- `system/webapi_native_fc/services.py`

## 1. 목적

기존 planner 흉내 구조가 아니라, `saya_rp_4b_v3`의 `chat_template.jinja`에 이미 들어 있는:

- `tools`
- `<tool_call>...</tool_call>`
- `<tool_response>...</tool_response>`

규약을 그대로 사용해 도구 호출 루프를 실행한다.

## 2. 엔진 구조

`NativeFunctionCallingEngine`는 HF 4bit 경로로 모델을 로드하고, 아래 두 기능을 직접 제공한다.

1. `apply_chat_template(messages, tools=...)`로 도구 목록을 모델 prompt에 주입
2. assistant 출력에서 `<tool_call>{...}</tool_call>` 블록을 파싱

## 3. 도구 루프

`run_tool_loop()`는 아래 순서로 동작한다.

1. 현재 messages + tools로 assistant 생성
2. tool call이 있으면 파싱
3. assistant 메시지를 `tool_calls`와 함께 history에 추가
4. 각 tool 결과를 `role=\"tool\"` 메시지로 추가
5. 다시 assistant 생성
6. tool call이 없어지면 최종 RP 응답으로 종료

## 4. 현재 도구

- `memory_lookup`
- `recent_history_lookup`
- `image_analyze`

`image_analyze`는 `KananaVLMEngine`을 별도 도구로 사용한다.

## 5. LangGraph 구조

그래프 노드:
1. `prepare_messages`
2. `native_fc_generate`
3. `translate`
4. `emotion`
5. `tts`
6. `commit`

`native_fc_generate` 노드 내부에서 네이티브 function calling 루프가 돈다.

## 6. API

- `GET /health`
- `POST /api/turn`
- `POST /api/main-loop`

응답은 기존 turn 응답과 유사하지만 `tool_trace`를 포함한다.
