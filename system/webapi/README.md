# system/webapi

REST 우선 구조의 Web API 계층입니다.

- REST API: `FastAPI`
- 데모 UI: `Gradio` (`/demo` 마운트)
- `POST /api/turn`은 `main_loop.py`의 파이프라인 순서를 그대로 따릅니다.
  - 입력 -> PromptCompiler + history + memory
  - LLM 생성 -> RP 파싱 -> 대사 번역 -> TTS -> history/memory 업데이트

## Run

```bash
uv run uvicorn system.webapi.app:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health`
- `POST /api/chat`
- `POST /api/translate`
- `POST /api/parse`
- `POST /api/tts`
- `POST /api/turn`

## Demo

- API docs: `http://127.0.0.1:8000/docs`
- Gradio demo: `http://127.0.0.1:8000/demo`

## Optional Env

- `WEBAPI_ENABLE_GRADIO=0|1` (default `1`)
- `WEBAPI_BASE_URL=http://127.0.0.1:8000`
- `QWEN_MODEL_DIR=...`
- `TRANS_BASE_DIR=...`
- `TRANS_LORA_DIR=...`
