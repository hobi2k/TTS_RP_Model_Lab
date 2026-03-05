from __future__ import annotations

"""FastAPI 애플리케이션 진입점.

실행 예:
    uv run uvicorn system.webapi.app:app --host 0.0.0.0 --port 8000

구성:
- REST 엔드포인트(`/api/chat`, `/api/turn` 등)
- 런타임 서비스 컨테이너(`RuntimeServices`) 연결
- 선택적 Gradio 데모(`/demo`) 마운트
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app

from system.webapi.demo import build_demo
from system.webapi.schemas import (
    ChatRequest,
    ChatResponse,
    ParseRequest,
    ParseResponse,
    TranslateRequest,
    TranslateResponse,
    TTSRequest,
    TTSResponse,
    TurnRequest,
    TurnResponse,
)
from system.webapi.services import RuntimeServices

print(f"[WEBAPI] app module loaded: {__file__}")
print(f"[WEBAPI] cwd={os.getcwd()}")

services = RuntimeServices()
app = FastAPI(title="saya_char_qwen2.5 REST API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """헬스체크 엔드포인트."""
    return {"status": "ok"}


@app.get("/")
def root():
    """루트 접속 시 데모 또는 OpenAPI 문서로 리다이렉트한다."""
    if os.getenv("WEBAPI_ENABLE_GRADIO", "1") == "1":
        return RedirectResponse(url="/demo")
    return RedirectResponse(url="/docs")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """LLM 단일 응답을 생성한다.

    이 엔드포인트는 파싱/번역/TTS를 수행하지 않고,
    RP 원문 텍스트만 반환한다.
    """
    try:
        text = services.chat(
            req.text,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        return ChatResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat failed: {e}")


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    """한국어 대사 한 줄을 일본어로 번역한다."""
    try:
        text_ja = services.translate(
            req.text_ko,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return TranslateResponse(text_ja=text_ja)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"translate failed: {e}")


@app.post("/api/parse", response_model=ParseResponse)
def parse(req: ParseRequest):
    """RP 원문을 서술/대사 구조로 분해한다."""
    try:
        block = services.parse(req.text)
        return ParseResponse(
            narration=block.narration,
            action=block.action,
            dialogue=block.dialogue_en,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse failed: {e}")


@app.post("/api/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    """일본어 텍스트를 TTS로 합성하고 WAV 경로를 반환한다."""
    try:
        wav_path = services.tts(
            req.text_ja,
            style_index=req.style_index,
            style_weight=req.style_weight,
            speaker_name=req.speaker_name,
        )
        return TTSResponse(wav_path=wav_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")


@app.post("/api/turn", response_model=TurnResponse)
def turn(req: TurnRequest):
    """메인 턴 파이프라인(LLM -> parse -> translate -> TTS)을 실행한다."""
    try:
        result = services.turn(
            req.text_ko,
            style_index=req.style_index,
            style_weight=req.style_weight,
            speaker_name=req.speaker_name,
        )
        return TurnResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"turn failed: {e}")


@app.post("/api/main-loop", response_model=TurnResponse)
def main_loop(req: TurnRequest):
    """`/api/turn`과 동일한 파이프라인을 데모 호환 이름으로 노출한다."""
    try:
        result = services.turn(
            req.text_ko,
            style_index=req.style_index,
            style_weight=req.style_weight,
            speaker_name=req.speaker_name,
        )
        return TurnResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"main-loop failed: {e}")


if os.getenv("WEBAPI_ENABLE_GRADIO", "1") == "1":
    base_url = os.getenv("WEBAPI_BASE_URL", "http://127.0.0.1:8000")
    print(f"[WEBAPI] building Gradio demo from system.webapi.demo with base_url={base_url}")
    demo = build_demo(base_url)
    app = mount_gradio_app(app, demo, path="/demo")
