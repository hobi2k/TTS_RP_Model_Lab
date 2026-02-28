from __future__ import annotations
"""
uv run uvicorn system.webapi.app:app --host 0.0.0.0 --port 8000
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
    return {"status": "ok"}


@app.get("/")
def root():
    if os.getenv("WEBAPI_ENABLE_GRADIO", "1") == "1":
        return RedirectResponse(url="/demo")
    return RedirectResponse(url="/docs")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
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
