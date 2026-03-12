from __future__ import annotations

"""saya_rp_4b_v3 네이티브 function calling 전용 FastAPI 앱."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from system.webapi_native_fc.schemas import TurnRequest, TurnResponse
from system.webapi_native_fc.services import RuntimeServices

services = RuntimeServices()
app = FastAPI(title="saya_native_fc REST API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """헬스체크."""
    return {"status": "ok"}


@app.get("/")
def root():
    """루트 접속 시 OpenAPI 문서로 보낸다."""
    return RedirectResponse(url="/docs")


@app.post("/api/turn", response_model=TurnResponse)
def turn(req: TurnRequest):
    """네이티브 function calling 메인 턴."""
    try:
        return TurnResponse(
            **services.turn(
                req.text_ko,
                image_path=req.image_path,
                style_index=req.style_index,
                style_weight=req.style_weight,
                speaker_name=req.speaker_name,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"turn failed: {e}")


@app.post("/api/main-loop", response_model=TurnResponse)
def main_loop(req: TurnRequest):
    """데모 호환 이름의 네이티브 function calling 메인 턴."""
    try:
        return TurnResponse(
            **services.turn(
                req.text_ko,
                image_path=req.image_path,
                style_index=req.style_index,
                style_weight=req.style_weight,
                speaker_name=req.speaker_name,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"main-loop failed: {e}")
