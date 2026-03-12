from __future__ import annotations

"""Tool-use 전용 FastAPI 애플리케이션."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from system.webapi_tooluse.schemas import (
    ToolPlanRequest,
    ToolPlanResponse,
    TurnRequest,
    TurnResponse,
)
from system.webapi_tooluse.services import RuntimeServices

services = RuntimeServices()
app = FastAPI(title="saya_tool_use REST API", version="0.1.0")

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
    """루트 접속 시 OpenAPI 문서로 리다이렉트한다."""
    return RedirectResponse(url="/docs")


@app.post("/api/tool-plan", response_model=ToolPlanResponse)
def tool_plan(req: ToolPlanRequest):
    """planner가 선택한 도구 호출 목록만 반환한다."""
    try:
        return ToolPlanResponse(tool_calls=services.tool_plan(req.text_ko, req.image_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"tool-plan failed: {e}")


@app.post("/api/turn", response_model=TurnResponse)
def turn(req: TurnRequest):
    """tool-use 메인 턴 파이프라인을 실행한다."""
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
    """데모 호환 이름의 tool-use 메인 턴 엔드포인트."""
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
