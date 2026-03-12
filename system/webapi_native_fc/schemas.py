from __future__ import annotations

"""네이티브 function calling WebAPI 스키마."""

from pydantic import BaseModel, Field


class ToolTraceModel(BaseModel):
    """실행된 도구 로그."""

    name: str
    args: dict = Field(default_factory=dict)
    result_preview: str = ""


class TurnRequest(BaseModel):
    """`POST /api/turn`, `POST /api/main-loop` 요청 바디."""

    text_ko: str = Field(..., min_length=1)
    image_path: str | None = None
    style_index: int = Field(default=0, ge=0)
    style_weight: float = Field(default=1.0, ge=0.0, le=3.0)
    speaker_name: str = Field(default="saya", pattern="^(saya|mai)$")


class EmotionState(BaseModel):
    """대사 결과의 one-hot 감정 상태."""

    neutral: int = Field(default=1, ge=0, le=1)
    sad: int = Field(default=0, ge=0, le=1)
    happy: int = Field(default=0, ge=0, le=1)
    angry: int = Field(default=0, ge=0, le=1)


class TurnResponse(BaseModel):
    """네이티브 function calling 메인 턴 응답."""

    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None = None
    emotion: EmotionState = Field(default_factory=EmotionState)
    tool_trace: list[ToolTraceModel] = Field(default_factory=list)
