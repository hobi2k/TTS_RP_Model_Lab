from __future__ import annotations

"""FastAPI 요청/응답 스키마 정의.

각 모델은 `/api/*` 엔드포인트와 1:1 대응되며,
Pydantic validation 규칙을 통해 입력 범위(길이, 값 범위, 패턴)를 강제한다.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """`POST /api/chat` 요청 바디."""

    text: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=200)


class ChatResponse(BaseModel):
    """`POST /api/chat` 응답 바디."""

    response: str


class TranslateRequest(BaseModel):
    """`POST /api/translate` 요청 바디."""

    text_ko: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class TranslateResponse(BaseModel):
    """`POST /api/translate` 응답 바디."""

    text_ja: str


class ParseRequest(BaseModel):
    """`POST /api/parse` 요청 바디."""

    text: str = Field(..., min_length=1)


class ParseResponse(BaseModel):
    """`POST /api/parse` 응답 바디."""

    narration: str
    action: str
    dialogue: str


class TTSRequest(BaseModel):
    """`POST /api/tts` 요청 바디."""

    text_ja: str = Field(..., min_length=1)
    style_index: int = Field(default=0, ge=0)
    style_weight: float = Field(default=1.0, ge=0.0, le=3.0)
    speaker_name: str = Field(default="saya", pattern="^(saya|mai)$")


class TTSResponse(BaseModel):
    """`POST /api/tts` 응답 바디."""

    wav_path: str


class TurnRequest(BaseModel):
    """`POST /api/turn`, `POST /api/main-loop` 요청 바디."""

    text_ko: str = Field(..., min_length=1)
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
    """턴 단위 통합 파이프라인 응답 바디."""

    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None = None
    emotion: EmotionState = Field(default_factory=EmotionState)
