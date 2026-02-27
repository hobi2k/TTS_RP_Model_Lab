from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=200)


class ChatResponse(BaseModel):
    response: str


class TranslateRequest(BaseModel):
    text_ko: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class TranslateResponse(BaseModel):
    text_ja: str


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ParseResponse(BaseModel):
    narration: str
    action: str
    dialogue: str


class TTSRequest(BaseModel):
    text_ja: str = Field(..., min_length=1)
    style_index: int = Field(default=0, ge=0)
    style_weight: float = Field(default=1.0, ge=0.0, le=3.0)


class TTSResponse(BaseModel):
    wav_path: str


class TurnRequest(BaseModel):
    text_ko: str = Field(..., min_length=1)
    style_index: int = Field(default=0, ge=0)
    style_weight: float = Field(default=1.0, ge=0.0, le=3.0)


class EmotionState(BaseModel):
    neutral: int = Field(default=1, ge=0, le=1)
    sad: int = Field(default=0, ge=0, le=1)
    happy: int = Field(default=0, ge=0, le=1)
    angry: int = Field(default=0, ge=0, le=1)


class TurnResponse(BaseModel):
    rp_text: str
    narration: str
    dialogue_ko: str
    dialogue_ja: str
    wav_path: str | None = None
    emotion: EmotionState = Field(default_factory=EmotionState)
