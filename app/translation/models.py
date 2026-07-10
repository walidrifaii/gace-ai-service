"""Pydantic request/response contracts for the translation API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source: str = Field(default="auto", description="Source language code, or 'auto' to detect")
    target: str = Field(..., description="Target language code, e.g. 'ar'")

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value

    @field_validator("source", "target")
    @classmethod
    def _lower_code(cls, value: str) -> str:
        return value.strip().lower()


class TranslateResponse(BaseModel):
    success: bool = True
    translatedText: str
    detectedSource: str | None = None
    cached: bool = False


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    success: bool = True
    languages: list[LanguageInfo]
