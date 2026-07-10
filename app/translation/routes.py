"""HTTP surface for the translation feature."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .config import settings
from .models import ErrorResponse, LanguageInfo, LanguagesResponse, TranslateRequest, TranslateResponse
from .translator import TranslationError, TranslationService

logger = logging.getLogger("app.translation.routes")

router = APIRouter()


def _check_api_key(x_api_key: str | None) -> None:
    expected = settings.translate_api_key.strip()
    if expected and (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_service(request: Request) -> TranslationService:
    return request.app.state.translation_service


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages(request: Request):
    service: TranslationService = _get_service(request)
    languages = [
        LanguageInfo(code=code, name=name)
        for code, name in sorted(service.supported_languages().items())
    ]
    return LanguagesResponse(success=True, languages=languages)


@router.post(
    "/translate",
    responses={200: {"model": TranslateResponse}, 400: {"model": ErrorResponse}},
)
async def translate(
    payload: TranslateRequest,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    service: TranslationService = _get_service(request)

    try:
        result, was_cached = await run_in_threadpool(
            service.translate, payload.text, payload.source, payload.target
        )
    except TranslationError as exc:
        logger.info("Translation rejected: %s", exc)
        return JSONResponse(content=ErrorResponse(success=False, message=str(exc)).model_dump())
    except Exception:
        logger.exception("Unexpected translation failure")
        return JSONResponse(
            content=ErrorResponse(success=False, message="Translation failed").model_dump()
        )

    return TranslateResponse(
        success=True,
        translatedText=result.translated_text,
        detectedSource=result.detected_source,
        cached=was_cached,
    )
