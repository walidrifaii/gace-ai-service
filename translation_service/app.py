"""FastAPI entry point for the free message translation service.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8002

The heavy Argos Translate models are loaded once at startup (see
`warm_up`) and reused for every request — they are never reloaded per
request, which is what keeps short-message translation fast.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cache import TranslationCache
from config import settings
from routes import router
from translator import ArgosTranslator, TranslationService

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("translation_service.app")

app = FastAPI(
    title="Ehkini Translation Service",
    description="Free, offline chat message translation powered by Argos Translate.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup() -> None:
    translator = ArgosTranslator(settings)
    cache = TranslationCache(max_size=settings.translation_cache_max_size)
    app.state.translation_service = TranslationService(translator, cache)

    logger.info("Warming up Argos Translate language packages...")
    translator.warm_up()
    logger.info("Translation service ready. Supported languages: %s", sorted(translator.supported_languages()))


@app.get("/")
async def root():
    return {"service": "ehkini-translation-service", "status": "ok"}
