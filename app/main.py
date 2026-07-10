import logging

from fastapi import FastAPI

from app.config import settings
from app.routes.face import router as face_router
from app.translation.cache import TranslationCache
from app.translation.config import settings as translation_settings
from app.translation.routes import router as translation_router
from app.translation.translator import ArgosTranslator, TranslationService

logger = logging.getLogger("app.main")

app = FastAPI(title="Ehkini Face AI Service", version="1.0.0")
app.include_router(face_router)
app.include_router(translation_router)


@app.on_event("startup")
async def _warm_up_translation() -> None:
    translator = ArgosTranslator(translation_settings)
    cache = TranslationCache(max_size=translation_settings.translation_cache_max_size)
    app.state.translation_service = TranslationService(translator, cache)

    logger.info("Warming up Argos Translate language packages...")
    translator.warm_up()
    logger.info(
        "Translation ready. Supported languages: %s",
        sorted(translator.supported_languages()),
    )


@app.get("/health")
def health():
    return {"status": "ok", "threshold": settings.face_similarity_threshold}
