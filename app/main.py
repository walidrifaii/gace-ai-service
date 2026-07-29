import logging

from fastapi import FastAPI

from app.config import settings
from app.routes.face import router as face_router
from app.services.recognizer import FaceRecognizer, _available_onnx_providers
from app.translation.cache import TranslationCache
from app.translation.config import settings as translation_settings
from app.translation.routes import router as translation_router
from app.translation.translator import ArgosTranslator, TranslationService

logger = logging.getLogger("app.main")

app = FastAPI(title="Ehkini Face AI Service", version="1.1.0")
app.include_router(face_router)
app.include_router(translation_router)


@app.on_event("startup")
async def _warm_up() -> None:
    translator = ArgosTranslator(translation_settings)
    cache = TranslationCache(max_size=translation_settings.translation_cache_max_size)
    app.state.translation_service = TranslationService(translator, cache)

    logger.info("Warming up Argos Translate language packages...")
    translator.warm_up()
    logger.info(
        "Translation ready. Supported languages: %s",
        sorted(translator.supported_languages()),
    )

    logger.info("Warming up InsightFace model...")
    recognizer = FaceRecognizer.get()
    logger.info(
        "InsightFace ready. ctx_id=%s providers=%s onnx=%s",
        recognizer.ctx_id,
        recognizer.providers,
        _available_onnx_providers(),
    )


@app.get("/health")
def health():
    try:
        recognizer = FaceRecognizer.get()
        ctx_id = recognizer.ctx_id
        providers = recognizer.providers
    except Exception:
        ctx_id = None
        providers = []
    return {
        "status": "ok",
        "threshold": settings.face_similarity_threshold,
        "ctx_id": ctx_id,
        "providers": providers,
        "gpu": bool(ctx_id is not None and ctx_id >= 0),
        "onnx_providers": _available_onnx_providers(),
    }
