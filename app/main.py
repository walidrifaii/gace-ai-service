from fastapi import FastAPI

from app.config import settings
from app.routes.face import router as face_router

app = FastAPI(title="Ehkini Face AI Service", version="1.0.0")
app.include_router(face_router)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": settings.face_similarity_threshold}
