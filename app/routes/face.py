from __future__ import annotations

import random

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from app.config import settings
from app.services.liveness import CHALLENGES
from app.services.recognizer import FaceRecognizer
from app.utils import parse_embedding_json, read_image_bytes

router = APIRouter(prefix="/face", tags=["face"])


def _check_api_key(x_api_key: str | None) -> None:
    expected = settings.face_api_key.strip()
    if expected and (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@router.get("/challenge")
def random_challenge(x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    challenge = random.choice(CHALLENGES)
    return {"success": True, "challenge": challenge}


@router.post("/register")
async def register_face(
    image: UploadFile = File(...),
    user_id: int = Form(...),
    challenge: str | None = Form(default=None),
    prior_image: UploadFile | None = File(default=None),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="No valid face detected")

    try:
        frame = read_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recognizer = FaceRecognizer.get()
    # Enrollment uses quality checks only — liveness is for login verify.
    result = recognizer.analyze(frame, require_liveness=False)

    if not result.embedding:
        if result.liveness and not result.liveness.passed:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "message": result.liveness.message or "Liveness check failed",
                },
            )
        message = result.quality.issues[0] if result.quality.issues else "No valid face detected"
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": message, "issues": result.quality.issues},
        )

    return {
        "success": True,
        "user_id": user_id,
        "embedding": result.embedding,
        "quality_score": result.quality.score,
        "det_score": result.det_score,
    }


@router.post("/extract")
async def extract_face(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="No valid face detected")

    try:
        frame = read_image_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recognizer = FaceRecognizer.get()
    result = recognizer.analyze(frame, require_liveness=False)

    if not result.embedding:
        message = result.quality.issues[0] if result.quality.issues else "No valid face detected"
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": message, "issues": result.quality.issues},
        )

    return {
        "success": True,
        "embedding": result.embedding,
        "quality_score": result.quality.score,
        "det_score": result.det_score,
    }


@router.post("/verify")
async def verify_face(
    image: UploadFile = File(...),
    stored_embedding: str = Form(...),
    challenge: str | None = Form(default=None),
    prior_image: UploadFile | None = File(default=None),
    blink_image: UploadFile | None = File(default=None),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="No valid face detected")

    try:
        frame = read_image_bytes(data)
        reference = parse_embedding_json(stored_embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prior_frame = None
    if prior_image is not None:
        prior_bytes = await prior_image.read()
        if prior_bytes:
            try:
                prior_frame = read_image_bytes(prior_bytes)
            except ValueError:
                prior_frame = None

    blink_frame = None
    if blink_image is not None:
        blink_bytes = await blink_image.read()
        if blink_bytes:
            try:
                blink_frame = read_image_bytes(blink_bytes)
            except ValueError:
                blink_frame = None

    recognizer = FaceRecognizer.get()
    result = recognizer.analyze(
        frame,
        challenge=challenge,
        prior_image=prior_frame,
        blink_image=blink_frame,
        require_liveness=bool(challenge),
    )

    if not result.embedding:
        if result.liveness and not result.liveness.passed:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "message": result.liveness.message or "Liveness check failed",
                },
            )
        message = result.quality.issues[0] if result.quality.issues else "No valid face detected"
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": message, "issues": result.quality.issues},
        )

    matched, score = recognizer.compare(result.embedding, reference)
    return {
        "success": True,
        "matched": matched,
        "score": round(score, 4),
        "confidence": round(score, 4),
        "threshold": settings.face_similarity_threshold,
    }
