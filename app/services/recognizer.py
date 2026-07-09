from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis

from app.config import settings
from app.services.liveness import LivenessResult, validate_challenge
from app.services.quality import QualityResult, validate_face_quality
from app.utils import cosine_similarity, normalize_embedding


@dataclass
class FaceAnalysisResult:
    embedding: list[float]
    quality: QualityResult
    liveness: LivenessResult | None
    det_score: float


class FaceRecognizer:
    _instance: "FaceRecognizer | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._app = FaceAnalysis(name=settings.face_model_name)
        self._app.prepare(ctx_id=-1, det_size=(settings.face_det_size, settings.face_det_size))

    @classmethod
    def get(cls) -> "FaceRecognizer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def analyze(
        self,
        image: np.ndarray,
        *,
        challenge: str | None = None,
        prior_image: np.ndarray | None = None,
        blink_image: np.ndarray | None = None,
        require_liveness: bool = False,
    ) -> FaceAnalysisResult:
        faces = self._app.get(image)
        quality = validate_face_quality(image, np.zeros(4), np.zeros((5, 2)), len(faces))
        if not faces:
            return FaceAnalysisResult([], quality, None, 0.0)

        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        quality = validate_face_quality(image, face.bbox, face.kps, 1)
        if not quality.passed:
            return FaceAnalysisResult([], quality, None, float(face.det_score))

        prior_kps = None
        prior_bbox = None
        if prior_image is not None and challenge:
            prior_faces = self._app.get(prior_image)
            if prior_faces:
                prior_face = max(
                    prior_faces,
                    key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                )
                prior_kps = prior_face.kps
                prior_bbox = prior_face.bbox

        blink_kps = None
        blink_bbox = None
        if blink_image is not None and challenge:
            blink_faces = self._app.get(blink_image)
            if blink_faces:
                blink_face = max(
                    blink_faces,
                    key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                )
                blink_kps = blink_face.kps
                blink_bbox = blink_face.bbox

        liveness: LivenessResult | None = None
        if require_liveness and challenge:
            liveness = validate_challenge(
                challenge,
                face.kps,
                face.bbox,
                image,
                prior_kps,
                prior_bbox=prior_bbox,
                prior_image=prior_image,
                blink_image=blink_image,
                blink_kps=blink_kps,
                blink_bbox=blink_bbox,
            )
            if not liveness.passed:
                return FaceAnalysisResult([], quality, liveness, float(face.det_score))

        embedding = normalize_embedding(face.embedding.astype(np.float32))
        return FaceAnalysisResult(
            embedding.tolist(),
            quality,
            liveness,
            float(face.det_score),
        )

    def compare(self, probe: list[float], reference: list[float]) -> tuple[bool, float]:
        score = cosine_similarity(np.asarray(probe), np.asarray(reference))
        matched = score >= settings.face_similarity_threshold
        return matched, score
