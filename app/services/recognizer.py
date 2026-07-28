from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis

from app.config import settings
from app.services.liveness import LivenessResult, validate_challenge
from app.services.quality import QualityResult, validate_face_quality
from app.utils import cosine_similarity, crop_around_bbox, normalize_embedding


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
        # Slightly lower threshold so faces at frame edges / corners still detect.
        detection = getattr(self._app, "models", {}).get("detection")
        if detection is not None and hasattr(detection, "det_thresh"):
            detection.det_thresh = min(float(detection.det_thresh), 0.40)

    @classmethod
    def get(cls) -> "FaceRecognizer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _largest_face(faces: list):
        return max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )

    def _detect(self, image: np.ndarray) -> list:
        try:
            return list(self._app.get(image) or [])
        except Exception:
            return []

    def _detect_in_regions(self, image: np.ndarray) -> tuple[list, np.ndarray]:
        """Search overlapping regions so off-center faces (corners/edges) are found."""
        h, w = image.shape[:2]
        if h < 80 or w < 80:
            return [], image

        # Fractions: left/right/top/bottom thirds + mid bands.
        regions = [
            (0.0, 0.0, 0.70, 0.70),  # top-left
            (0.30, 0.0, 1.0, 0.70),  # top-right
            (0.0, 0.30, 0.70, 1.0),  # bottom-left
            (0.30, 0.30, 1.0, 1.0),  # bottom-right
            (0.15, 0.0, 0.85, 0.70),  # top-center
            (0.15, 0.30, 0.85, 1.0),  # bottom-center
            (0.0, 0.15, 0.70, 0.85),  # left-center
            (0.30, 0.15, 1.0, 0.85),  # right-center
            (0.10, 0.10, 0.90, 0.90),  # inner frame
        ]

        best_faces: list = []
        best_crop = image
        best_area = 0.0

        for fx1, fy1, fx2, fy2 in regions:
            x1, y1 = int(w * fx1), int(h * fy1)
            x2, y2 = int(w * fx2), int(h * fy2)
            if x2 - x1 < 80 or y2 - y1 < 80:
                continue
            crop = image[y1:y2, x1:x2]
            faces = self._detect(crop)
            if not faces:
                continue
            face = self._largest_face(faces)
            area = float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            if area > best_area:
                best_area = area
                best_faces = faces
                best_crop = crop

        return best_faces, best_crop

    def _resolve_face(self, image: np.ndarray) -> tuple[object | None, np.ndarray]:
        """
        Detect a face anywhere in the frame, then re-crop around it so quality
        and embedding work the same for center / left / right / top / bottom.
        """
        faces = self._detect(image)
        work = image
        if not faces:
            faces, work = self._detect_in_regions(image)
        if not faces:
            return None, image

        face = self._largest_face(faces)
        centered = crop_around_bbox(work, face.bbox)
        refined = self._detect(centered)
        if refined:
            return self._largest_face(refined), centered
        return face, work

    def analyze(
        self,
        image: np.ndarray,
        *,
        challenge: str | None = None,
        prior_image: np.ndarray | None = None,
        blink_image: np.ndarray | None = None,
        require_liveness: bool = False,
    ) -> FaceAnalysisResult:
        # Reject multi-person frames on the full image before region search/crop.
        full_faces = self._detect(image)
        if len(full_faces) > 1:
            face = self._largest_face(full_faces)
            quality = validate_face_quality(image, face.bbox, face.kps, len(full_faces))
            return FaceAnalysisResult([], quality, None, float(face.det_score))

        face, work_image = self._resolve_face(image)
        if face is None:
            quality = validate_face_quality(image, np.zeros(4), np.zeros((5, 2)), 0)
            return FaceAnalysisResult([], quality, None, 0.0)

        quality = validate_face_quality(work_image, face.bbox, face.kps, 1)
        if not quality.passed:
            return FaceAnalysisResult([], quality, None, float(face.det_score))

        prior_kps = None
        prior_bbox = None
        prior_work = prior_image
        if prior_image is not None and challenge:
            prior_face, prior_work = self._resolve_face(prior_image)
            if prior_face is not None:
                prior_kps = prior_face.kps
                prior_bbox = prior_face.bbox

        blink_kps = None
        blink_bbox = None
        blink_work = blink_image
        if blink_image is not None and challenge:
            blink_face, blink_work = self._resolve_face(blink_image)
            if blink_face is not None:
                blink_kps = blink_face.kps
                blink_bbox = blink_face.bbox

        liveness: LivenessResult | None = None
        if require_liveness and challenge:
            liveness = validate_challenge(
                challenge,
                face.kps,
                face.bbox,
                work_image,
                prior_kps,
                prior_bbox=prior_bbox,
                prior_image=prior_work,
                blink_image=blink_work,
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
