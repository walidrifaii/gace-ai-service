from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
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

    @staticmethod
    def _eye_level_score(face) -> float:
        """Higher when eyes are roughly horizontal (upright face)."""
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 2:
            return 0.0
        left_eye, right_eye = kps[0], kps[1]
        dy = float(right_eye[1] - left_eye[1])
        dx = float(right_eye[0] - left_eye[0])
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        return max(0.0, 1.0 - angle / 90.0)

    @staticmethod
    def _signed_eye_angle(face) -> float:
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 2:
            return 0.0
        left_eye, right_eye = kps[0], kps[1]
        dy = float(right_eye[1] - left_eye[1])
        dx = float(right_eye[0] - left_eye[0])
        return float(np.degrees(np.arctan2(dy, dx)))

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

        regions = [
            (0.0, 0.0, 0.70, 0.70),
            (0.30, 0.0, 1.0, 0.70),
            (0.0, 0.30, 0.70, 1.0),
            (0.30, 0.30, 1.0, 1.0),
            (0.15, 0.0, 0.85, 0.70),
            (0.15, 0.30, 0.85, 1.0),
            (0.0, 0.15, 0.70, 0.85),
            (0.30, 0.15, 1.0, 0.85),
            (0.10, 0.10, 0.90, 0.90),
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

    def _deskew(self, image: np.ndarray, face) -> tuple[object | None, np.ndarray]:
        """Rotate so eyes are level — fixes mild phone tilt like Face ID."""
        angle = self._signed_eye_angle(face)
        if abs(angle) < 6.0 or abs(angle) > 40.0:
            return face, image

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        faces = self._detect(rotated)
        if not faces:
            return face, image
        return self._largest_face(faces), rotated

    def _best_orientation(self, image: np.ndarray) -> tuple[object | None, np.ndarray]:
        """
        Try 0/90/180/270 so login/register works even if the phone photo
        is sideways (common cause of false 'Face rotated').
        """
        candidates: list[np.ndarray] = [image]
        for code in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            candidates.append(cv2.rotate(image, code))

        best_face = None
        best_image = image
        best_score = -1.0

        for candidate in candidates:
            faces = self._detect(candidate)
            work = candidate
            if not faces:
                faces, work = self._detect_in_regions(candidate)
            if not faces:
                continue
            face = self._largest_face(faces)
            area = float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
            score = area * (0.35 + 0.65 * self._eye_level_score(face))
            if score > best_score:
                best_score = score
                best_face = face
                best_image = work

        return best_face, best_image

    def _resolve_face(self, image: np.ndarray) -> tuple[object | None, np.ndarray]:
        """
        Detect a face anywhere in the frame (any orientation / position),
        upright it, then crop around it for stable embeddings.
        """
        face, work = self._best_orientation(image)
        if face is None:
            return None, image

        face, work = self._deskew(work, face)
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
        strict_quality: bool = True,
    ) -> FaceAnalysisResult:
        # Quick multi-face check on original (and upright variants if needed).
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
        # Enrollment can keep going with a soft quality miss (still produce embedding).
        if not quality.passed and strict_quality:
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
