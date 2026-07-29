from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import settings
from app.services.liveness import LivenessResult, validate_challenge
from app.services.quality import QualityResult, validate_face_quality
from app.utils import cosine_similarity, crop_around_bbox, normalize_embedding

logger = logging.getLogger("app.recognizer")


@dataclass
class FaceAnalysisResult:
    embedding: list[float]
    quality: QualityResult
    liveness: LivenessResult | None
    det_score: float


def _available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def _resolve_ctx_and_providers() -> tuple[int, list[str]]:
    """
    Prefer CUDA GPU when available (or FACE_CTX_ID >= 0), else CPU.
    InsightFace: ctx_id >= 0 → GPU, ctx_id == -1 → CPU.
    """
    available = _available_onnx_providers()
    wants_gpu = settings.face_ctx_id >= 0 or settings.face_force_gpu
    cuda_ok = "CUDAExecutionProvider" in available

    if wants_gpu and cuda_ok:
        ctx_id = settings.face_ctx_id if settings.face_ctx_id >= 0 else 0
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("InsightFace using GPU ctx_id=%s providers=%s", ctx_id, providers)
        return ctx_id, providers

    if wants_gpu and not cuda_ok:
        logger.warning(
            "GPU requested but CUDAExecutionProvider missing (available=%s). "
            "Install onnxruntime-gpu. Falling back to CPU.",
            available,
        )

    logger.info("InsightFace using CPU providers=%s", ["CPUExecutionProvider"])
    return -1, ["CPUExecutionProvider"]


class FaceRecognizer:
    _instance: "FaceRecognizer | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        ctx_id, providers = _resolve_ctx_and_providers()
        self.ctx_id = ctx_id
        self.providers = providers
        self._app = FaceAnalysis(name=settings.face_model_name, providers=providers)
        self._app.prepare(
            ctx_id=ctx_id,
            det_size=(settings.face_det_size, settings.face_det_size),
        )
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
        """Only used after a full-frame miss."""
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

    def _resolve_face(self, image: np.ndarray) -> tuple[object | None, np.ndarray]:
        """
        Fast path: detect once on the original frame.
        Only rotate / search regions if that first detection fails.
        """
        faces = self._detect(image)
        work = image

        if not faces:
            # Lazy orientation search — only after first miss.
            for code in (
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ):
                candidate = cv2.rotate(image, code)
                found = self._detect(candidate)
                if found:
                    faces = found
                    work = candidate
                    break

        if not faces:
            faces, work = self._detect_in_regions(work if work is not image else image)

        if not faces:
            return None, image

        face = self._largest_face(faces)
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
