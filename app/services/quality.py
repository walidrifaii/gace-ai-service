from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityResult:
    passed: bool
    score: float
    issues: list[str]


def _face_brightness(image: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _blur_score(image: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_size_ratio(bbox: np.ndarray, image: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox[:4]
    face_area = max(0.0, (x2 - x1) * (y2 - y1))
    img_area = float(image.shape[0] * image.shape[1])
    return face_area / img_area if img_area > 0 else 0.0


def _rotation_angle(kps: np.ndarray) -> float:
    left_eye, right_eye = kps[0], kps[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    return abs(float(np.degrees(np.arctan2(dy, dx))))


def _eyes_open(kps: np.ndarray, bbox: np.ndarray) -> bool:
    face_h = max(1.0, bbox[3] - bbox[1])
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    nose_to_eyes = float(np.linalg.norm(nose - (left_eye + right_eye) / 2))
    ratio = nose_to_eyes / eye_dist if eye_dist > 0 else 0.0
    return ratio > 0.18 and face_h > 40


def validate_face_quality(
    image: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray,
    face_count: int,
) -> QualityResult:
    issues: list[str] = []
    scores: list[float] = []

    if face_count == 0:
        return QualityResult(False, 0.0, ["No valid face detected"])
    if face_count > 1:
        return QualityResult(False, 0.0, ["More than one face detected"])

    h, w = image.shape[:2]
    if w < 320 or h < 320:
        issues.append("Low resolution")

    brightness = _face_brightness(image, bbox)
    if brightness < 45:
        issues.append("Too dark")
    elif brightness > 220:
        issues.append("Too bright")
    scores.append(min(1.0, brightness / 128.0))

    blur = _blur_score(image, bbox)
    if blur < 80:
        issues.append("Too blurry")
    scores.append(min(1.0, blur / 200.0))

    size_ratio = _face_size_ratio(bbox, image)
    if size_ratio < 0.04:
        issues.append("Face too small")
    scores.append(min(1.0, size_ratio / 0.12))

    angle = _rotation_angle(kps)
    if angle > 20:
        issues.append("Face rotated")
    scores.append(max(0.0, 1.0 - angle / 30.0))

    if not _eyes_open(kps, bbox):
        issues.append("Eyes closed or face covered")

    score = float(np.mean(scores)) if scores else 0.0
    passed = len(issues) == 0
    return QualityResult(passed, score, issues)
