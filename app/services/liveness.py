from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CHALLENGES = ("blink", "turn_left", "turn_right", "look_up", "smile")


@dataclass
class LivenessResult:
    passed: bool
    message: str
    challenge: str | None = None


def _eye_region_openness(image: np.ndarray, kps: np.ndarray, bbox: np.ndarray) -> float:
    """Estimate how open the eyes are from local image texture around eye landmarks."""
    face_w = max(1.0, float(bbox[2] - bbox[0]))
    radius = max(5, int(face_w * 0.075))
    h, w = image.shape[:2]
    scores: list[float] = []

    for eye in (kps[0], kps[1]):
        cx, cy = int(eye[0]), int(eye[1])
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w, cx + radius), min(h, cy + radius)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(np.std(gray))
        edges = float(cv2.Canny(gray, 40, 120).mean())
        scores.append(lap * 0.45 + contrast * 2.5 + edges * 3.0)

    return float(np.mean(scores)) if scores else 0.0


def _mouth_smile_ratio(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_mouth, right_mouth = kps[3], kps[4]
    mouth_w = float(np.linalg.norm(right_mouth - left_mouth))
    face_w = max(1.0, bbox[2] - bbox[0])
    return mouth_w / face_w


def _head_yaw(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    face_w = max(1.0, bbox[2] - bbox[0])
    return float((nose[0] - eye_center_x) / face_w)


def _head_pitch(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    face_h = max(1.0, bbox[3] - bbox[1])
    return float((nose[1] - eye_center_y) / face_h)


def _print_attack_score(image: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 1.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    high_freq = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    color_std = float(np.std(roi))
    texture = min(1.0, high_freq / 120.0)
    color = min(1.0, color_std / 35.0)
    return (texture + color) / 2.0


def _validate_blink(
    prior_image: np.ndarray,
    prior_kps: np.ndarray,
    prior_bbox: np.ndarray,
    blink_image: np.ndarray | None,
    blink_kps: np.ndarray | None,
    blink_bbox: np.ndarray | None,
    fallback_image: np.ndarray,
    fallback_kps: np.ndarray,
    fallback_bbox: np.ndarray,
) -> LivenessResult:
    open_score = _eye_region_openness(prior_image, prior_kps, prior_bbox)
    if open_score <= 0.0:
        return LivenessResult(False, "Blink not detected — keep eyes open first", "blink")

    probe_image = blink_image if blink_image is not None else fallback_image
    probe_kps = blink_kps if blink_kps is not None else fallback_kps
    probe_bbox = blink_bbox if blink_bbox is not None else fallback_bbox

    closed_score = _eye_region_openness(probe_image, probe_kps, probe_bbox)
    ratio = closed_score / open_score if open_score > 0 else 1.0
    drop = open_score - closed_score

    passed = ratio < 0.78 or drop > max(3.0, open_score * 0.18)
    return LivenessResult(
        passed,
        "Blink detected" if passed else "Blink not detected — close your eyes when prompted",
        "blink",
    )


def validate_challenge(
    challenge: str,
    kps: np.ndarray,
    bbox: np.ndarray,
    image: np.ndarray,
    prior_kps: np.ndarray | None = None,
    prior_bbox: np.ndarray | None = None,
    prior_image: np.ndarray | None = None,
    blink_image: np.ndarray | None = None,
    blink_kps: np.ndarray | None = None,
    blink_bbox: np.ndarray | None = None,
) -> LivenessResult:
    challenge = (challenge or "").strip().lower()
    if challenge not in CHALLENGES:
        return LivenessResult(False, "Unknown liveness challenge", challenge)

    anti_spoof = _print_attack_score(image, bbox)
    if anti_spoof < 0.08:
        return LivenessResult(False, "Liveness check failed", challenge)

    if challenge == "blink":
        if prior_kps is None or prior_image is None:
            return LivenessResult(False, "Blink not detected — please try again", challenge)
        blink_bbox_ref = prior_bbox if prior_bbox is not None else bbox
        return _validate_blink(
            prior_image,
            prior_kps,
            blink_bbox_ref,
            blink_image,
            blink_kps,
            blink_bbox,
            image,
            kps,
            bbox,
        )

    yaw = _head_yaw(kps, bbox)

    if challenge == "turn_left":
        passed = yaw < -0.035 or yaw > 0.035
        return LivenessResult(
            passed,
            "Head turn confirmed" if passed else "Please turn your head to the side",
            challenge,
        )

    if challenge == "turn_right":
        passed = yaw < -0.035 or yaw > 0.035
        return LivenessResult(
            passed,
            "Head turn confirmed" if passed else "Please turn your head to the side",
            challenge,
        )

    if challenge == "look_up":
        passed = _head_pitch(kps, bbox) < -0.05
        return LivenessResult(
            passed,
            "Look up confirmed" if passed else "Please look up",
            challenge,
        )

    if challenge == "smile":
        passed = _mouth_smile_ratio(kps, bbox) > 0.34
        return LivenessResult(
            passed,
            "Smile detected" if passed else "Please smile",
            challenge,
        )

    return LivenessResult(False, "Liveness check failed", challenge)
