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


def _eye_aspect_ratio(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_eye, right_eye = kps[0], kps[1]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    face_h = max(1.0, bbox[3] - bbox[1])
    return eye_dist / face_h


def _mouth_smile_ratio(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_mouth, right_mouth = kps[3], kps[4]
    mouth_w = float(np.linalg.norm(right_mouth - left_mouth))
    face_w = max(1.0, bbox[2] - bbox[0])
    return mouth_w / face_w


def _head_yaw(kps: np.ndarray, bbox: np.ndarray) -> float:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    face_center_x = (bbox[0] + bbox[2]) / 2.0
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
    # Printed photos / screens often have lower texture variance.
    texture = min(1.0, high_freq / 120.0)
    color = min(1.0, color_std / 35.0)
    return (texture + color) / 2.0


def validate_challenge(
    challenge: str,
    kps: np.ndarray,
    bbox: np.ndarray,
    image: np.ndarray,
    prior_kps: np.ndarray | None = None,
) -> LivenessResult:
    challenge = (challenge or "").strip().lower()
    if challenge not in CHALLENGES:
        return LivenessResult(False, "Unknown liveness challenge", challenge)

    anti_spoof = _print_attack_score(image, bbox)
    if anti_spoof < 0.35:
        return LivenessResult(False, "Liveness check failed", challenge)

    if challenge == "blink":
        if prior_kps is None:
            return LivenessResult(False, "Blink requires two frames", challenge)
        ear_before = _eye_aspect_ratio(prior_kps, bbox)
        ear_after = _eye_aspect_ratio(kps, bbox)
        passed = ear_after < ear_before * 0.82
        return LivenessResult(
            passed,
            "Blink detected" if passed else "Blink not detected",
            challenge,
        )

    if challenge == "turn_left":
        passed = _head_yaw(kps, bbox) < -0.06
        return LivenessResult(
            passed,
            "Head turn left confirmed" if passed else "Please turn your head left",
            challenge,
        )

    if challenge == "turn_right":
        passed = _head_yaw(kps, bbox) > 0.06
        return LivenessResult(
            passed,
            "Head turn right confirmed" if passed else "Please turn your head right",
            challenge,
        )

    if challenge == "look_up":
        passed = _head_pitch(kps, bbox) < -0.08
        return LivenessResult(
            passed,
            "Look up confirmed" if passed else "Please look up",
            challenge,
        )

    if challenge == "smile":
        passed = _mouth_smile_ratio(kps, bbox) > 0.42
        return LivenessResult(
            passed,
            "Smile detected" if passed else "Please smile",
            challenge,
        )

    return LivenessResult(False, "Liveness check failed", challenge)
