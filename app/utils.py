import json
from typing import Any

import cv2
import numpy as np


def read_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = normalize_embedding(np.asarray(a, dtype=np.float32))
    b_norm = normalize_embedding(np.asarray(b, dtype=np.float32))
    return float(np.dot(a_norm, b_norm))


def parse_embedding_json(raw: str) -> list[float]:
    data = json.loads(raw)
    if isinstance(data, dict) and "embedding" in data:
        data = data["embedding"]
    if not isinstance(data, list):
        raise ValueError("stored_embedding must be a JSON array of floats")
    return [float(v) for v in data]
