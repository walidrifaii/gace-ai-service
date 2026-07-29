from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.utils import normalize_embedding

logger = logging.getLogger("app.gallery")


@dataclass
class GalleryMatch:
    user_id: int
    score: float


class FaceGallery:
    """
    In-memory vector gallery for fast 1:N face login.
    Uses FAISS IndexFlatIP when available (cosine via L2-normalized vectors),
    otherwise a NumPy matrix multiply fallback.
    """

    _instance: "FaceGallery | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._user_ids: list[int] = []
        self._matrix: np.ndarray | None = None  # shape (n, dim), L2-normalized
        self._faiss_index = None
        self._dim: int | None = None
        self._faiss = None
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            logger.info("FAISS available — using IndexFlatIP for face gallery")
        except Exception:
            logger.info("FAISS not installed — using NumPy gallery fallback")

    @classmethod
    def get(cls) -> "FaceGallery":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def size(self) -> int:
        return len(self._user_ids)

    def clear(self) -> None:
        with self._lock:
            self._user_ids = []
            self._matrix = None
            self._faiss_index = None
            self._dim = None

    def upsert(self, user_id: int, embedding: list[float]) -> None:
        vector = normalize_embedding(np.asarray(embedding, dtype=np.float32))
        if vector.size == 0:
            return
        with self._lock:
            if self._dim is None:
                self._dim = int(vector.size)
            if int(vector.size) != self._dim:
                logger.warning(
                    "gallery.upsert dim mismatch user_id=%s got=%s expected=%s",
                    user_id,
                    vector.size,
                    self._dim,
                )
                return

            if user_id in self._user_ids:
                idx = self._user_ids.index(user_id)
                assert self._matrix is not None
                self._matrix[idx] = vector
            else:
                self._user_ids.append(user_id)
                row = vector.reshape(1, -1)
                if self._matrix is None:
                    self._matrix = row.copy()
                else:
                    self._matrix = np.vstack([self._matrix, row])

            self._rebuild_faiss_locked()

    def remove(self, user_id: int) -> None:
        with self._lock:
            if user_id not in self._user_ids:
                return
            idx = self._user_ids.index(user_id)
            self._user_ids.pop(idx)
            if self._matrix is None:
                return
            self._matrix = np.delete(self._matrix, idx, axis=0)
            if self._matrix.size == 0:
                self._matrix = None
                self._faiss_index = None
                self._dim = None
            else:
                self._rebuild_faiss_locked()

    def rebuild(self, items: list[tuple[int, list[float]]]) -> int:
        with self._lock:
            self._user_ids = []
            rows: list[np.ndarray] = []
            dim: int | None = None
            for user_id, embedding in items:
                vector = normalize_embedding(np.asarray(embedding, dtype=np.float32))
                if vector.size == 0:
                    continue
                if dim is None:
                    dim = int(vector.size)
                if int(vector.size) != dim:
                    continue
                self._user_ids.append(int(user_id))
                rows.append(vector)
            self._dim = dim
            self._matrix = np.vstack(rows) if rows else None
            self._rebuild_faiss_locked()
            return len(self._user_ids)

    def search(self, probe: list[float], *, top_k: int = 1) -> GalleryMatch | None:
        vector = normalize_embedding(np.asarray(probe, dtype=np.float32))
        if vector.size == 0:
            return None

        with self._lock:
            if not self._user_ids or self._matrix is None:
                return None
            if self._dim is not None and int(vector.size) != self._dim:
                return None

            k = max(1, min(top_k, len(self._user_ids)))
            if self._faiss_index is not None and self._faiss is not None:
                scores, indices = self._faiss_index.search(vector.reshape(1, -1), k)
                idx = int(indices[0][0])
                score = float(scores[0][0])
                if idx < 0 or idx >= len(self._user_ids):
                    return None
                return GalleryMatch(self._user_ids[idx], score)

            # NumPy: cosine = dot for L2-normalized rows
            scores = self._matrix @ vector
            idx = int(np.argmax(scores))
            return GalleryMatch(self._user_ids[idx], float(scores[idx]))

    def _rebuild_faiss_locked(self) -> None:
        self._faiss_index = None
        if self._faiss is None or self._matrix is None or self._dim is None:
            return
        try:
            index = self._faiss.IndexFlatIP(self._dim)
            index.add(np.ascontiguousarray(self._matrix, dtype=np.float32))
            self._faiss_index = index
        except Exception as exc:
            logger.warning("FAISS rebuild failed: %s", exc)
            self._faiss_index = None


def gallery_threshold() -> float:
    return float(settings.face_similarity_threshold)
