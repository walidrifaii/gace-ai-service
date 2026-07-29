from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

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
    FAISS IndexFlatIP gallery for fast 1:N face login.
    Vectors are L2-normalized so inner-product == cosine similarity.
    Index is persisted under FACE_GALLERY_PATH so restarts keep the gallery.
    """

    _instance: "FaceGallery | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._user_ids: list[int] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._dim: int | None = None
        self._faiss = None
        self._backend = "numpy"
        self._persist_dir = Path(settings.face_gallery_path)

        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._backend = "faiss"
            logger.info("FAISS loaded — IndexFlatIP enabled for face gallery")
        except Exception as exc:
            if settings.face_require_faiss:
                raise RuntimeError(
                    "FAISS is required but not installed. "
                    "Add faiss-cpu to requirements and redeploy."
                ) from exc
            logger.warning("FAISS not installed — using NumPy fallback: %s", exc)

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted()

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

    @property
    def backend(self) -> str:
        return self._backend if self._faiss_index is not None or self._faiss is not None else "numpy"

    @property
    def using_faiss(self) -> bool:
        return self._faiss is not None and self._faiss_index is not None

    def status(self) -> dict:
        return {
            "size": self.size,
            "dim": self._dim,
            "backend": "faiss" if self.using_faiss else ("faiss_ready" if self._faiss else "numpy"),
            "faiss": self._faiss is not None,
            "faiss_index_ready": self.using_faiss,
            "persist_path": str(self._persist_dir),
        }

    def clear(self) -> None:
        with self._lock:
            self._user_ids = []
            self._matrix = None
            self._faiss_index = None
            self._dim = None
            self._persist_locked()

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
                self._user_ids.append(int(user_id))
                row = vector.reshape(1, -1)
                if self._matrix is None:
                    self._matrix = row.copy()
                else:
                    self._matrix = np.vstack([self._matrix, row])

            self._rebuild_faiss_locked()
            self._persist_locked()

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
            self._persist_locked()

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
            self._persist_locked()
            logger.info(
                "gallery.rebuild size=%s backend=%s",
                len(self._user_ids),
                "faiss" if self.using_faiss else "numpy",
            )
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

            # Prefer FAISS whenever the index is ready.
            if self._faiss_index is not None and self._faiss is not None:
                scores, indices = self._faiss_index.search(
                    np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32),
                    k,
                )
                idx = int(indices[0][0])
                score = float(scores[0][0])
                if idx < 0 or idx >= len(self._user_ids):
                    return None
                return GalleryMatch(self._user_ids[idx], score)

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
            self._backend = "faiss"
        except Exception as exc:
            logger.warning("FAISS rebuild failed: %s", exc)
            self._faiss_index = None
            if settings.face_require_faiss:
                raise

    def _ids_path(self) -> Path:
        return self._persist_dir / "user_ids.json"

    def _matrix_path(self) -> Path:
        return self._persist_dir / "embeddings.npy"

    def _persist_locked(self) -> None:
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._ids_path().write_text(json.dumps(self._user_ids), encoding="utf-8")
            if self._matrix is not None:
                np.save(self._matrix_path(), self._matrix)
            elif self._matrix_path().exists():
                os.remove(self._matrix_path())
        except Exception as exc:
            logger.warning("gallery.persist failed: %s", exc)

    def _load_persisted(self) -> None:
        try:
            ids_path = self._ids_path()
            matrix_path = self._matrix_path()
            if not ids_path.exists() or not matrix_path.exists():
                return
            user_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            matrix = np.load(matrix_path)
            if not isinstance(user_ids, list) or matrix.ndim != 2:
                return
            if len(user_ids) != matrix.shape[0]:
                logger.warning("gallery.persist size mismatch — ignoring saved index")
                return
            self._user_ids = [int(x) for x in user_ids]
            self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            self._dim = int(self._matrix.shape[1])
            self._rebuild_faiss_locked()
            logger.info(
                "gallery.loaded size=%s dim=%s backend=%s",
                self.size,
                self._dim,
                "faiss" if self.using_faiss else "numpy",
            )
        except Exception as exc:
            logger.warning("gallery.load failed: %s", exc)
