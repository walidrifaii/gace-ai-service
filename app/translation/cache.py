"""Thread-safe, in-memory LRU cache for translated strings.

Keyed on (text, source, target) so the same message is never sent through
the translation engine twice. A process-local dict is enough here — the
cache is naturally rebuilt on restart. Swap this for Redis later by
implementing the same get/set interface if the service is ever scaled out
horizontally.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedTranslation:
    translated_text: str
    detected_source: str


def make_cache_key(text: str, source: str, target: str) -> str:
    raw = f"{source}|{target}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TranslationCache:
    """A minimal thread-safe LRU cache keyed by translation request."""

    def __init__(self, max_size: int = 5000) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()
        self._store: OrderedDict[str, CachedTranslation] = OrderedDict()

    def get(self, key: str) -> CachedTranslation | None:
        if self._max_size <= 0:
            return None
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def set(self, key: str, value: CachedTranslation) -> None:
        if self._max_size <= 0:
            return
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
