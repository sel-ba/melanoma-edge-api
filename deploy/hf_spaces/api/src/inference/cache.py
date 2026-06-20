from __future__ import annotations

import time
from collections import OrderedDict


class PredictionCache:
    """Simple LRU cache with TTL for prediction responses."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[object, float]] = OrderedDict()

    def get(self, key: str) -> object | None:
        item = self._store.get(key)
        if item is None:
            return None
        value, timestamp = item
        if self._is_expired(timestamp):
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        now = time.time()
        self._store[key] = (value, now)
        self._store.move_to_end(key)
        self._evict()

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    def _evict(self) -> None:
        expired_keys = [
            key for key, (_, ts) in self._store.items() if self._is_expired(ts)
        ]
        for key in expired_keys:
            self._store.pop(key, None)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)
