from __future__ import annotations

import time

from src.inference.cache import PredictionCache


class TestPredictionCache:
    """Tests for LRU + TTL prediction cache."""

    def test_set_and_get(self) -> None:
        cache = PredictionCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self) -> None:
        cache = PredictionCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self) -> None:
        cache = PredictionCache(max_size=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a" (oldest)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4

    def test_ttl_expiration(self) -> None:
        cache = PredictionCache(max_size=10, ttl_seconds=0.1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_overwrite_existing_key(self) -> None:
        cache = PredictionCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_access_refreshes_lru_position(self) -> None:
        cache = PredictionCache(max_size=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to move it to end
        cache.get("a")
        # Adding "d" should now evict "b" (oldest after "a" was refreshed)
        cache.set("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
