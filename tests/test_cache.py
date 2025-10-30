"""
Test response caching module.

Tests LRU eviction, TTL expiration, thread-safety, and hit rates.
"""

import time
import pytest
from src.cache import LRUResponseCache, init_cache, get_cache


class TestCacheBasics:
    """Test basic cache operations."""

    def test_cache_get_miss(self):
        """Cache miss returns None."""
        cache = LRUResponseCache(max_size=10)
        result = cache.get("query1", 5)
        assert result is None

    def test_cache_set_and_get(self):
        """Cache stores and retrieves responses."""
        cache = LRUResponseCache(max_size=10)
        response = {"results": [{"rank": 1, "score": 0.9}]}

        cache.set("query1", 5, response)
        retrieved = cache.get("query1", 5)

        assert retrieved is not None
        assert retrieved["results"] == response["results"]

    def test_cache_key_includes_namespace(self):
        """Different namespaces produce different cache keys."""
        cache = LRUResponseCache(max_size=10)
        response1 = {"results": [{"rank": 1, "score": 0.9}]}
        response2 = {"results": [{"rank": 1, "score": 0.8}]}

        cache.set("query1", 5, response1, namespace="ns1")
        cache.set("query1", 5, response2, namespace="ns2")

        retrieved_ns1 = cache.get("query1", 5, namespace="ns1")
        retrieved_ns2 = cache.get("query1", 5, namespace="ns2")

        assert retrieved_ns1["results"][0]["score"] == 0.9
        assert retrieved_ns2["results"][0]["score"] == 0.8

    def test_cache_key_includes_k(self):
        """Different k values produce different cache keys."""
        cache = LRUResponseCache(max_size=10)
        response_k3 = {"results": [{"rank": 1}, {"rank": 2}, {"rank": 3}]}
        response_k5 = {"results": [{"rank": 1}, {"rank": 2}, {"rank": 3}, {"rank": 4}, {"rank": 5}]}

        cache.set("query1", 3, response_k3)
        cache.set("query1", 5, response_k5)

        retrieved_k3 = cache.get("query1", 3)
        retrieved_k5 = cache.get("query1", 5)

        assert len(retrieved_k3["results"]) == 3
        assert len(retrieved_k5["results"]) == 5


class TestCacheTTL:
    """Test time-to-live expiration."""

    def test_cache_respects_ttl(self):
        """Expired entries are not returned."""
        cache = LRUResponseCache(max_size=10, default_ttl=1)
        response = {"results": [{"rank": 1}]}

        cache.set("query1", 5, response, ttl=1)

        # Before expiration
        retrieved = cache.get("query1", 5)
        assert retrieved is not None

        # After expiration
        time.sleep(1.1)
        retrieved = cache.get("query1", 5)
        assert retrieved is None

    def test_cache_custom_ttl(self):
        """Custom TTL overrides default."""
        cache = LRUResponseCache(max_size=10, default_ttl=60)
        response = {"results": [{"rank": 1}]}

        cache.set("query1", 5, response, ttl=1)

        time.sleep(1.1)
        retrieved = cache.get("query1", 5)
        assert retrieved is None


class TestCacheLRUEviction:
    """Test least-recently-used eviction."""

    def test_cache_evicts_lru_when_full(self):
        """LRU item is evicted when cache exceeds max_size."""
        cache = LRUResponseCache(max_size=3)

        # Fill cache
        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})

        assert cache.stats()["size"] == 3

        # Add one more (should evict q1)
        cache.set("q4", 5, {"id": 4})

        assert cache.stats()["size"] == 3
        assert cache.get("q1", 5) is None  # q1 was evicted
        assert cache.get("q2", 5) is not None
        assert cache.get("q3", 5) is not None
        assert cache.get("q4", 5) is not None

    def test_cache_access_order_updated_on_hit(self):
        """Accessing an item updates its recency."""
        cache = LRUResponseCache(max_size=3)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})

        # Access q1 (makes it recently used)
        cache.get("q1", 5)

        # Add q4 (should evict q2, not q1)
        cache.set("q4", 5, {"id": 4})

        assert cache.get("q1", 5) is not None  # q1 still there (was accessed)
        assert cache.get("q2", 5) is None  # q2 was evicted (least recently used)
        assert cache.get("q3", 5) is not None
        assert cache.get("q4", 5) is not None


class TestCacheStats:
    """Test cache statistics."""

    def test_cache_hit_rate_calculation(self):
        """Cache tracks hit rate correctly."""
        cache = LRUResponseCache(max_size=10)

        # Add two entries
        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})

        # 2 misses
        cache.get("q3", 5)
        cache.get("q4", 5)

        # 3 hits
        cache.get("q1", 5)
        cache.get("q2", 5)
        cache.get("q1", 5)

        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate_pct"] == 60.0

    def test_cache_eviction_counter(self):
        """Cache tracks eviction count."""
        cache = LRUResponseCache(max_size=2)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})  # Evicts q1
        cache.set("q4", 5, {"id": 4})  # Evicts q2
        cache.set("q5", 5, {"id": 5})  # Evicts q3

        stats = cache.stats()
        assert stats["evictions"] == 3

    def test_cache_clear(self):
        """Clear empties the cache."""
        cache = LRUResponseCache(max_size=10)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})

        assert cache.stats()["size"] == 2

        cache.clear()

        assert cache.stats()["size"] == 0
        assert cache.get("q1", 5) is None


class TestGlobalCache:
    """Test global cache singleton."""

    def test_init_cache_creates_singleton(self):
        """init_cache creates a global cache instance."""
        cache1 = init_cache(max_size=100)
        cache2 = get_cache()

        assert cache1 is cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
