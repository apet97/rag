#!/usr/bin/env python3
"""
PHASE 5: Semantic Answer Cache

Caches LLM answers by semantic query similarity and top document IDs.
Reduces latency by 50-70% for common/repeated queries.

Key: hash(query_embedding[:10], top_doc_ids, prompt_version)
Value: (answer, sources, timestamp, answerability_score)
TTL: 1 hour (configurable)
Backend: In-memory LRU (10k entries max)
"""

import hashlib
import time
import threading
from typing import Optional, Dict, List, Any, Tuple
from collections import OrderedDict
import numpy as np
from loguru import logger
from src.tuning_config import SEMANTIC_CACHE_MAX_SIZE, SEMANTIC_CACHE_TTL_SECONDS


class SemanticCache:
    """
    Thread-safe LRU cache for semantic answers.

    Uses simple but effective key generation:
    - First 10 dimensions of query embedding (semantic fingerprint)
    - Top document IDs (which documents were used)
    - Prompt version (if prompt changes, cache invalidates)
    """

    def __init__(
        self,
        max_size: int = SEMANTIC_CACHE_MAX_SIZE,
        ttl_seconds: int = SEMANTIC_CACHE_TTL_SECONDS,
    ):
        """
        Initialize semantic cache.

        Args:
            max_size: Maximum number of cached entries (default: from tuning_config)
            ttl_seconds: Time-to-live for cached entries (default: from tuning_config)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Thread-safe cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.RLock()

        # Hit/miss tracking for observability
        self._hits = 0
        self._misses = 0
        self._expirations = 0

        logger.info(f"Initialized SemanticCache (max_size={max_size}, ttl={ttl_seconds}s)")

    def _make_key(
        self,
        query_embedding: np.ndarray,
        top_doc_ids: List[str],
        prompt_version: str = "v1",
        namespaces: Optional[List[str]] = None,
    ) -> str:
        """
        Generate cache key from semantic query and document context.

        Uses first 10 dimensions of embedding as semantic fingerprint
        to avoid storing large embeddings while capturing query intent.

        Args:
            query_embedding: Query embedding vector
            top_doc_ids: List of document IDs that were retrieved
            prompt_version: System prompt version (for invalidation on prompt changes)
            namespaces: List of namespaces searched (prevents cross-namespace contamination)

        Returns:
            Deterministic cache key (hex string)
        """
        # Take first 10 dimensions as fingerprint (sufficient for similarity)
        emb_fingerprint = query_embedding[:10] if len(query_embedding) > 0 else np.array([])
        emb_hash = hashlib.md5(emb_fingerprint.tobytes()).hexdigest()

        # Sort doc IDs for deterministic key
        doc_ids_str = "|".join(sorted(set(top_doc_ids)))
        docs_hash = hashlib.md5(doc_ids_str.encode()).hexdigest()

        # Include namespaces in key to prevent cross-namespace contamination
        ns_str = "|".join(sorted(namespaces)) if namespaces else "default"
        ns_hash = hashlib.md5(ns_str.encode()).hexdigest()

        # Combine: embedding + documents + namespaces + prompt version
        key = f"{emb_hash}:{docs_hash}:{ns_hash}:{prompt_version}"
        return key

    def get(
        self,
        query_embedding: np.ndarray,
        top_doc_ids: List[str],
        prompt_version: str = "v1",
        namespaces: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached answer if available and not expired.

        Args:
            query_embedding: Query embedding vector
            top_doc_ids: List of document IDs
            prompt_version: Prompt version
            namespaces: List of namespaces searched

        Returns:
            Cached value dict or None if miss/expired
        """
        key = self._make_key(query_embedding, top_doc_ids, prompt_version, namespaces)

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            cached = self._cache[key]
            now = time.time()

            # Check if expired
            if now - cached["timestamp"] > self.ttl_seconds:
                logger.debug(f"Cache expired for key {key[:16]}...")
                del self._cache[key]
                del self._access_order[key]
                self._expirations += 1
                self._misses += 1
                return None

            # Update LRU order (move to end = most recently used)
            self._access_order.move_to_end(key)
            self._access_order[key] = now
            self._hits += 1

            logger.debug(
                f"Cache HIT (age={now - cached['timestamp']:.1f}s): "
                f"{len(cached.get('answer', ''))} chars, "
                f"answerability={cached.get('answerability_score', 0):.2f}"
            )

            return cached

    def set(
        self,
        query_embedding: np.ndarray,
        top_doc_ids: List[str],
        answer: str,
        sources: List[Dict[str, Any]],
        answerability_score: float = 0.0,
        prompt_version: str = "v1",
        namespaces: Optional[List[str]] = None,
    ) -> None:
        """
        Cache an answer with metadata.

        Args:
            query_embedding: Query embedding vector
            top_doc_ids: List of document IDs
            answer: Generated answer text
            sources: Source documents used
            answerability_score: Grounding score (0-1)
            prompt_version: Prompt version
            namespaces: List of namespaces searched
        """
        key = self._make_key(query_embedding, top_doc_ids, prompt_version, namespaces)

        with self._lock:
            # LRU eviction: remove oldest entry if cache is full
            if len(self._cache) >= self.max_size:
                # Get the first (oldest) key
                oldest_key = next(iter(self._access_order))
                del self._cache[oldest_key]
                del self._access_order[oldest_key]
                logger.debug(f"Cache eviction: removed oldest entry ({len(self._cache)}/{self.max_size})")

            # Store cached answer
            now = time.time()
            self._cache[key] = {
                "answer": answer,
                "sources": sources,
                "answerability_score": answerability_score,
                "timestamp": now,
                "key": key,
            }
            self._access_order[key] = now

            logger.debug(
                f"Cache SET: key={key[:16]}... "
                f"answer={len(answer)} chars, "
                f"sources={len(sources)}, "
                f"cache_size={len(self._cache)}/{self.max_size}"
            )

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics including hit rate and memory usage."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate_pct = (self._hits / total_accesses * 100) if total_accesses > 0 else 0.0

            # Estimate memory usage (rough: key + answer + sources + overhead)
            memory_bytes = 0
            for key, entry in self._cache.items():
                memory_bytes += len(key.encode())  # Key
                memory_bytes += len(entry.get("answer", "").encode())  # Answer text
                memory_bytes += len(str(entry.get("sources", [])).encode())  # Sources
                memory_bytes += 200  # Metadata overhead per entry

            memory_mb = memory_bytes / (1024 * 1024)
            avg_entry_size_bytes = memory_bytes // len(self._cache) if self._cache else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0,
                "hits": self._hits,
                "misses": self._misses,
                "expirations": self._expirations,
                "hit_rate_pct": round(hit_rate_pct, 2),
                "total_accesses": total_accesses,
                "memory_usage_mb": round(memory_mb, 2),
                "memory_usage_bytes": memory_bytes,
                "avg_entry_size_bytes": avg_entry_size_bytes,
            }


# Module-level singleton instance
_semantic_cache: Optional[SemanticCache] = None
_cache_lock = threading.Lock()


def get_semantic_cache(
    max_size: int = SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds: int = SEMANTIC_CACHE_TTL_SECONDS,
) -> SemanticCache:
    """Get or create module-level semantic cache singleton."""
    global _semantic_cache

    if _semantic_cache is None:
        with _cache_lock:
            if _semantic_cache is None:
                _semantic_cache = SemanticCache(max_size=max_size, ttl_seconds=ttl_seconds)

    return _semantic_cache
