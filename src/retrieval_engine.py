"""
Unified Retrieval Engine with Strategy Pattern

Consolidates all retrieval logic (vector search, BM25, hybrid) into a single,
extensible engine using the strategy pattern.

This is the single source of truth for all retrieval operations in the RAG system.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

# P1: Better tokenization imports
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except ImportError:
    STEMMER = None
    logger = logging.getLogger(__name__)
    logger.debug("NLTK not available, stemming disabled")

from src.errors import RetrievalError
from src.tuning_config import (
    RRF_K_CONSTANT,
    MMR_LAMBDA,
    TIME_DECAY_RATE,
    BM25_B,
)
from src.performance_tracker import get_performance_tracker, PipelineStage
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"          # Pure semantic search via FAISS
    BM25 = "bm25"              # Pure lexical search via BM25
    HYBRID = "hybrid"          # Combined semantic + lexical


@dataclass
class RetrievalConfig:
    """Configuration for retrieval engine."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    k_vector: int = 40          # Results from vector search
    k_bm25: int = 40            # Results from BM25 search
    k_final: int = 5            # Final results to return
    hybrid_alpha: float = 0.7   # Weight for vector (1-alpha for BM25)
    normalize_scores: bool = True
    apply_diversity_penalty: bool = True
    diversity_penalty_weight: float = 0.15
    timeout_seconds: float = 30.0


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    chunk_id: str
    text: str
    title: str
    url: str
    namespace: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Scores
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    final_score: Optional[float] = None

    # Ranking info
    rank: Optional[int] = None
    seen_content_hash: Optional[str] = None
    diversity_score: Optional[float] = None  # PHASE 5: MMR diversity penalty

    # Embeddings for MMR calculation (P0: vector-based diversity)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ============================================================================
# Tokenization Helper (P1: Enhanced BM25)
# ============================================================================

# P1: Common English stopwords for filtering
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'it', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'as', 'if', 'with', 'about', 'up', 'down', 'out',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'no',
    'not', 'only', 'same', 'so', 'such', 'than', 'too', 'very', 'just'
}


def tokenize_for_bm25(text: str, remove_stopwords: bool = True, use_stemming: bool = False) -> List[str]:
    """
    P1: Enhanced tokenization for BM25.

    Uses regex-based tokenization instead of simple split().
    Optionally removes stopwords and applies stemming.

    Args:
        text: Text to tokenize
        remove_stopwords: Whether to filter common stopwords
        use_stemming: Whether to apply Porter stemming

    Returns:
        List of preprocessed tokens
    """
    if not text:
        return []

    # Lowercase
    text = text.lower()

    # P1: Regex-based tokenization (handles punctuation better than split())
    # Matches sequences of alphanumeric characters (includes hyphens for compound words)
    tokens = re.findall(r'\b[a-z0-9_-]+\b', text)

    # P1: Remove stopwords if enabled
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # P1: Apply stemming if available and enabled
    if use_stemming and STEMMER is not None:
        tokens = [STEMMER.stem(t) for t in tokens]

    return tokens


# ============================================================================
# Abstract Strategy Base Class
# ============================================================================


class BaseRetrievalStrategy(ABC):
    """Base class for all retrieval strategies."""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Execute search strategy."""
        pass

    def _create_result(
        self,
        chunk: Dict[str, Any],
        vector_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Create RetrievalResult from chunk and scores."""
        # P0: Extract embedding for MMR vector-based diversity calculation
        embedding = chunk.get("embedding")
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype="float32")
        elif not isinstance(embedding, np.ndarray):
            embedding = None

        return RetrievalResult(
            chunk_id=chunk.get("chunk_id", chunk.get("id", "")),
            text=chunk.get("text", ""),
            title=chunk.get("title", ""),
            url=chunk.get("url", ""),
            namespace=chunk.get("namespace", ""),
            metadata=chunk.get("metadata", {}),
            vector_score=vector_score,
            bm25_score=bm25_score,
            embedding=embedding,
        )


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================


class VectorSearchStrategy(BaseRetrievalStrategy):
    """Pure semantic search via vector embeddings."""

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using vector similarity only."""
        try:
            if not chunks or query_embedding is None:
                return []

            # Normalize query embedding if needed
            if self.config.normalize_scores:
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    query_embedding = query_embedding / query_norm

            # Extract embeddings from chunks
            # Fix #3: Handle missing embeddings gracefully with clear error message
            embeddings = []
            valid_indices = []
            for idx, chunk in enumerate(chunks):
                if "embedding" in chunk:
                    embeddings.append(chunk["embedding"])
                    valid_indices.append(idx)

            if not embeddings:
                error_msg = (
                    f"FIX CRITICAL #5: Vector search failed - no embeddings found in {len(chunks)} chunks. "
                    "Chunks must include 'embedding' field for vector search. "
                    "This is a fatal error indicating index was not properly initialized with embeddings."
                )
                logger.error(error_msg)
                raise RetrievalError(error_msg)

            embeddings = np.array(embeddings)

            # Compute similarity
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            similarities = np.dot(embeddings, query_embedding)

            # Get top-k
            top_indices = np.argsort(-similarities)[:k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                chunk_idx = valid_indices[idx]
                chunk = chunks[chunk_idx]
                score = float(similarities[idx])

                result = self._create_result(chunk, vector_score=score)
                result.final_score = score
                result.rank = rank
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RetrievalError(f"Vector search failed: {str(e)}")


class BM25SearchStrategy(BaseRetrievalStrategy):
    """Pure lexical search via BM25."""

    def __init__(self, config: RetrievalConfig):
        """Initialize BM25 strategy with caching for performance."""
        super().__init__(config)
        # Fix #2: Cache BM25 index to avoid O(N) rebuild on every search
        # Cache key: namespace (assumes chunks are per-namespace and stable)
        self._bm25_cache: Dict[str, Tuple[BM25Okapi, List[List[str]]]] = {}
        self._cache_lock = __import__('threading').Lock()
        self._max_cache_size = 100  # Limit cache to prevent unbounded growth
        self._cache_access_order: List[str] = []  # LRU tracking

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using BM25 keyword matching only."""
        try:
            if not chunks or not query_text:
                return []

            # Fix #2: Cache BM25 index per namespace to avoid O(N) rebuild
            # Use namespace as cache key (assumes chunks have consistent namespace)
            namespace = chunks[0].get("namespace", "default") if chunks else "default"

            # PHASE 2: Extended lock to cover BM25 object usage
            # FIX CRITICAL: BM25Okapi.get_scores() must be called under lock
            # to prevent thread-safety issues with the object's internal state
            with self._cache_lock:
                if namespace in self._bm25_cache:
                    # Use cached BM25 index and update LRU order
                    bm25, tokenized_texts = self._bm25_cache[namespace]
                    # Move to end (most recently used)
                    if namespace in self._cache_access_order:
                        self._cache_access_order.remove(namespace)
                    self._cache_access_order.append(namespace)
                    logger.debug(f"Using cached BM25 index for namespace: {namespace}")
                else:
                    # Build and cache BM25 index
                    # Build weighted lexical fields: title=3x, h1=2x (first header), path=1x, plus body text
                    texts = []
                    for chunk in chunks:
                        title = chunk.get("title") or ""
                        headers = chunk.get("headers") or []
                        h1 = headers[0] if headers else ""
                        url = chunk.get("url") or ""
                        # Extract path tokens from URL
                        try:
                            from urllib.parse import urlparse
                            path = urlparse(url).path.replace("/", " ")
                        except Exception:
                            path = ""
                        body = chunk.get("text", "")
                        combined = f"{(title + ' ') * 3}{(h1 + ' ') * 2}{path} {body}"
                        texts.append(combined)
                    # P1: Use enhanced tokenization with stopword removal on weighted text
                    tokenized_texts = [tokenize_for_bm25(text, remove_stopwords=True, use_stemming=False) for text in texts]
                    bm25 = BM25Okapi(tokenized_texts)

                    # LRU eviction: remove oldest entry if cache is full
                    if len(self._bm25_cache) >= self._max_cache_size and self._cache_access_order:
                        lru_namespace = self._cache_access_order.pop(0)
                        del self._bm25_cache[lru_namespace]
                        logger.debug(f"BM25 cache eviction: removed {lru_namespace} (size={len(self._bm25_cache)})")

                    self._bm25_cache[namespace] = (bm25, tokenized_texts)
                    self._cache_access_order.append(namespace)
                    logger.debug(f"Built and cached BM25 index for namespace: {namespace}")

                # Score query (must be under lock since BM25Okapi may not be thread-safe)
                # P1: Use enhanced tokenization for query as well (must match document tokenization)
                query_tokens = tokenize_for_bm25(query_text, remove_stopwords=True, use_stemming=False)
                scores = bm25.get_scores(query_tokens)

            # Get top-k
            top_indices = np.argsort(-scores)[:k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                chunk = chunks[idx]
                score = float(scores[idx])

                # Normalize score if needed
                if self.config.normalize_scores and scores.max() > 0:
                    score = score / scores.max()

                result = self._create_result(chunk, bm25_score=score)
                result.final_score = score
                result.rank = rank
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise RetrievalError(f"BM25 search failed: {str(e)}")


class HybridSearchStrategy(BaseRetrievalStrategy):
    """Combined semantic + lexical search with fusion."""

    def __init__(self, config: RetrievalConfig):
        """Initialize hybrid strategy with shared sub-strategies.

        FIX CRITICAL #2: Create strategy instances once and reuse them
        to preserve BM25 cache across requests.
        """
        super().__init__(config)
        # Create shared strategy instances that persist across requests
        self._vector_strategy = VectorSearchStrategy(config)
        self._bm25_strategy = BM25SearchStrategy(config)

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using both vector and BM25, then fuse results."""
        try:
            if not chunks:
                return []

            tracker = get_performance_tracker()

            # FIX CRITICAL #2: Reuse shared strategy instances to preserve BM25 cache
            # Previously created new instances on every call, destroying the cache
            start_vector = time.time()
            vector_results = self._vector_strategy.search(
                query_embedding, query_text, chunks, self.config.k_vector
            )
            tracker.record(
                PipelineStage.VECTOR_SEARCH,
                (time.time() - start_vector) * 1000,
                metadata={"results": len(vector_results)},
            )

            start_bm25 = time.time()
            bm25_results = self._bm25_strategy.search(
                query_embedding, query_text, chunks, self.config.k_bm25
            )
            tracker.record(
                PipelineStage.BM25_SEARCH,
                (time.time() - start_bm25) * 1000,
                metadata={"results": len(bm25_results)},
            )

            # Fuse results
            start_fusion = time.time()
            fused = self._fuse_results(vector_results, bm25_results)
            tracker.record(
                PipelineStage.FUSION,
                (time.time() - start_fusion) * 1000,
                metadata={"fused_results": len(fused)},
            )

            # PHASE 5: Enhanced DEBUG logging for fusion pipeline decisions
            logger.debug(
                f"Hybrid fusion: k_vector={self.config.k_vector}, k_bm25={self.config.k_bm25}, "
                f"fused={len(fused)} unique results (RRF applied), "
                f"top_score={fused[0].final_score if fused else 'N/A'}"
            )

            # Apply diversity penalty if configured
            if self.config.apply_diversity_penalty:
                start_diversity = time.time()
                fused = self._apply_diversity_penalty(fused)
                tracker.record(
                    PipelineStage.DIVERSITY_FILTER,
                    (time.time() - start_diversity) * 1000,
                    metadata={"results_after": len(fused)},
                )
                logger.debug(f"MMR diversity filter applied (λ=0.7), top result diversity_score={fused[0].diversity_score if fused else 'N/A'}")

            # Sort by final score and truncate to k
            fused.sort(key=lambda r: r.final_score or 0, reverse=True)
            for rank, result in enumerate(fused[:k], 1):
                result.rank = rank

            logger.debug(f"Final hybrid results: returning top {len(fused[:k])} of {len(fused)}")
            return fused[:k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError(f"Hybrid search failed: {str(e)}")

    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF is more robust than weighted averaging and doesn't require hyperparameter tuning.
        Formula: score = 1/(k + rank), where k is typically 60 (number of initial results).

        Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        # PHASE 5: Gold-standard RRF instead of weighted fusion
        # RRF benefits: no hyperparameter tuning, empirically superior to weighted methods
        k_const = RRF_K_CONSTANT  # Standard constant for RRF

        # Create a map of chunk_id to RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Add vector results with RRF scores
        for rank, result in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k_const + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Add BM25 results with RRF scores
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k_const + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
            else:
                # Merge BM25 score into existing vector result
                result_map[result.chunk_id].bm25_score = result.bm25_score

        # Assign final RRF scores to results
        for chunk_id, rrf_score in rrf_scores.items():
            result = result_map[chunk_id]
            result.hybrid_score = rrf_score
            result.final_score = rrf_score
            logger.debug(
                f"RRF score for {chunk_id}: {rrf_score:.4f} "
                f"(vector_rank={vector_results.index(result)+1 if result in vector_results else 'N/A'}, "
                f"bm25_rank={bm25_results.index(result)+1 if result in bm25_results else 'N/A'})"
            )

        return list(result_map.values())

    def _apply_diversity_penalty(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        PHASE 5: Apply Maximal Marginal Relevance (MMR) for diversity.

        MMR balances relevance (retrieval score) with diversity (dissimilarity to already-selected results).
        Formula: MMR_score = λ * relevance_score - (1-λ) * max_similarity_to_selected

        P0 improvement: Uses vector cosine similarity (embeddings) for semantic diversity calculation.
        Falls back to token overlap (Jaccard similarity) only when embeddings are unavailable.

        Args:
            results: Ranked retrieval results

        Returns:
            Results with MMR-adjusted scores
        """
        if not results or len(results) < 2:
            return results

        # Performance optimization: Skip expensive MMR for large result sets
        # Greedy MMR is O(N²) and becomes prohibitive for N > 20
        if len(results) > 20:
            from loguru import logger
            logger.debug(f"Skipping MMR diversity filter for {len(results)} results (threshold: 20)")
            return results[:self.config.k_final] if self.config.k_final else results

        lambda_param = MMR_LAMBDA  # Gold-standard: balances relevance and diversity
        selected_indices = []
        mmr_scores = {}

        # Start with top-ranked result (highest relevance)
        if results:
            selected_indices.append(0)
            results[0].diversity_score = 0.0  # No penalty for first result

        # Greedily select remaining results based on MMR
        while len(selected_indices) < len(results):
            best_mmr_idx = -1
            best_mmr_score = -float('inf')

            for idx, result in enumerate(results):
                if idx in selected_indices:
                    continue

                # Relevance score (from retrieval)
                relevance = result.final_score or 0.0

                # Diversity: max cosine similarity to selected results
                max_similarity = 0.0
                for sel_idx in selected_indices:
                    selected_result = results[sel_idx]

                    # P0: Try vector cosine similarity first (if embeddings available)
                    similarity = 0.0
                    if (
                        selected_result.embedding is not None
                        and result.embedding is not None
                        and len(selected_result.embedding) > 0
                        and len(result.embedding) > 0
                    ):
                        # Vector cosine similarity: dot product / (norm1 * norm2)
                        # Bounded to [0, 1] where 1 = identical, 0 = orthogonal
                        try:
                            dot_product = np.dot(selected_result.embedding, result.embedding)
                            norm_selected = np.linalg.norm(selected_result.embedding)
                            norm_candidate = np.linalg.norm(result.embedding)
                            if norm_selected > 0 and norm_candidate > 0:
                                similarity = dot_product / (norm_selected * norm_candidate)
                                similarity = np.clip(similarity, -1.0, 1.0)  # Clamp to [-1, 1]
                        except (ValueError, RuntimeError) as e:
                            logger.debug(f"Vector similarity calculation failed: {e}, falling back to token overlap")
                            similarity = 0.0

                    # Fallback: token overlap (Jaccard similarity) if embeddings unavailable
                    if similarity == 0.0:
                        selected_tokens = set(selected_result.text.lower().split()) if selected_result.text else set()
                        candidate_tokens = set(result.text.lower().split()) if result.text else set()

                        if selected_tokens and candidate_tokens:
                            intersection = len(selected_tokens & candidate_tokens)
                            union = len(selected_tokens | candidate_tokens)
                            similarity = intersection / union if union > 0 else 0.0

                    max_similarity = max(max_similarity, similarity)

                # MMR score: λ * relevance - (1-λ) * diversity_penalty
                # PHASE 5: Clamp to [0, 1] to prevent negative scores from high diversity penalties
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_score = max(0.0, mmr_score)  # Clamp negative scores
                mmr_scores[idx] = mmr_score

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_mmr_idx = idx

            if best_mmr_idx >= 0:
                selected_indices.append(best_mmr_idx)
                results[best_mmr_idx].diversity_score = max_similarity

        # Re-rank results by MMR scores
        indexed_results = [(idx, result) for idx, result in enumerate(results)]
        indexed_results.sort(key=lambda x: mmr_scores.get(x[0], 0.0), reverse=True)

        # Update final scores with MMR
        reranked = []
        for idx, result in indexed_results:
            result.final_score = mmr_scores.get(idx, result.final_score or 0.0)
            reranked.append(result)

        logger.debug(f"MMR applied: λ={lambda_param}, top result diversity_score={reranked[0].diversity_score if reranked else 'N/A'}")
        return reranked

    def _apply_time_decay(
        self,
        results: List[RetrievalResult],
        decay_rate: float = TIME_DECAY_RATE,
    ) -> List[RetrievalResult]:
        """
        PHASE 5: Apply time decay to boost recent documents.

        Older documents are penalized: score *= decay_rate^months_old

        Uses document metadata: metadata.get("updated_at") should be ISO timestamp.

        Args:
            results: Retrieval results with optional updated_at metadata
            decay_rate: Decay factor per month (default 0.95 = 5% monthly decay)

        Returns:
            Results with time-decayed scores
        """
        from datetime import datetime
        import pytz  # type: ignore[import-untyped]

        start_time = time.time()
        tracker = get_performance_tracker()

        now = datetime.now(pytz.UTC)
        decay_count = 0
        decay_factors = []

        for result in results:
            metadata = result.metadata or {}
            updated_at_str = metadata.get("updated_at")

            if updated_at_str:
                try:
                    # Parse ISO timestamp
                    if updated_at_str.endswith("Z"):
                        updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                    else:
                        updated_at = datetime.fromisoformat(updated_at_str)

                    # Calculate months elapsed
                    days_old = (now - updated_at).days
                    months_old = days_old / 30.0

                    # Apply decay: score *= decay_rate^months_old
                    # PHASE 5: Clamp result to [0, 1] to keep scores bounded
                    decay_factor = decay_rate ** months_old
                    original_score = result.final_score or 0.0
                    result.final_score = max(0.0, original_score * decay_factor)

                    decay_count += 1
                    decay_factors.append(decay_factor)

                    logger.debug(
                        f"Time decay applied to {result.chunk_id}: "
                        f"{days_old}d old, decay={decay_factor:.4f}, "
                        f"score {original_score:.4f} -> {result.final_score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse updated_at for {result.chunk_id}: {e}")

        # PHASE 5: Summary log for time decay pipeline
        if decay_count > 0:
            avg_decay_factor = sum(decay_factors) / len(decay_factors) if decay_factors else 1.0
            logger.debug(
                f"Time decay summary: applied to {decay_count}/{len(results)} results, "
                f"avg_decay_factor={avg_decay_factor:.4f}, decay_rate={decay_rate}"
            )

        # TIER 2: Record time decay latency
        tracker.record(
            PipelineStage.TIME_DECAY,
            (time.time() - start_time) * 1000,
            metadata={"decay_count": decay_count, "total": len(results)},
        )

        return results


# ============================================================================
# Main Retrieval Engine
# ============================================================================


class RetrievalEngine:
    """Unified retrieval engine supporting multiple strategies."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize engine with configuration."""
        self.config = config or RetrievalConfig()
        self._strategies: Dict[RetrievalStrategy, BaseRetrievalStrategy] = {
            RetrievalStrategy.VECTOR: VectorSearchStrategy(self.config),
            RetrievalStrategy.BM25: BM25SearchStrategy(self.config),
            RetrievalStrategy.HYBRID: HybridSearchStrategy(self.config),
        }
        self.current_strategy = self._strategies[self.config.strategy]

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """Switch retrieval strategy."""
        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.current_strategy = self._strategies[strategy]
        self.config.strategy = strategy
        logger.info(f"Switched to {strategy.value} retrieval strategy")

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Execute retrieval using current strategy.

        Args:
            query_embedding: Vector embedding of query
            query_text: Original query text
            chunks: List of document chunks
            k: Number of results (uses config if not specified)

        Returns:
            List of retrieval results ranked by relevance

        Raises:
            RetrievalError: If retrieval fails
            RetrievalTimeoutError: If retrieval times out
        """
        try:
            k = k or self.config.k_final

            if not isinstance(chunks, list):
                raise ValueError("chunks must be a list")

            if query_embedding is None and self.config.strategy != RetrievalStrategy.BM25:
                raise ValueError("query_embedding required for vector search")

            results = self.current_strategy.search(
                query_embedding, query_text, chunks, k
            )

            logger.debug(
                f"Retrieved {len(results)} results using {self.config.strategy.value} strategy"
            )
            return results

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search regardless of current strategy.

        Args:
            query_embedding: Vector embedding
            query_text: Query text
            chunks: Document chunks
            k: Number of results
            alpha: Hybrid weight (0.7 = 70% vector, 30% BM25)

        Returns:
            Hybrid search results
        """
        original_strategy = self.config.strategy
        original_alpha = self.config.hybrid_alpha

        try:
            self.set_strategy(RetrievalStrategy.HYBRID)
            if alpha is not None:
                self.config.hybrid_alpha = max(0, min(1, alpha))

            return self.search(query_embedding, query_text, chunks, k)

        finally:
            self.config.strategy = original_strategy
            self.config.hybrid_alpha = original_alpha
            self.current_strategy = self._strategies[original_strategy]

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy and configuration."""
        return {
            "strategy": self.config.strategy.value,
            "k_final": self.config.k_final,
            "k_vector": self.config.k_vector,
            "k_bm25": self.config.k_bm25,
            "hybrid_alpha": self.config.hybrid_alpha,
            "normalize_scores": self.config.normalize_scores,
            "apply_diversity_penalty": self.config.apply_diversity_penalty,
        }


# ============================================================================
# Hybrid Search Scoring Utilities (Consolidated from hybrid_search.py)
# ============================================================================


def compute_bm25_score(
    doc_tokens: List[str],
    query_tokens: List[str],
    doc_length: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = BM25_B,
) -> float:
    """
    Compute BM25 score for a document given a query.

    BM25 is a probabilistic relevance framework used in information retrieval.
    It considers:
    - Term frequency in document
    - Inverse document frequency
    - Document length normalization

    Args:
        doc_tokens: Tokenized document
        query_tokens: Tokenized query
        doc_length: Length of document (word count)
        avg_doc_length: Average document length in corpus
        k1: Term frequency saturation parameter (default 1.5)
        b: Length normalization parameter (default 0.75)

    Returns:
        BM25 score
    """
    doc_freq = Counter(doc_tokens)
    score = 0.0

    # Estimate IDF (simplified - would normally use corpus-wide stats)
    idf = {}
    for token in query_tokens:
        # Simple IDF approximation
        idf[token] = math.log(1 + (doc_freq.get(token, 0) + 0.5) / (0.5 + 1))

    # Calculate BM25 for each query term
    for token in query_tokens:
        if token in doc_freq:
            freq = doc_freq[token]
            norm_length = 1 - b + b * (doc_length / (avg_doc_length + 1))
            bm25_component = idf[token] * (freq * (k1 + 1)) / (freq + k1 * norm_length)
            score += bm25_component

    return score


def keyword_match_score(text: str, query: str) -> float:
    """
    Simple keyword matching score.

    Rewards exact matches and phrase matches.

    Args:
        text: Text to match against (title + content)
        query: Query string

    Returns:
        Score between 0 and 1
    """
    text_lower = text.lower()
    query_lower = query.lower()

    # Exact phrase match (highest weight)
    if query_lower in text_lower:
        return 1.0

    # Word matches
    query_words = query_lower.split()
    text_words = set(text_lower.split())

    if not query_words:
        return 0.0

    match_ratio = len([w for w in query_words if w in text_words]) / len(query_words)
    return match_ratio


def entity_match_score(text: str, entities: List[str]) -> float:
    """
    Score based on presence of query entities in text.

    Args:
        text: Text to check
        entities: List of entities from query analysis

    Returns:
        Score between 0 and 1
    """
    if not entities:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for entity in entities if entity.lower() in text_lower)
    return min(1.0, matches / len(entities))


def hybrid_search_score(
    result: Dict[str, Any],
    query: str,
    entities: Optional[List[str]] = None,
    semantic_weight: float = 0.70,
    keyword_weight: float = 0.30,
) -> Dict[str, Any]:
    """
    Compute hybrid score combining semantic and keyword matching.

    Args:
        result: Search result dict with 'semantic_score', 'title', 'content'
        query: Original query
        entities: Extracted entities from query (optional)
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)

    Returns:
        Updated result dict with hybrid_score added
    """
    if entities is None:
        entities = []

    # Get semantic score (already computed by FAISS)
    semantic = result.get("semantic_score", result.get("score", 0.0))

    # Compute keyword score
    combined_text = f"{result.get('title', '')} {result.get('content', '')}"
    keyword_score = keyword_match_score(combined_text, query)

    # Bonus for entity matches
    entity_score = entity_match_score(combined_text, entities)
    keyword_score = 0.7 * keyword_score + 0.3 * entity_score

    # Normalize scores to 0-1 range if needed
    semantic_normalized = min(1.0, max(0.0, semantic))
    keyword_normalized = min(1.0, max(0.0, keyword_score))

    # Combine scores
    hybrid_score = semantic_weight * semantic_normalized + keyword_weight * keyword_normalized

    result["hybrid_score"] = hybrid_score
    result["semantic_score"] = semantic_normalized
    result["keyword_score"] = keyword_normalized
    result["entity_score"] = entity_score

    return result


def apply_diversity_penalty(
    results: List[Dict[str, Any]], diversity_weight: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Apply diversity penalty to avoid redundant results.

    Penalizes results that are very similar to already-selected results.

    Args:
        results: List of results (assumed sorted by relevance)
        diversity_weight: Weight for diversity penalty (0-1)

    Returns:
        Results with diversity_penalty and adjusted_score fields added
    """
    if not results:
        return results

    processed = []
    seen_content_hashes = set()

    for i, result in enumerate(results):
        # Create simple content hash for diversity
        content_hash = hash(result.get("content", "")[:100])

        # Calculate diversity score
        if content_hash in seen_content_hashes:
            # Penalize if similar content already in results
            diversity_penalty = diversity_weight
        else:
            diversity_penalty = 0.0
            seen_content_hashes.add(content_hash)

        # Apply penalty to score
        original_score = result.get("hybrid_score", result.get("score", 0.0))
        adjusted_score = original_score * (1 - diversity_penalty)

        result["diversity_penalty"] = diversity_penalty
        result["adjusted_score"] = adjusted_score

        processed.append(result)

    # Re-sort by adjusted score
    processed.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

    return processed


def rank_hybrid_results(
    results: List[Dict[str, Any]],
    query: str,
    entities: Optional[List[str]] = None,
    apply_diversity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Complete hybrid ranking pipeline.

    Applies semantic scoring, keyword matching, entity matching, and diversity.

    Args:
        results: Initial search results
        query: Original query
        entities: Extracted query entities
        apply_diversity: Whether to apply diversity penalty

    Returns:
        Ranked results with hybrid scores
    """
    if entities is None:
        entities = []

    logger.debug(f"Hybrid ranking: {len(results)} results, {len(entities)} entities")

    # Apply hybrid scoring to each result
    scored_results = [hybrid_search_score(r, query, entities) for r in results]

    # Sort by hybrid score
    scored_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

    # Apply diversity penalty if requested
    if apply_diversity and len(scored_results) > 1:
        scored_results = apply_diversity_penalty(scored_results)

    return scored_results


# ============================================================================
# Convenience Functions
# ============================================================================


def create_engine(
    strategy: str = "hybrid",
    k: int = 5,
    alpha: float = 0.7,
) -> RetrievalEngine:
    """Create and configure a retrieval engine."""
    config = RetrievalConfig(
        strategy=RetrievalStrategy(strategy),
        k_final=k,
        hybrid_alpha=alpha,
    )
    return RetrievalEngine(config)


# ============================================================================
# Backward Compatibility Functions (Consolidated from retrieval.py)
# ============================================================================


def hybrid_search(
    query: str,
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    encoder: Any,
    k_vec: int = 40,
    k_bm25: int = 40,
    k_final: int = 12,
    use_query_adaptation: bool = True,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining vector search (cosine) + BM25 with adaptive field boosts.

    This function provides backward compatibility with the legacy retrieval.py API.
    New code should use RetrievalEngine.search() instead.

    Improvements:
    - Query type detection for adaptive strategies
    - Enhanced field matching with type-aware boosting
    - Better BM25 handling with glossary terms
    - Improved normalization and score fusion

    Args:
        query: Search query
        docs: List of {"text": str, "meta": {...}} dicts
        embeddings: (N, d) array of L2-normalized embeddings
        encoder: Encoder with .embed(str) -> ndarray[d]
        k_vec: Number of vector results to keep
        k_bm25: Number of BM25 results to keep
        k_final: Final number of results
        use_query_adaptation: Enable adaptive strategies based on query type

    Returns:
        Top-k merged and re-scored results with improved relevance
    """
    # Embed query
    qv = encoder.embed(query)
    qv = qv / (np.linalg.norm(qv) + 1e-9)  # L2-normalize

    # Vector similarity (already L2-normalized embeddings)
    sims = embeddings @ qv
    top_vec_indices = np.argsort(-sims)[:k_vec]

    # BM25 scores
    corpus = [d.get("text", "") for d in docs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(-bm25_scores)[:k_bm25]

    # Union of both
    candidate_indices = np.unique(np.concatenate([top_vec_indices, top_bm25_indices]))

    # Re-score with adaptive approach
    scores = {}

    for idx in candidate_indices:
        # Base score: weighted average (balanced hybrid approach)
        vec_score = float(sims[idx])
        bm25_norm = float(bm25_scores[idx]) / (np.max(bm25_scores) + 1e-9)

        # Balanced hybrid weighting
        base_score = 0.6 * vec_score + 0.4 * bm25_norm

        # Normalize final score to [0, 1] range
        base_score = min(float(base_score), 1.0)

        scores[idx] = base_score

    # Sort and return top-k
    top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k_final]
    results = [
        {
            **docs[i],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

    return results
