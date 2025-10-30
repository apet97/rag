from __future__ import annotations

"""
Optional cross-encoder reranking.

AXIOM 5: Use "BAAI/bge-reranker-base" if available. If not installed, skip silently.
Never block retrieval on reranker availability.
"""

import os
from typing import Optional, List, Dict, Any

from loguru import logger

try:
    from FlagEmbedding import FlagReranker
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False
    FlagReranker = None  # type: ignore
    logger.debug("FlagEmbedding not installed. Reranking disabled. Install with: pip install FlagEmbedding")

# Environment-based reranker control: disable in CI or when explicitly requested
RERANK_DISABLED = (
    os.getenv("RERANK_DISABLED", "0") == "1"
    or os.getenv("EMBEDDINGS_BACKEND") == "stub"  # Auto-disable for stub backend
)

_reranker: Optional[object] = None


def _get_reranker() -> Optional[object]:
    """Lazy-load reranker if available."""
    global _reranker
    if RERANK_DISABLED:
        return None
    if not RERANK_AVAILABLE:
        return None

    if _reranker is None:
        try:
            logger.info("Loading reranker model: BAAI/bge-reranker-base")
            _reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
            return None

    return _reranker


def rerank(query: str, docs: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    """
    Rerank documents using cross-encoder if available.
    Falls back to score-based sorting if reranker not available.
    """
    if not docs:
        return []

    reranker = _get_reranker()
    if reranker is None:
        sorted_docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_docs[:topk]

    try:
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = reranker.compute_score(pairs, normalize=True)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        out = []
        for doc, score in ranked[:topk]:
            doc_copy = dict(doc)
            doc_copy["score"] = float(score)
            out.append(doc_copy)
        return out
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Falling back to original scores.")
        sorted_docs = sorted(docs, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_docs[:topk]


def is_available() -> bool:
    """Return whether reranking is available."""
    return RERANK_AVAILABLE and _get_reranker() is not None


def warmup_reranker() -> None:
    """Preload reranker model on startup to avoid first-query latency spike.

    This should be called during API startup to ensure the model is loaded
    before handling requests. Falls back gracefully if reranker is unavailable.
    """
    if RERANK_DISABLED:
        logger.debug("Reranker warmup skipped: disabled via RERANK_DISABLED or stub backend")
        return
    if not RERANK_AVAILABLE:
        logger.debug("Reranker warmup skipped: FlagEmbedding not installed")
        return

    try:
        logger.info("Warming up reranker model (BAAI/bge-reranker-base)...")
        reranker = _get_reranker()
        if reranker is None:
            logger.warning("Reranker warmup: model failed to load, continuing without reranking")
            return

        # Test with a sample pair to ensure model is ready
        test_pairs = [("test query", "test document")]
        reranker.compute_score(test_pairs, normalize=True)
        logger.info("✓ Reranker model warmed up and ready")
    except Exception as e:
        logger.warning(f"Reranker warmup failed: {e}. Continuing without reranking.")
