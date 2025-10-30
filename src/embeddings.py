from __future__ import annotations

"""
Unified Embeddings Module

Provides synchronous embedding operations with:
- SentenceTransformer (E5 model) with correct prefixes and L2 normalization
- Thread-safe singleton embedder with double-check locking
- Backward compatibility with legacy encode module API
"""

import os
import threading
import numpy as np
from typing import Optional, List, Dict, Any, Union
from loguru import logger

# Backend selection
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "real")

# Conditional import: only import SentenceTransformer when not using stub backend
# This prevents CI failures when sentence-transformers is not installed
if EMBEDDINGS_BACKEND != "stub":
    from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")


def _resolve_embedding_dim() -> int:
    """
    Returns the actual embedding dimension for the active backend.

    - stub: defaults to 384, override with STUB_EMBEDDING_DIM
    - real: uses EMBEDDING_DIM env var if set, otherwise 768 default

    This ensures dimension consistency across encode_texts, embed_query,
    encode_weighted_variants, and zero-vector fallbacks.
    """
    if EMBEDDINGS_BACKEND == "stub":
        return int(os.getenv("STUB_EMBEDDING_DIM", "384"))
    return int(os.getenv("EMBEDDING_DIM", "768"))


# Single source of truth for embedding dimension
EMBEDDING_DIM = _resolve_embedding_dim()

_embedder: Optional[Any] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Any:
    """Get or load the global embedder instance (thread-safe).

    Uses double-check locking pattern to ensure thread-safe initialization
    without performance penalty for repeated access.

    Supports two backends:
    - "real" (default): SentenceTransformer model
    - "stub" (CI testing): Lightweight deterministic embedder
    """
    global _embedder

    # First check (no lock - fast path)
    if _embedder is not None:
        return _embedder

    # Second check with lock (slow path - only on first access)
    with _embedder_lock:
        # Double-check pattern: another thread may have initialized while waiting
        if _embedder is None:
            if EMBEDDINGS_BACKEND == "stub":
                # CI/testing mode: use stub embedder
                from src.embeddings_stub import get_stub_embedder
                logger.info("Loading stub embedder (CI mode, EMBEDDINGS_BACKEND=stub)")
                _embedder = get_stub_embedder()
                logger.info("✓ Stub embedder loaded")
            else:
                # Production mode: use real SentenceTransformer
                logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
                # SECURITY: Do not use trust_remote_code=True (RCE risk)
                # Only vetted models from official sources should be used
                _embedder = SentenceTransformer(EMBEDDING_MODEL)
                _embedder.max_seq_length = 512
                logger.info(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")

    return _embedder


def embed_passages(texts: List[str]) -> np.ndarray:
    """Embed passages with E5 'passage: ' prefix and L2 normalization.

    Note: Stub backend omits prefixes and returns deterministic vectors.

    Args:
        texts: List of passage texts to embed

    Returns:
        L2-normalized embeddings as float32 array (batch_size, D) where D = EMBEDDING_DIM
    """
    prefixed = [f"passage: {text.strip()}" for text in texts]
    embedder = get_embedder()
    embeddings: np.ndarray = embedder.encode(prefixed, convert_to_numpy=True).astype(np.float32)
    # L2-normalize: divide by norm + epsilon to avoid division by zero
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings


def embed_query(text: str) -> np.ndarray:
    """Embed query with E5 'query: ' prefix and L2 normalization.

    Note: Stub backend omits prefixes and returns deterministic vectors.

    Args:
        text: Query text to embed

    Returns:
        L2-normalized embedding as float32 array (1, D) where D = EMBEDDING_DIM
    """
    prefixed = f"query: {text.strip()}"
    embedder = get_embedder()
    embedding: np.ndarray = embedder.encode([prefixed], convert_to_numpy=True).astype(np.float32)
    # L2-normalize
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)
    return embedding


# ============================================================================
# Backward Compatibility Functions (from legacy encode.py)
# ============================================================================


def encode_query(text: str) -> np.ndarray:
    """
    Encode single query with L2 normalization.

    Backward compatible with legacy encode module. Uses SentenceTransformer
    instead of Ollama HTTP API for better performance and reliability.

    Args:
        text: Query text to encode

    Returns:
        L2-normalized embedding vector as float32 array
    """
    return embed_query(text)


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Batch encode texts with L2 normalization.

    Backward compatible with legacy encode module. Uses SentenceTransformer
    instead of Ollama HTTP API for better performance and reliability.
    Returns matrix of shape (len(texts), EMBEDDING_DIM).

    Args:
        texts: List of texts to encode

    Returns:
        L2-normalized embeddings as float32 array (batch_size, D) where D = EMBEDDING_DIM
    """
    return embed_passages(texts)


def embed_queries(texts: List[str]) -> np.ndarray:
    """Embed multiple queries with E5 'query: ' prefix and L2 normalization.

    Note: Stub backend omits prefixes and returns deterministic vectors.

    Args:
        texts: List of query texts to embed

    Returns:
        L2-normalized embeddings as float32 array (batch_size, D) where D = EMBEDDING_DIM
    """
    prefixed = [f"query: {text.strip()}" for text in texts]
    embedder = get_embedder()
    embeddings: np.ndarray = embedder.encode(prefixed, convert_to_numpy=True).astype(np.float32)
    # L2-normalize: divide by norm + epsilon to avoid division by zero
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings


def encode_weighted_variants(variants: List[Dict[str, Any]]) -> np.ndarray:
    """
    Encode structured query variants with weights and return weighted averaged embedding.

    Each variant should be a dict with:
    - text (str): The variant text (query, not passage)
    - weight (float): Weight for this variant (0-1, where 1.0 is highest influence)

    Internally:
    1. Encode each variant as a QUERY (using "query: " prefix for real backend)
    2. Apply weight to each encoded vector
    3. Average the weighted vectors
    4. L2-normalize the result

    This allows query expansions to be weighted by confidence/source,
    e.g., original query 1.0, boost_terms 0.9, glossary 0.8.

    Args:
        variants: List of dicts with {text: str, weight: float}

    Returns:
        L2-normalized weighted average embedding (1, D) where D = EMBEDDING_DIM
    """
    if not variants:
        return np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

    texts = [v.get("text", "") for v in variants]
    weights = [v.get("weight", 1.0) for v in variants]

    # Encode all variants as QUERIES (with "query: " prefix for real backend)
    embeddings = embed_queries(texts)  # (n_variants, D), L2-normalized

    # Apply weights: scale each vector by its weight
    weighted_embeddings: np.ndarray = np.array([
        embeddings[i] * weights[i]
        for i in range(len(embeddings))
    ], dtype=np.float32)

    # Average the weighted vectors
    avg_embedding: np.ndarray = np.mean(weighted_embeddings, axis=0, keepdims=True).astype(np.float32)

    # L2-normalize the result
    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding, axis=1, keepdims=True) + 1e-12)

    return avg_embedding


def warmup() -> None:
    """
    Test embedding model readiness.

    Loads the embedding model to ensure it's available and working.
    Part of application startup validation.
    """
    logger.info(f"Testing embedding model: {EMBEDDING_MODEL}...")
    try:
        embedder = get_embedder()
        # Test with a simple embedding
        test_embedding = embedder.encode(["test"], convert_to_numpy=True)
        dim = test_embedding.shape[1]
        logger.info(f"✓ Embedding model ready: {EMBEDDING_MODEL} (dim={dim})")
    except Exception as e:
        logger.error(f"✗ Embedding model test failed: {e}")
        raise
