"""
FAISS Index Manager

Handles loading and caching of FAISS indexes for all namespaces.
Provides thread-safe singleton access to pre-loaded vector indexes with embeddings.

This module encapsulates all index-related operations that were previously in server.py,
including:
- Index loading from disk
- Vector reconstruction and caching
- Multi-namespace index management
- Thread-safe lazy initialization (double-checked locking pattern)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict

import faiss
from loguru import logger


class NamespaceIndex(TypedDict):
    """Type definition for a loaded namespace index."""
    index: faiss.Index          # The FAISS index object
    metas: List[Dict[str, Any]]  # Chunks with embedded vectors cached
    dim: int                     # Vector dimension


class IndexManager:
    """Singleton manager for FAISS indexes with thread-safe lazy initialization."""

    _instance: Optional[IndexManager] = None
    _lock = threading.Lock()
    _indexes: Dict[str, NamespaceIndex] = {}
    _index_normalized: Dict[str, bool] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, index_root: Path, namespaces: List[str]):
        """Initialize the index manager with configuration.

        Args:
            index_root: Root path containing namespace subdirectories with FAISS indexes
            namespaces: List of namespace names to load
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.index_root = index_root
        self.namespaces = namespaces
        self._loaded = False
        self._load_lock = threading.Lock()
        self._initialized = True

    def ensure_loaded(self) -> None:
        """Load all namespace indexes into memory (thread-safe).

        FIX CRITICAL #4: Uses double-checked locking pattern to safely initialize
        _indexes dict in multi-threaded environment. Prevents multiple threads
        from simultaneously attempting to load indexes.
        """
        # First check (fast path, no lock)
        if self._loaded:
            return

        # Second check with lock (slow path, only on first access)
        with self._load_lock:
            # Double-check: another thread may have loaded while we waited for lock
            if self._loaded:
                return

            logger.info("Loading FAISS indexes for all namespaces...")
            for ns in self.namespaces:
                meta_data = json.loads((self.index_root / ns / "meta.json").read_text())
                self._index_normalized[ns] = meta_data.get("normalized", False)
                self._indexes[ns] = self._load_index_for_ns(ns)

            self._loaded = True
            logger.info(f"✓ Loaded {len(self._indexes)} namespaces: {list(self._indexes.keys())}")

    def _load_index_for_ns(self, ns: str) -> NamespaceIndex:
        """Load a single namespace's index from disk.

        Args:
            ns: Namespace name

        Returns:
            NamespaceIndex with FAISS index and pre-cached embeddings

        Raises:
            RuntimeError: If index files not found or embedding reconstruction fails
        """
        root = self.index_root / ns

        # Try .faiss first, then .bin for compatibility
        idx_path = root / "index.faiss"
        if not idx_path.exists():
            idx_path = root / "index.bin"

        meta_path = root / "meta.json"

        if not idx_path.exists() or not meta_path.exists():
            raise RuntimeError(
                f"Index for namespace '{ns}' not found under {root}\n"
                f"Expected: {root / 'index.faiss'} or {root / 'index.bin'}\n"
                f"Expected metadata: {meta_path}"
            )

        logger.info(f"Loading FAISS index for namespace '{ns}' from {idx_path}")
        index = faiss.read_index(str(idx_path))

        # P1: Preload FAISS index with make_direct_map for faster MMR
        # This pre-computes the direct mapping to vectors, eliminating I/O overhead during retrieval
        try:
            if hasattr(index, 'make_direct_map'):
                index.make_direct_map()
                logger.info(f"✓ FAISS index preloaded with make_direct_map for namespace '{ns}'")
            else:
                logger.debug(f"Index type {type(index).__name__} does not support make_direct_map")
        except Exception as e:
            logger.warning(f"Failed to call make_direct_map on FAISS index: {e}")

        metas = json.loads(meta_path.read_text())
        rows = metas.get("rows") or metas.get("chunks", [])

        # PERFORMANCE: Attempt to reconstruct all vectors at startup and cache them.
        # If reconstruction is not supported by the index type, degrade gracefully:
        # keep metas without embeddings and rely on FAISS-only vector search paths.
        logger.info(f"Reconstructing {len(rows)} vectors for namespace '{ns}' (if supported)...")

        chunks_with_embeddings = []
        reconstruction_supported = True

        for i, chunk in enumerate(rows):
            if not reconstruction_supported:
                # Skip attempts once we know it's unsupported
                break
            try:
                # Reconstruct vector from FAISS at position i
                vector = index.reconstruct(i)
                # Add embedding to chunk metadata
                chunk_with_emb = {**chunk, "embedding": vector}
                chunks_with_embeddings.append(chunk_with_emb)
            except Exception as e:
                # Degrade gracefully: stop reconstruction attempts, use raw metas only
                reconstruction_supported = False
                chunks_with_embeddings = []  # discard partial to avoid mixed states
                logger.warning(
                    "FAISS reconstruct() not supported for index type '%s' in namespace '%s': %s. "
                    "Proceeding without cached embeddings (hybrid search may fall back to vector-only).",
                    type(index).__name__, ns, e
                )

        if reconstruction_supported:
            logger.info(f"✓ Cached {len(chunks_with_embeddings)} vectors for namespace '{ns}'")
            metas_out = chunks_with_embeddings
        else:
            metas_out = rows  # No 'embedding' key; safe for JSON serialization

        return {
            "index": index,
            "metas": metas_out,
            "dim": metas.get("dim") or metas.get("dimension", 768)
        }

    def get_index(self, ns: str) -> NamespaceIndex:
        """Get a loaded namespace index.

        Args:
            ns: Namespace name

        Returns:
            NamespaceIndex containing FAISS index and metadata

        Raises:
            KeyError: If namespace not loaded
        """
        self.ensure_loaded()
        return self._indexes[ns]

    def get_all_indexes(self) -> Dict[str, NamespaceIndex]:
        """Get all loaded indexes.

        Returns:
            Dict mapping namespace names to NamespaceIndex objects
        """
        self.ensure_loaded()
        return self._indexes.copy()

    def is_normalized(self, ns: str) -> bool:
        """Check if a namespace's embeddings are L2-normalized.

        Args:
            ns: Namespace name

        Returns:
            True if embeddings are L2-normalized
        """
        self.ensure_loaded()
        return self._index_normalized.get(ns, False)
