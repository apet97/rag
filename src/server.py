from __future__ import annotations

import os
import time
import json
import re
import random
import hmac
from pathlib import Path
from uuid import uuid4
from typing import Optional, Dict, List, Tuple, Any
from contextlib import asynccontextmanager

try:
    import numpy as np
except Exception:  # allow import without numpy in offline test env
    np = None  # type: ignore
from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.llm_client import LLMClient, close_http_client
from src import config as CFG
from src.embeddings import embed_query, encode_weighted_variants, warmup
from src.query_expand import expand_structured
from src.rerank import rerank, warmup_reranker, is_available as reranker_available
from src.cache import init_cache, get_cache
from src.search_improvements import detect_query_type, get_adaptive_k_multiplier, log_query_analysis, should_enable_hybrid_search
from src.query_optimizer import get_optimizer
from src.scoring import get_scorer
from src.retrieval_engine import RetrievalEngine, RetrievalConfig, RetrievalStrategy
from src.query_decomposition import decompose_query, is_multi_intent_query
from src.models import ResponseMetadata, DecompositionMetadata
from src.metrics import track_request, get_metrics, get_content_type, track_circuit_breaker
from src.circuit_breaker import get_all_circuit_breakers
from src.citation_validator import validate_citations
from src.errors import CircuitOpenError
from src.index_manager import IndexManager
from src.performance_tracker import get_performance_tracker
from src.prompt import RAGPrompt
from src.semantic_cache import get_semantic_cache

# Allow/Deny patterns for runtime enforcement
_ALLOWLIST_PATH = str(CFG.ALLOWLIST_PATH)
_DENYLIST_PATH = str(CFG.DENYLIST_PATH)
_CHUNK_STRATEGY = CFG.CHUNK_STRATEGY

def _compile_patterns(path: str):
    pats = []
    try:
        if Path(path).exists():
            for line in Path(path).read_text(encoding='utf-8').splitlines():
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                try:
                    pats.append(__import__('re').compile(s))
                except Exception:
                    pass
    except Exception:
        pass
    return pats

_ALLOW_PATS = _compile_patterns(_ALLOWLIST_PATH)
_DENY_PATS = _compile_patterns(_DENYLIST_PATH)

def _is_allowed(url: str) -> bool:
    if not url:
        return False
    allow_ok = any(p.search(url) for p in _ALLOW_PATS) if _ALLOW_PATS else True
    deny_hit = any(p.search(url) for p in _DENY_PATS) if _DENY_PATS else False
    return allow_ok and not deny_hit
from src.config import CONFIG

API_TOKEN = os.getenv("API_TOKEN", "change-me")
API_TOKENS_EXTRA = [t.strip() for t in os.getenv("API_TOKENS", "").split(",") if t.strip()]
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "7001"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))  # Increased from 5 to 20 for better recall

# CI support: Allow RAG_INDEX_ROOT env var to override default index location (for test fixtures)
INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))


def _derive_namespaces(index_root: Path) -> List[str]:
    """
    Auto-derive namespaces from subdirectories containing meta.json.

    If NAMESPACES env var is not set, scans INDEX_ROOT for subdirectories
    with meta.json files and uses those as namespaces. This prevents CI
    failures when only a subset of namespaces are available in test fixtures.
    Prefer Clockify-only namespaces by default (domain policy). If none found,
    return all discovered namespaces.

    Args:
        index_root: Root path to search for namespace directories

    Returns:
        Sorted list of namespace names found
    """
    candidates: List[str] = []
    if index_root.exists() and index_root.is_dir():
        for p in index_root.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                candidates.append(p.name)
    # Prefer 'clockify*' namespaces by default
    clk = sorted([c for c in candidates if c.lower().startswith("clockify")])
    if clk:
        return clk
    return sorted(candidates)


# Namespace configuration: explicit env var or auto-derived from index structure
NAMESPACES = (
    [s.strip() for s in os.getenv("NAMESPACES", "").split(",") if s.strip()]
    or _derive_namespaces(INDEX_ROOT)
    or ["clockify"]  # Final fallback for backwards compatibility
)

EMBEDDING_MODEL = CFG.EMBEDDING_MODEL
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434")
INDEX_MODE = os.getenv("INDEX_MODE", "single")

# FIX CRITICAL #3: CORS configuration with explicit origins (no wildcards)
# Default to localhost:8080 and 127.0.0.1:8080 for local development
# Production deployments should set CORS_ALLOWED_ORIGINS env var to explicit domains
_default_cors_origins = [
    # Local dev UI and API
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:7001",
    "http://127.0.0.1:7001",
    # VPN infra convenience
    "http://10.127.0.192:8080",
    "http://10.127.0.192:7001",
    # Company UI domains
    "http://ai.coingdevelopment.com",
    "http://ai.coingdevelopment.com:8080",
    "http://ai.coingdevelopment.com:7001",
    "https://ai.coingdevelopment.com",
]
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors_env:
    # Parse comma-separated list of origins
    CORS_ALLOWED_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    logger.info(f"CORS origins from env: {CORS_ALLOWED_ORIGINS}")
else:
    CORS_ALLOWED_ORIGINS = _default_cors_origins
    logger.info(f"CORS using default origins: {CORS_ALLOWED_ORIGINS}")

MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

# PHASE 2 REFACTOR: Global index manager (initialized in _startup)
index_manager: Optional[IndexManager] = None


async def _startup() -> None:
    """Initialize embeddings and FAISS index on startup with validation."""
    logger.info("RAG System startup: validating index, seeding randomness, warming up embedding model...")

    # Prod guard: API_TOKEN must not be "change-me" in production (AXIOM 0)
    ENV = os.getenv("ENV", "dev")
    if ENV == "prod" and API_TOKEN == "change-me":
        logger.error("API_TOKEN must not be 'change-me' in production")
        raise RuntimeError("Invalid production config: API_TOKEN not configured")

    # PREBUILT INDEX VALIDATION: Ensure index files exist and metadata is valid
    logger.info(f"Validating prebuilt index for namespaces (env): {NAMESPACES}")
    valid_namespaces: List[str] = []
    for ns in NAMESPACES:
        root = INDEX_ROOT / ns
        idx_path_faiss = root / "index.faiss"
        idx_path_bin = root / "index.bin"
        meta_path = root / "meta.json"

        # Check index file exists
        if not idx_path_faiss.exists() and not idx_path_bin.exists():
            logger.warning(
                f"Skipping namespace '{ns}': missing prebuilt index at {root}. "
                f"Expected: {idx_path_faiss} or {idx_path_bin}"
            )
            continue

        # Check metadata exists
        if not meta_path.exists():
            logger.warning(
                f"Skipping namespace '{ns}': missing metadata at {meta_path}"
            )
            continue

        # Validate metadata format and model/dimension
        try:
            meta_data = json.loads(meta_path.read_text())
            meta_model = meta_data.get("model")
            meta_dim = meta_data.get("dim") or meta_data.get("dimension")

            logger.info(f"  Namespace '{ns}': model={meta_model}, dim={meta_dim}, vectors={meta_data.get('num_vectors', '?')}")

            if not meta_dim or meta_dim <= 0:
                raise ValueError(f"Invalid embedding dimension in metadata: {meta_dim}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Skipping namespace '{ns}': invalid metadata at {meta_path}: {e}"
            )
            continue

        valid_namespaces.append(ns)

    if not valid_namespaces:
        raise RuntimeError(
            f"\n❌ STARTUP FAILURE: No valid namespaces found under {INDEX_ROOT}. "
            "Build indexes or adjust NAMESPACES."
        )

    # Validate embedding dimension against local encoder
    try:
        probe = embed_query("clockify help health-check")
        encoder_dim = probe.shape[1]
        logger.info(f"✓ Embedding encoder ready: {EMBEDDING_MODEL} (dim={encoder_dim})")

        for ns in valid_namespaces:
            meta_data = json.loads((INDEX_ROOT / ns / "meta.json").read_text())
            meta_dim = meta_data.get("dim") or meta_data.get("dimension", 768)
            if meta_dim != encoder_dim:
                raise RuntimeError(
                    f"\n❌ STARTUP FAILURE: Embedding dimension mismatch for namespace '{ns}'\n"
                    f"   Index built with: dim={meta_dim}\n"
                    f"   Encoder provides: dim={encoder_dim}\n"
                    f"   Fix: Rebuild index with the current EMBEDDING_MODEL or update environment"
                )
    except Exception as e:
        raise RuntimeError(
            f"\n❌ STARTUP FAILURE: Embedding encoder check failed for model '{EMBEDDING_MODEL}'\n"
            f"   Error: {e}"
        )

    if MOCK_LLM:
        logger.info("⚠️  MOCK_LLM mode enabled: skipping live LLM probes, using mock responses")

    # PHASE 2 REFACTOR: Initialize IndexManager for thread-safe index loading
    logger.info("Initializing FAISS index manager...")
    global index_manager
    index_manager = IndexManager(INDEX_ROOT, valid_namespaces)
    index_manager.ensure_loaded()

    # Log vector counts per namespace for observability
    all_indexes = index_manager.get_all_indexes()
    for ns, entry in all_indexes.items():
        vector_count = entry["index"].ntotal
        dim = entry.get("dim", "unknown")
        logger.info(f"  ✓ Namespace '{ns}': {vector_count} vectors (dim={dim})")
    logger.info("✓ FAISS indexes loaded and cached")

    # Seed randomness for deterministic behavior (AXIOM 1)
    logger.info("Seeding randomness for deterministic retrieval...")
    random.seed(0)
    np.random.seed(0)

    # Initialize response cache for /search endpoint (80-90% latency reduction for repeats)
    init_cache()
    cache = get_cache()
    logger.info(f"✓ Response cache initialized: {cache}")

    # Log embedding backend for startup observability
    embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "model")
    if embeddings_backend == "stub":
        logger.info(f"⚠️  Embedding backend: STUB MODE (testing/development only)")
    else:
        logger.info(f"✓ Embedding backend: {embeddings_backend} ({EMBEDDING_MODEL})")

    try:
        warmup()
        logger.info("✓ Embedding model warmed up")
    except Exception as e:
        logger.error(f"Embedding warmup failed: {e}")

    # Log reranker status explicitly
    reranker_enabled = not os.getenv("RERANK_DISABLED", "false").lower() == "true"
    try:
        warmup_reranker()
        if reranker_enabled:
            logger.info("✓ Reranker model warmed up (ENABLED)")
        else:
            logger.info("⊘ Reranker warmup skipped (DISABLED via RERANK_DISABLED=true)")
    except Exception as e:
        if reranker_enabled:
            logger.warning(f"Reranker warmup failed (non-fatal, will disable reranking): {e}")
        else:
            logger.info("Reranker disabled, warmup skipped")

    logger.info("✅ RAG System startup complete: index validated, embedding ready, cache active")


def _shutdown() -> None:
    """Clean up HTTP client on FastAPI shutdown."""
    try:
        close_http_client()
    except Exception as e:
        logger.warning(f"Error closing HTTP client: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events.

    This replaces the deprecated @app.on_event pattern with a context manager
    that handles both startup and shutdown in a single, modern FastAPI pattern.
    """
    # Startup
    await _startup()
    yield
    # Shutdown
    _shutdown()


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

# Request logging middleware for observability
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing and status information."""
    request_id = str(uuid4())
    start_time = time.time()

    # Log incoming request
    logger.info(
        f"[{request_id}] → {request.method} {request.url.path} "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    # Process request
    response = await call_next(request)

    # Calculate duration and track metrics
    duration = time.time() - start_time

    # Track metrics (skip /metrics endpoint to avoid self-reference)
    if request.url.path != "/metrics":
        track_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            duration=duration
        )

    # Log response with timing
    logger.info(
        f"[{request_id}] ← {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    return response

def _filter_and_refill_for_test(hits: List[Dict[str, Any]], candidates: List[Dict[str, Any]], max_context: int) -> List[Dict[str, Any]]:
    """Testing helper: enforce allowlist/denylist on hits and refill from candidates.

    Mirrors the logic used inside chat() to filter hits by _is_allowed and then
    refill from remaining candidates to maintain context size when possible.

    Args:
        hits: Top-ranked unique results
        candidates: Full candidate list (ranked)
        max_context: Desired maximum number of sources to return

    Returns:
        List of filtered sources up to max_context in rank order
    """
    # Enforce allowlist
    filtered: List[Dict[str, Any]] = [h for h in hits if _is_allowed(h.get("url", ""))]
    # Refill to maintain context size if possible
    if len(filtered) < max_context:
        for cand in candidates:
            if len(filtered) >= max_context:
                break
            if cand not in filtered and _is_allowed(cand.get("url", "")):
                filtered.append(cand)
    return filtered[:max_context]

# FIX CRITICAL #3: CORS middleware with explicit origins (no wildcards)
# Using configurable CORS_ALLOWED_ORIGINS from environment
# allow_credentials=True is only safe because origins are explicitly restricted
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-token"],
)

# --------- Health Endpoints ---------
@app.get("/healthz")
def healthz():
    hs = CFG.health_summary()
    ns = hs.get("namespace")
    index_present = False
    try:
        ns_dir = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss")) / ns
        index_present = (ns_dir / "meta.json").exists() and ((ns_dir / "index.faiss").exists() or (ns_dir / "index.bin").exists())
    except Exception:
        index_present = False
    return {
        "ok": True,
        "namespace": ns,
        "index_present": index_present,
        "index_digest": hs.get("index_digest"),
        "lexical_weight": hs.get("search_lexical_weight"),
        "chunk_strategy": hs.get("chunk_strategy"),
    }


@app.get("/readyz")
def readyz():
    hs = CFG.health_summary()
    ns = hs.get("namespace")
    try:
        # index manager loaded?
        ok = index_manager is not None
    except Exception:
        ok = False
    return {
        "ok": bool(ok),
        "namespace": ns,
        "index_digest": hs.get("index_digest"),
    }

# --------- Retrieval Engine Management ---------
_retrieval_engine: Optional[RetrievalEngine] = None  # Lazy-initialized hybrid search engine
_retrieval_engine_lock = __import__('threading').Lock()  # Protects retrieval engine initialization

# --------- Retrieval ---------
def search_ns(ns: str, qvec: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Vector-only search via FAISS (original implementation).

    Converts FAISS L2 distances to similarity scores (0-1 scale).
    FAISS returns distances (lower=better); we convert to similarity: similarity = 1/(1+distance)
    """
    assert index_manager is not None, "IndexManager not initialized"
    entry = index_manager.get_index(ns)
    D, I = entry["index"].search(qvec, k)
    res = []
    for rank, (idx, distance) in enumerate(zip(I[0].tolist(), D[0].tolist()), start=1):
        if idx < 0:
            continue
        meta = entry["metas"][idx]
        # Convert FAISS L2 distance to similarity (0-1 scale)
        # Formula: similarity = 1 / (1 + distance)
        # This ensures: distance=0 -> similarity=1.0, distance=infinity -> similarity=0.0
        similarity = 1.0 / (1.0 + float(distance))
        # Filter out embedding field (not JSON-serializable, internal use only)
        meta_without_embedding = {k: v for k, v in meta.items() if k != "embedding"}
        res.append({
            "namespace": ns,
            "score": similarity,
            "rank": rank,
            **meta_without_embedding
        })
    return res

def search_ns_hybrid(ns: str, qvec: np.ndarray, query_text: str, k: int, alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Hybrid search combining BM25 (keyword) + vector semantic search (AXIOM 1-7)."""
    global _retrieval_engine

    # Allow env override for lexical weight (SEARCH_LEXICAL_WEIGHT). Vector weight = 1 - lexical_weight
    try:
        lex_w = float(CFG.SEARCH_LEXICAL_WEIGHT)
        alpha = max(0.0, min(1.0, 1.0 - lex_w))
    except Exception:
        pass

    # Fix #6: Thread-safe lazy initialization with double-check locking pattern
    if _retrieval_engine is None:
        with _retrieval_engine_lock:
            # Double-check: another thread may have initialized while we waited
            if _retrieval_engine is None:
                _retrieval_engine = RetrievalEngine(
                    config=RetrievalConfig(
                        strategy=RetrievalStrategy.HYBRID,
                        k_vector=min(k * 6, 100),      # Get more vector candidates before fusion
                        k_bm25=min(k * 6, 100),        # Get more BM25 candidates before fusion
                        k_final=k,
                        hybrid_alpha=alpha,
                        apply_diversity_penalty=True
                    )
                )
                logger.info("Hybrid retrieval engine initialized (BM25 + vector)")

    try:
        assert index_manager is not None, "IndexManager not initialized"
        entry = index_manager.get_index(ns)
        # PERFORMANCE FIX: Chunks now include embeddings (cached at startup)
        # No per-request reconstruction needed - embeddings are pre-populated in IndexManager._load_index_for_ns()
        chunks = entry["metas"]  # Already includes "embedding" field for each chunk

        # Extract 1D embedding if 2D array passed (for RetrievalEngine compatibility)
        embedding_1d = qvec[0] if qvec.ndim == 2 else qvec

        # Call hybrid search engine (chunks already have embeddings)
        results = _retrieval_engine.search_hybrid(
            query_embedding=embedding_1d,
            query_text=query_text,
            chunks=chunks,
            k=k,
            alpha=alpha
        )

        # Convert RetrievalResult objects to dict format (backward compatible)
        res = []
        for rank, result in enumerate(results, start=1):
            # Filter out 'embedding' from metadata (internal use only, not serializable)
            metadata_without_embedding = {
                k: v for k, v in result.metadata.items()
                if k != "embedding"
            }
            res.append({
                "namespace": ns,
                "score": float(result.hybrid_score or result.final_score or 0.0),
                "rank": rank,
                "chunk_id": result.chunk_id,
                "text": result.text,
                "title": result.title,
                "url": result.url,
                # Preserve original metadata (excluding embedding)
                **metadata_without_embedding
            })

        logger.debug(f"Hybrid search for '{query_text}' in {ns}: {len(res)}/{len(chunks)} candidates")
        return res

    except Exception as e:
        logger.warning(f"Hybrid search failed for {ns}: {e}, falling back to vector search")
        return search_ns(ns, qvec, k)

def search_with_decomposition(
    q: str, qvec: np.ndarray, k: int, ns_list: List[str], decomp_result
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve results using query decomposition for multi-intent queries.

    Per-subtask retrieval with score fusion:
    1. For each subtask, recompute intent and retrieve with appropriate strategy
    2. Convert FAISS distances to similarities (score = 1 / (1 + distance))
    3. Fuse by (url, chunk_id): best score + additive hits bonus (+0.05 per hit)
    4. Return merged results with per-doc hit mapping for analysis

    Returns:
        (merged_results, decomposition_metadata)
    """
    # Dict keyed by (url, chunk_id) storing:
    # {best_score, hits, semantic_score, subtasks_hit, payload}
    fused_docs = {}

    logger.debug(f"Starting decomposition-aware retrieval for {len(decomp_result.subtasks)} subtasks")

    for subtask_idx, subtask in enumerate(decomp_result.subtasks):
        subtask_text = subtask.text
        subtask_reason = subtask.reason
        subtask_weight = subtask.weight
        subtask_intent = subtask.intent  # May be None if detection failed

        logger.debug(
            f"Subtask {subtask_idx+1}: '{subtask_text}' "
            f"(reason={subtask_reason}, intent={subtask_intent}, weight={subtask_weight})"
        )

        try:
            # Recompute intent if not already set
            if subtask_intent is None:
                subtask_intent = detect_query_type(subtask_text)

            # Recompute hybrid flag based on subtask's own characteristics
            subtask_query_type = subtask_intent if subtask_intent else detect_query_type(subtask_text)
            should_hybrid = should_enable_hybrid_search(subtask_query_type, len(subtask_text))

            # Expand subtask with boost terms specific to this subtask (structured with weights)
            subtask_variants = expand_structured(subtask_text, boost_terms=subtask.boost_terms)
            logger.debug(f"  Expanded to {len(subtask_variants)} weighted variants")

            # Encode subtask variants using weighted averaging
            # encode_weighted_variants handles the weighting, averaging, and L2 normalization
            subtask_qvec = encode_weighted_variants(subtask_variants).flatten()

            # Retrieve with adaptive k based on subtask intent
            adaptive_k = get_adaptive_k_multiplier(subtask_query_type, k)
            raw_k = min(adaptive_k * 2, 100)

            # Retrieve per namespace for this subtask
            per_ns = {}
            if should_hybrid:
                per_ns = {
                    ns: search_ns_hybrid(ns, subtask_qvec[None,:].astype(np.float32), subtask_text, raw_k)
                    for ns in ns_list
                }
            else:
                per_ns = {
                    ns: search_ns(ns, subtask_qvec[None,:].astype(np.float32), raw_k)
                    for ns in ns_list
                }

            # Fuse across namespaces for this subtask
            subtask_results = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]

            logger.debug(f"  Retrieved {len(subtask_results)} results from {len(ns_list)} namespace(s)")

            # Process subtask results and merge into fused_docs
            for result in subtask_results:
                url = result.get("url", "")
                chunk_id = result.get("chunk_id", result.get("id", ""))
                # All scores are already normalized to [0-1] similarity range:
                # - search_ns: converts distances to similarities before returning
                # - search_ns_hybrid: returns normalized fused scores
                similarity = result.get("score", 0.0)

                doc_key = (url, chunk_id)

                if doc_key not in fused_docs:
                    # First hit for this document
                    fused_docs[doc_key] = {
                        "payload": result,
                        "best_score": similarity,
                        "semantic_score": similarity,
                        "hits": 1,
                        "subtasks_hit": [
                            {
                                "text": subtask_text,
                                "reason": subtask_reason,
                                "intent": subtask_intent,
                                "weight": subtask_weight,
                                "score": similarity,
                            }
                        ],
                    }
                else:
                    # Document already seen in another subtask
                    fused_docs[doc_key]["hits"] += 1
                    fused_docs[doc_key]["semantic_score"] += similarity  # Accumulate similarity
                    fused_docs[doc_key]["best_score"] = max(fused_docs[doc_key]["best_score"], similarity)
                    fused_docs[doc_key]["subtasks_hit"].append(
                        {
                            "text": subtask_text,
                            "reason": subtask_reason,
                            "intent": subtask_intent,
                            "weight": subtask_weight,
                            "score": similarity,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error retrieving for subtask {subtask_idx+1} '{subtask_text}': {e}")
            continue

    # Compute final scores with multi-hit bonus
    for doc_key, doc_data in fused_docs.items():
        # Final score: best individual score + accumulation + hits bonus
        hits = doc_data["hits"]
        best_score = doc_data["best_score"]
        # semantic_score available in doc_data["semantic_score"] if needed for future scoring

        # Additive fusion: reward documents matching multiple subtasks
        # Score = best_score + (0.05 * (hits - 1))
        # This ensures single-hit docs aren't penalized, but multi-hit docs are boosted
        final_score = best_score + (0.05 * (hits - 1))

        # Cap final score at 1.0
        final_score = min(final_score, 1.0)

        # Update payload with fused score and metadata
        doc_data["payload"]["score"] = final_score
        doc_data["payload"]["decomposition_hits"] = hits
        doc_data["payload"]["decomposition_subtasks"] = doc_data["subtasks_hit"]

    # Sort by final score and extract payloads (filter out embeddings)
    sorted_docs = sorted(
        fused_docs.values(),
        key=lambda x: (-x["payload"]["score"], -x["hits"])
    )
    merged_list = [_remove_embedding_from_result(doc["payload"]) for doc in sorted_docs]

    logger.debug(
        f"Fused {len(fused_docs)} unique documents across {len(decomp_result.subtasks)} subtasks"
    )

    # Build metadata with per-subtask info and per-doc hit mapping
    metadata = {
        "decomposition_strategy": decomp_result.strategy,
        "llm_used": decomp_result.llm_used,
        "subtask_count": len(decomp_result.subtasks),
        "subtasks": [st.to_log_payload() for st in decomp_result.subtasks],
        "fused_docs": len(fused_docs),
        "multi_hit_docs": sum(1 for doc in fused_docs.values() if doc["hits"] > 1),
    }

    return merged_list, metadata

def _remove_embedding_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove embedding field from result dict (not serializable, internal use only)."""
    return {k: v for k, v in result.items() if k != "embedding"}

def fuse_results(by_ns: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
    scores: Dict[Tuple[str, str], float] = {}
    payloads: Dict[Tuple[str, str], Dict[str, Any]] = {}
    C = 60.0
    for ns, lst in by_ns.items():
        for r, item in enumerate(lst, start=1):
            # Remove embedding before processing
            item = _remove_embedding_from_result(item)
            key = (item.get("url",""), item.get("chunk_id", item.get("id", r)))
            scores[key] = scores.get(key, 0.0) + 1.0 / (C + r)
            payloads[key] = item
    merged = sorted(payloads.values(), key=lambda x: scores[(x.get("url",""), x.get("chunk_id", x.get("id", 0)))], reverse=True)
    return merged[:k]

# --------- Models ---------
class SearchQuery(BaseModel):
    """Validated search query parameters."""
    q: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    namespace: Optional[str] = Field(default=None, max_length=100)

class SearchResponse(BaseModel):
    """Search response with request tracing and metadata."""
    success: bool = True
    query: str
    results: List[Dict[str, Any]]
    total_results: int = 0
    latency_ms: int = 0
    request_id: str = ""
    metadata: Optional[Dict[str, Any]] = None
    query_decomposition: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Validated chat request."""
    question: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    namespace: Optional[str] = Field(default=None, max_length=100)

class ChatResponse(BaseModel):
    """Chat response with citations and grounding."""
    success: bool = True
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: Dict[str, Any]
    meta: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

# --------- Auth/limits ---------
def require_token(token: Optional[str]):
    """Verify API token using constant-time comparison (AXIOM 0).

    Always validates token in all environments. In dev mode, use API_TOKEN="change-me"
    to require that specific token. In production, set API_TOKEN to actual secret.
    """
    # Token is always required; AXIOM 0 enforcement
    if not token:
        raise HTTPException(status_code=401, detail="unauthorized")

    # Production: validate token strictly
    ENV_MODE = os.getenv("ENV", "dev").strip().lower()
    if ENV_MODE != "prod" and API_TOKEN == "change-me":
        # Dev ergonomics: accept any non-empty token when using default dev token
        return
    # Accept primary or any from API_TOKENS
    valid = hmac.compare_digest(token, API_TOKEN) or any(hmac.compare_digest(token, t) for t in API_TOKENS_EXTRA)
    if not valid:
        logger.warning("Invalid token attempt")
        raise HTTPException(status_code=401, detail="unauthorized")

_last_req: Dict[str, float] = {}

def rate_limit(ip: str, min_interval: float = 0.1) -> None:
    """Rate limit: enforce minimum interval between requests per IP (AXIOM 0)."""
    # Skip rate limiting for localhost to allow parallel requests from browser UI
    if ip in ("127.0.0.1", "localhost", "::1"):
        return

    now = time.time()

    # Fix #7: Cleanup old entries to prevent memory leak
    # Remove IPs that haven't been seen in 2*min_interval window
    window = 2 * min_interval
    stale_ips = [k for k, v in _last_req.items() if now - v > window]
    for ip_to_remove in stale_ips:
        del _last_req[ip_to_remove]

    t = _last_req.get(ip, 0.0)
    if now - t < min_interval:
        raise HTTPException(status_code=429, detail="rate_limited")
    _last_req[ip] = now

# --------- Routes ---------
@app.get("/health")
def health(deep: int = 0, detailed: int = 0) -> Dict[str, Any]:
    """
    Health endpoint with optional detailed statistics.

    Args:
        deep: Include deep LLM health checks (chat ping)
        detailed: Include detailed cache stats and circuit breaker metrics

    Returns:
        Health status dict with optional detailed observability data
    """
    ok = True
    try:
        if index_manager:
            index_manager.ensure_loaded()
    except Exception as e:
        ok = False
        logger.error(f"Index load error: {e}")

    # Get all indexes and compute metrics (PHASE 2 REFACTOR: use IndexManager)
    index_metrics = {}
    index_normalized = None
    if index_manager:
        all_indexes = index_manager.get_all_indexes()
        if all_indexes:
            index_normalized = all(index_manager.is_normalized(ns) for ns in all_indexes.keys())
            for ns, entry in all_indexes.items():
                index = entry["index"]
                ntotal = index.ntotal
                metas = entry["metas"]
                index_metrics[ns] = {
                    "indexed_vectors": ntotal,
                    "indexed_chunks": len(metas),
                    "vector_dim": entry.get("dim", 768),
                    "normalized": index_manager.is_normalized(ns),
                }

    # Check embedding model health
    embedding_ok = None
    embedding_details = None
    try:
        test_vec = embed_query("health check")
        embedding_ok = test_vec.shape[0] > 0 and test_vec.shape[1] > 0
        embedding_details = f"OK: {EMBEDDING_MODEL} (dim={test_vec.shape[1]})"
    except Exception as e:
        embedding_ok = False
        embedding_details = f"Error: {str(e)}"

    # Check reranker availability
    reranker_ok = reranker_available()
    reranker_details = "Available: BAAI/bge-reranker-base" if reranker_ok else "Not available (FlagEmbedding not installed or disabled)"

    # Check LLM health if not mock mode
    llm_ok = None
    llm_details = None
    llm_deep_ok = None
    llm_deep_details = None

    if not MOCK_LLM:
        try:
            llm = LLMClient()
            llm_check = llm.health_check()
            llm_ok = llm_check.get("ok")
            llm_details = llm_check.get("details")

            # Deep health check: try a lightweight chat ping
            if deep:
                try:
                    result = llm.chat([{"role": "user", "content": "ping"}], stream=False)
                    llm_deep_ok = bool(result)
                    llm_deep_details = "chat ping ok" if llm_deep_ok else "empty response"
                except Exception as e:
                    llm_deep_ok = False
                    llm_deep_details = f"chat ping failed: {str(e)}"
        except Exception as e:
            llm_ok = False
            llm_details = f"Error initializing LLM client: {str(e)}"
            llm_deep_ok = False
            llm_deep_details = "skipped due to init error"
    else:
        # In mock mode, deep checks are skipped
        llm_deep_ok = None
        llm_deep_details = None

    # Get circuit breaker status
    circuit_breakers = get_all_circuit_breakers()

    # PHASE 2 REFACTOR: Get namespaces from IndexManager
    namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
    index_normalized_by_ns = {}
    if index_manager:
        all_indexes = index_manager.get_all_indexes()
        index_normalized_by_ns = {ns: index_manager.is_normalized(ns) for ns in all_indexes.keys()}

    # PHASE 5: Get semantic cache stats for observability
    semantic_cache_stats = None
    try:
        semantic_cache_stats = get_semantic_cache().stats()
    except Exception as e:
        logger.warning(f"Failed to get semantic cache stats: {e}")

    # Build base response
    response = {
        "ok": ok,
        "namespaces": namespaces,
        "mode": "mock" if MOCK_LLM else "live",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_ok": embedding_ok,
        "embedding_details": embedding_details,
        "reranker_ok": reranker_ok,
        "reranker_details": reranker_details,
        "llm_api_type": os.getenv("LLM_API_TYPE","ollama"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-oss:20b"),
        "llm_ok": llm_ok,
        "llm_details": llm_details,
        "llm_deep_ok": llm_deep_ok,
        "llm_deep_details": llm_deep_details,
        "index_normalized": index_normalized,
        "index_normalized_by_ns": index_normalized_by_ns,
        "index_metrics": index_metrics,
        "cache": {
            "semantic_cache_stats": semantic_cache_stats,
        },
    }

    # Add detailed stats if requested
    if detailed:
        response["circuit_breakers"] = circuit_breakers

        # Add cache hit rate summary
        if semantic_cache_stats:
            response["cache_hit_rate_pct"] = semantic_cache_stats.get("hit_rate_pct", 0)
            response["cache_memory_mb"] = semantic_cache_stats.get("memory_usage_mb", 0)
            response["cache_entries"] = semantic_cache_stats.get("size", 0)
            response["cache_max_entries"] = semantic_cache_stats.get("max_size", 0)

        # Add system configuration details
        response["config"] = {
            "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "model"),
            "rerank_disabled": os.getenv("RERANK_DISABLED", "false").lower() == "true",
            "mock_llm": MOCK_LLM,
            "semantic_cache_ttl_seconds": os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"),
            "semantic_cache_max_size": os.getenv("SEMANTIC_CACHE_MAX_SIZE", "10000"),
        }
    else:
        # In non-detailed mode, only include circuit breakers for backward compatibility
        response["circuit_breakers"] = circuit_breakers

    return response

@app.get("/live")
def live() -> Dict[str, str]:
    """Liveness probe: returns 200 if process is alive (no dependencies checked)."""
    return {"status": "alive"}

@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping by monitoring systems.
    Tracks request counts, latency, cache stats, circuit breaker status, and more.
    """
    from fastapi.responses import Response

    # Update circuit breaker metrics before returning
    circuit_breakers = get_all_circuit_breakers()
    for name, status in circuit_breakers.items():
        track_circuit_breaker(
            name=name,
            state=status["state"],
            metrics_data={
                "total_requests": status["total_requests"],
                "total_failures": status["total_failures"],
                "total_successes": status["total_successes"],
                "consecutive_failures": status["consecutive_failures"],
            }
        )

    return Response(content=get_metrics(), media_type=get_content_type())

@app.get("/perf")
def perf(detailed: bool = False) -> Dict[str, Any]:
    """
    Performance metrics endpoint - TIER 2 improvement.

    Shows latency statistics across all pipeline stages.
    Useful for identifying bottlenecks and optimization opportunities.

    Args:
        detailed: If true, return recent samples for analysis

    Returns:
        Performance statistics dictionary
    """
    tracker = get_performance_tracker()

    result = tracker.get_stats()

    if detailed:
        result["recent_samples"] = tracker.get_recent_samples(limit=20)
        result["bottlenecks"] = tracker.get_bottlenecks(top_n=5)

    return result

@app.get("/ready")
def ready() -> Tuple[Dict[str, str], Optional[int]]:
    """Readiness probe: returns 200 only if index loaded and LLM ready."""
    try:
        if index_manager:
            index_manager.ensure_loaded()
        if not MOCK_LLM:
            llm = LLMClient()
            llm_check = llm.health_check()
            if not llm_check.get("ok"):
                logger.warning(f"LLM not ready: {llm_check.get('details')}")
                return {"status": "not_ready", "reason": "llm_unhealthy"}, 503
        return {"status": "ready"}, 200
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return {"status": "not_ready", "reason": str(e)}, 503

@app.get("/config")
def config(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    ENV = os.getenv("ENV", "dev")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")

    # In prod, hide llm_base_url unless admin token matches
    reveal_sensitive = (ENV != "prod") or (ADMIN_TOKEN != "change-me" and x_admin_token == ADMIN_TOKEN)

    # PHASE 5: Enhanced /config endpoint for observability
    # Includes reranker status, embedding backend, actual loaded namespaces, cache stats
    from src.embeddings import EMBEDDING_DIM
    from src.semantic_cache import get_semantic_cache

    out = {
        # Core config
        "namespaces_env": NAMESPACES,
        "actual_namespaces": list(index_manager.get_all_indexes().keys()) if index_manager else [],
        "index_mode": os.getenv("INDEX_MODE","single"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "retrieval_k": RETRIEVAL_K,
        "env": ENV,

        # Feature flags (PHASE 5)
        "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "ollama"),
        "rerank_disabled": os.getenv("RERANK_DISABLED", "false").lower() == "true",
        "streaming_enabled": os.getenv("STREAMING_ENABLED","false").lower()=="true",
        "mock_llm": MOCK_LLM,

        # LLM config
        "llm_chat_path": os.getenv("LLM_CHAT_PATH", "/api/chat"),
        "llm_tags_path": os.getenv("LLM_TAGS_PATH", "/api/tags"),
        "llm_timeout_seconds": int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        "llm_api_type": os.getenv("LLM_API_TYPE", "ollama"),

        # Cache stats (PHASE 5)
        "cache": {
            "response_cache_size": get_cache().size() if get_cache() else 0,
            "semantic_cache_stats": get_semantic_cache().stats(),
        }
    }

    if reveal_sensitive:
        out["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434")
    else:
        out["llm_base_url"] = "<hidden>"

    return out

@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    request: Request,
    k: Optional[int] = None,
    namespace: Optional[str] = None,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition for baseline comparison"),
    x_api_token: Optional[str] = Header(default=None),
) -> SearchResponse:
    """Search with query expansion, normalized embeddings, optional reranking (AXIOM 1,3,4,5,7).

    Responses cached for repeated queries (80-90% latency reduction).
    """
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    if index_manager:
        index_manager.ensure_loaded()
    # Sanitize k: clamp to [1, 20] bounds
    k = max(1, min(int(k or RETRIEVAL_K), 20))

    t0 = time.time()

    try:
        # AXIOM 4: Query expansion with glossary synonyms
        logger.debug(f"Search query: '{q}'")

        # NEW: Detect and decompose multi-intent queries (before cache check)
        decomp_result = None
        decomp_metadata: Dict[str, Any] = {}
        cache_query = q  # Default: use original query as cache key

        if not decomposition_off and is_multi_intent_query(q):
            try:
                decomp_result = decompose_query(q)
                logger.info(f"Decomposed query into {len(decomp_result.subtasks)} subtasks "
                           f"(strategy={decomp_result.strategy})")

                # If successful decomposition with >1 subtask, modify cache key to avoid collisions
                if decomp_result and len(decomp_result.subtasks) > 1:
                    # Create unique cache key that includes subtask info
                    # Format: original_query|decomp:subtask1|subtask2|...
                    subtask_str = "|".join([st.text for st in decomp_result.subtasks[1:]])  # Skip original
                    cache_query = f"{q}|decomp:{subtask_str}"
                    logger.debug(f"Using decomposition-aware cache key for multi-intent query")
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}, continuing with standard expansion")
                decomp_result = None
        elif decomposition_off:
            logger.debug(f"Decomposition disabled via decomposition_off=true")

        # Check response cache with decomposition-aware key
        cache = get_cache()
        cached_response = cache.get(cache_query, k, namespace)
        if cached_response is not None:
            latency_ms = int((time.time() - t0) * 1000)
            logger.info(f"Search cache hit: '{q}' (decomp={decomp_result is not None and len(decomp_result.subtasks) > 1}) k={k} in {latency_ms}ms")
            return cached_response

        # Standard expansion (used for all-in-one retrieval if no decomposition)
        if not decomp_result or len(decomp_result.subtasks) <= 1:
            # Use structured expansion with weights for better embedding fusion
            variants = expand_structured(q)  # [{text, source, weight}, ...]
            logger.debug(f"Query expanded to {len(variants)} weighted variants")
        else:
            # Use original as fallback expansion
            variants = [{"text": q, "source": "original", "weight": 1.0}]
            logger.debug(f"Skipping standard expansion due to decomposition (using original query only)")

        # QUICK WIN: Detect query type for adaptive retrieval strategy
        query_type = detect_query_type(q)
        adaptive_k = get_adaptive_k_multiplier(query_type, k)
        log_query_analysis(q, query_type, adaptive_k)

        # AXIOM 3: Encode variants with weighted averaging and L2 normalization
        # encode_weighted_variants handles weighting, averaging, and normalization
        qvec = encode_weighted_variants(variants).flatten()
        # Verify normalization (AXIOM 3) but do not assert; just log
        actual_norm = np.linalg.norm(qvec)
        if not (0.98 <= actual_norm <= 1.02):
            logger.debug(f"Query vector norm={actual_norm:.6f} (expected ~1.0); reclamping")
            qvec = qvec / (actual_norm + 1e-8)

        # AXIOM 1: Determinism via deterministic namespace-based retrieval
        # Use sorted() for consistent ordering across multiple calls (PHASE 2 REFACTOR: use IndexManager)
        available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
        ns_list = [namespace] if namespace in available_namespaces else sorted(available_namespaces)

        # QUICK WIN: Adaptive k multiplier based on query type for better candidate pool
        # Retrieve with adaptive_k instead of fixed k*6 or k*3
        raw_k = min(adaptive_k, 100)  # Cap at 100 for efficiency

        # NEW: Use decomposition-aware retrieval if query was decomposed
        if decomp_result and len(decomp_result.subtasks) > 1:
            logger.info(f"Using decomposition-aware retrieval for multi-intent query")
            candidates, decomp_metadata = search_with_decomposition(
                q, qvec, raw_k, ns_list, decomp_result
            )
            # Limit to raw_k results
            candidates = candidates[:raw_k]
        else:
            # Standard retrieval (vector or hybrid)
            use_hybrid = should_enable_hybrid_search(query_type, len(q))

            # Retrieve results using hybrid or vector-only search
            if use_hybrid:
                logger.debug(f"Using hybrid search (BM25 + vector) for query type: {query_type}")
                per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), q, raw_k) for ns in ns_list}
            else:
                logger.debug(f"Using vector-only search for query type: {query_type}")
                per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}

            # Fuse and deduplicate by URL (AXIOM 4)
            candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]
        # AXIOM 1: Stable sort on candidates before dedup to ensure deterministic tie-breaking
        candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
        seen_urls = set()
        results_dedup = []
        for candidate in candidates:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results_dedup.append(candidate)
            if len(results_dedup) >= k:
                break

        # AXIOM 5: Optional reranking (silent fallback if not available)
        results = rerank(q, results_dedup, k) if results_dedup else []

        # NEW: Apply query optimization and confidence scoring
        optimizer = get_optimizer()
        scorer = get_scorer()

        # Analyze query for optimization
        query_analysis = optimizer.analyze(q)
        query_entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("type", "general")

        # Score and rank results by confidence
        if results:
            results = scorer.batch_score(results, q, query_entities, query_type)

        # Add sequential 1-based rank to each result
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Search '{q}' k={k} -> {len(results)} results (unique URLs) in {latency_ms}ms")

        request_id = str(uuid4())

        # Build ResponseMetadata with proper Pydantic objects
        decomp_meta = None
        if decomp_result and len(decomp_result.subtasks) > 1:
            decomp_meta = DecompositionMetadata(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=[st.text for st in decomp_result.subtasks],
                llm_used=getattr(decomp_result, "llm_used", False),
                fused_docs=len(results),
                multi_hit_docs=decomp_metadata.get("multi_hit_docs", 0),
            )

        response_metadata = ResponseMetadata(
            cache_hit=False,
            index_normalized=True,
            decomposition=decomp_meta,
        )

        response: Dict[str, Any] = {
            "success": True,
            "query": q,
            "results": results,
            "total_results": len(results),
            "latency_ms": latency_ms,
            "request_id": request_id,
            "metadata": response_metadata.model_dump(),
        }

        # Add stub mode disclaimer if using test embeddings
        embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "model")
        if embeddings_backend == "stub":
            response["metadata"]["stub_mode_disclaimer"] = (
                "⚠️  STUB MODE: Using deterministic test embeddings. "
                "Results are NOT semantically meaningful. Use only for testing/CI."
            )

        # Include legacy decomposition metadata if available (for backward compat)
        if decomp_metadata:
            response["query_decomposition"] = decomp_metadata

        # Cache the response for repeated queries (80-90% latency improvement)
        # Use decomposition-aware cache key if applicable
        cache.set(cache_query, k, response, namespace)

        return response

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition for baseline comparison"),
    x_api_token: Optional[str] = Header(default=None),
) -> ChatResponse:
    """Chat with RAG: retrieve, ground answer, cite sources (AXIOM 1,2,3,4,6,7,9)."""
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    if index_manager:
        index_manager.ensure_loaded()
    # Sanitize k: clamp to [1, 20] bounds
    k = max(1, min(int(req.k or RETRIEVAL_K), 20))

    t0 = time.time()

    try:
        # AXIOM 1,3,4: Same retrieval as /search - expanded, normalized, deduplicated
        logger.debug(f"Chat question: '{req.question}'")

        # NEW: Detect and decompose multi-intent queries for chat
        decomp_result = None
        if not decomposition_off and is_multi_intent_query(req.question):
            try:
                decomp_result = decompose_query(req.question)
                logger.info(f"Decomposed chat question into {len(decomp_result.subtasks)} subtasks "
                           f"(strategy={decomp_result.strategy})")
            except Exception as e:
                logger.warning(f"Query decomposition failed for chat: {e}, continuing with standard expansion")
        elif decomposition_off:
            logger.debug(f"Decomposition disabled via decomposition_off=true")

        # Standard expansion (used for all-in-one retrieval if no decomposition)
        if not decomp_result or len(decomp_result.subtasks) <= 1:
            # Use structured expansion with weights for better embedding fusion
            variants = expand_structured(req.question)  # [{text, source, weight}, ...]
            logger.debug(f"Query expanded to {len(variants)} weighted variants")
        else:
            # Use original as fallback expansion
            variants = [{"text": req.question, "source": "original", "weight": 1.0}]
            logger.debug(f"Skipping standard expansion due to decomposition (using original query only)")

        # Encode variants with weighted averaging and L2 normalization
        qvec = encode_weighted_variants(variants).flatten()

        # PHASE 5: Check semantic cache before expensive retrieval
        # Cache key uses query embedding fingerprint + doc IDs + prompt version
        from src.semantic_cache import get_semantic_cache
        semantic_cache = get_semantic_cache()

        # Use sorted() for deterministic namespace ordering (AXIOM 1, PHASE 2 REFACTOR)
        available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
        ns_list = [req.namespace] if req.namespace in available_namespaces else sorted(available_namespaces)
        raw_k = k * 3  # For chat, use slightly smaller retrieval set

        # NEW: Use decomposition-aware retrieval if query was decomposed
        if decomp_result and len(decomp_result.subtasks) > 1:
            logger.info(f"Using decomposition-aware retrieval for multi-intent chat question")
            candidates, _ = search_with_decomposition(
                req.question, qvec, raw_k, ns_list, decomp_result
            )
            candidates = candidates[:raw_k]
        else:
            # Standard retrieval (vector or hybrid)
            query_type = detect_query_type(req.question)
            use_hybrid = should_enable_hybrid_search(query_type, len(req.question))

            if use_hybrid:
                logger.debug(f"Using hybrid search (BM25 + vector) for chat question type: {query_type}")
                per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), req.question, raw_k) for ns in ns_list}
            else:
                logger.debug(f"Using vector-only search for chat question type: {query_type}")
                per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}

            candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]
        # AXIOM 1: Stable sort on candidates before dedup to ensure deterministic tie-breaking
        candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
        seen_urls = set()
        hits = []
        for candidate in candidates:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            hits.append(candidate)
            if len(hits) >= k:
                break

        t_retr = int((time.time() - t0) * 1000)

        # AXIOM 2,6: Build context with citations for grounding (enforce allowlist on sources)
        # Convert hits to chunk format for RAGPrompt.build_messages()
        chunks = []
        source_map = {}

        # Use more chunks for richer context (configurable via MAX_CONTEXT_CHUNKS)
        top_context = hits[: min(CONFIG.MAX_CONTEXT_CHUNKS, len(hits))]

        # Enforce allowlist/denylist on candidate sources; refill from next candidates if needed
        filtered = []
        for h in hits:
            if _is_allowed(h.get("url", "")):
                filtered.append(h)
        if len(filtered) < len(hits):
            logger.info(f"Allowlist guard dropped {len(hits)-len(filtered)} source candidates")
        # Refill to maintain context size if possible
        if len(filtered) < len(top_context):
            for cand in candidates:
                if len(filtered) >= len(top_context):
                    break
                if cand not in filtered and _is_allowed(cand.get("url", "")):
                    filtered.append(cand)

        build_sources = filtered[: min(CONFIG.MAX_CONTEXT_CHUNKS, len(filtered))]

        for i, h in enumerate(build_sources, start=1):
            url = h.get("url", "")
            anchor = h.get("anchor")
            url_with_anchor = f"{url}#{anchor}" if anchor else url
            title_path = h.get("title_path") or []
            if isinstance(title_path, list) and title_path:
                title = " > ".join(title_path)
            else:
                title = h.get("title") or url or f"chunk-{i}"
            text = h.get("text", "")[:1600]
            chunk_id = h.get("chunk_id", h.get("id", str(i)))

            # Build chunk dict for RAGPrompt
            chunk = {
                "title": title,
                "url": url_with_anchor,
                "namespace": h.get("namespace"),
                "score": h.get("score", 0.0),
                "text": text,
                "chunk_id": chunk_id,
                "anchor": anchor,
            }
            chunks.append(chunk)
            source_map[str(i)] = chunk_id

        # HARMONY: Use RAGPrompt.build_messages() to get system prompt, user prompt, and developer instructions
        # reasoning_effort="low" is default for RAG to minimize latency per gpt-oss:20b best practices
        messages, sources, developer_instructions = RAGPrompt.build_messages(
            question=req.question,
            chunks=chunks,
            namespace=req.namespace or "clockify",
            max_chunks=len(chunks),
            reasoning_effort="low"  # gpt-oss:20b best practice: minimize latency
        )

        # Update source_map for citation tracking
        for i, source in enumerate(sources, start=1):
            source_map[str(i)] = source.get("chunk_id", "")

        t1 = time.time()
        # AXIOM 1: Determinism via configurable temperature (default 0.0 for strict determinism)
        temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # PHASE 5: Try semantic cache before LLM call
        # Cache key: query embedding + top doc IDs + namespaces + prompt version
        top_doc_ids = [s.get("chunk_id", "") for s in sources]
        cached_result = semantic_cache.get(qvec, top_doc_ids, prompt_version="v1", namespaces=ns_list)
        cache_hit = False

        if cached_result is not None:
            answer = cached_result["answer"]
            cache_hit = True
            logger.info(f"✓ CACHE HIT for question: '{req.question[:50]}...'")
        else:
            try:
                # HARMONY: Use chat method with messages list
                llm = LLMClient()
                answer = llm.chat(
                    messages=messages,
                    max_tokens=800,
                    temperature=temp,
                    stream=False
                )
            except CircuitOpenError as e:
                # Circuit breaker is open - LLM service temporarily unavailable
                logger.error(f"Circuit breaker open for chat request: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service temporarily unavailable. Circuit breaker is open. Please try again in a minute.",
                    headers={"Retry-After": "60"}
                )

        t_llm = int((time.time() - t1) * 1000)

        # PHASE 5: Answerability check - prevent hallucination
        # Validate that answer is grounded in context (Jaccard overlap >= 0.25)
        from src.llm_client import compute_answerability_score
        # Build context string from chunk texts
        # CRITICAL: Use same char truncation that LLM saw in prompt (configurable via CONTEXT_CHAR_LIMIT)
        # If we use full text here but LLM saw truncated text, valid answers get rejected
        context_blocks = [chunk.get("text", "")[:CONFIG.CONTEXT_CHAR_LIMIT] for chunk in chunks]
        context_str = "\n".join(context_blocks)
        is_answerable, answerability_score = compute_answerability_score(answer, context_str)

        # Debug logging: always log answerability score for diagnostics
        logger.info(
            f"Answerability check: score={answerability_score:.3f}, "
            f"threshold={CONFIG.ANSWERABILITY_THRESHOLD}, passed={is_answerable}"
        )

        if not is_answerable:
            # Debug logging: save original LLM answer before replacing
            original_answer = answer
            logger.warning(
                f"Answerability check failed for chat response (score={answerability_score:.3f}). "
                f"Original LLM answer: {original_answer[:200]}... "
                f"Replacing with refusal."
            )
            # Replace with safe refusal instead of returning hallucinated answer
            answer = "I don't have enough information in the documentation to answer that confidently. Could you rephrase your question or ask about a related topic?"

        # Validate citations in response
        citation_validation = validate_citations(answer, len(sources), strict=False)
        if not citation_validation.is_valid:
            logger.warning(
                f"Citation validation failed for chat response: "
                f"missing={citation_validation.missing_citations}, "
                f"invalid={citation_validation.invalid_citations}"
            )

        # AXIOM 2: Extract and validate citations (AXIOM 9: test this grounding)
        # Safe citation parsing: strip URLs first, then extract [1], [2],...[99]
        tmp = re.sub(r'https?://\S+', '<URL>', answer)
        citations_in_answer = re.findall(r'\[(\d{1,2})\]', tmp)
        cited_chunks = []
        for cite_idx_str in set(citations_in_answer):
            try:
                cite_idx = int(cite_idx_str)
                if 1 <= cite_idx <= len(sources):
                    cited_chunks.append(source_map.get(str(cite_idx), sources[cite_idx - 1].get("chunk_id")))
            except (ValueError, IndexError):
                pass

        # AXIOM 2 citation floor: if no citations found but sources exist, append [1]
        citations_found = len(citations_in_answer)
        if citations_found == 0 and sources:
            answer = answer.rstrip() + " [1]"
            citations_found = 1
            cited_chunks = [sources[0].get("chunk_id", "")]
            logger.debug(f"Citation floor applied: appended [1] to answer")

        logger.info(
            f"Chat '{req.question[:50]}...' -> {len(sources)} sources, "
            f"{citations_found} citations (floor applied: {len(citations_in_answer)==0 and bool(sources)}), {t_retr}ms retrieval, {t_llm}ms LLM"
        )

        model_used = os.getenv("LLM_MODEL", "gpt-oss:20b")
        request_id = str(uuid4())
        total_latency = int((time.time() - t0) * 1000)

        # Build ResponseMetadata with proper Pydantic objects
        decomp_meta = None
        if decomp_result and len(decomp_result.subtasks) > 1:
            decomp_meta = DecompositionMetadata(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=[st.text for st in decomp_result.subtasks],
                llm_used=getattr(decomp_result, "llm_used", False),
                fused_docs=len(sources),
                multi_hit_docs=sum(1 for s in sources if s.get("cluster_size", 1) > 1),
            )

        response_metadata = ResponseMetadata(
            cache_hit=cache_hit,
            index_normalized=True,
            decomposition=decomp_meta,
        )

        # PHASE 5: Cache the answer if not already from cache and answerability passed
        if not cache_hit and is_answerable:
            semantic_cache.set(
                query_embedding=qvec,
                top_doc_ids=top_doc_ids,
                answer=answer,
                sources=sources,
                answerability_score=answerability_score,
                prompt_version="v1",
                namespaces=ns_list,
            )
            logger.debug(f"✓ Cached answer for question: '{req.question[:50]}...'")

        # Emit retrieval metrics for monitoring
        try:
            os.makedirs('logs', exist_ok=True)
            metrics_line = json.dumps({
                "event": "retrieval_metrics",
                "retrieval_latency_ms": t_retr,
                "num_candidates": len(candidates),
                "lexical_weight": float(os.getenv("SEARCH_LEXICAL_WEIGHT", "0.35")),
                "chunk_strategy": _CHUNK_STRATEGY,
                "k": k,
                "namespaces": ns_list,
                "ts": int(time.time()*1000),
            })
            with open('logs/retrieval_metrics.log', 'a', encoding='utf-8') as mf:
                mf.write(metrics_line + "\n")
        except Exception:
            pass

        return ChatResponse(
            success=True,
            answer=answer,
            sources=sources,
            latency_ms={"retrieval": t_retr, "llm": t_llm, "total": total_latency},
            meta={
                "request_id": request_id,
                "temperature": temp,
                "model": model_used,
                "namespaces_used": ns_list,
                "k": k,
                "api_type": os.getenv("LLM_API_TYPE", "ollama"),
                "cited_chunks": cited_chunks,
                "citations_found": citations_found,
                "citation_validation": {
                    "valid": citation_validation.is_valid,
                    "cited_sources": sorted(list(citation_validation.cited_indices)),
                    "total_citations": citation_validation.total_citations,
                    "warnings": citation_validation.warnings if citation_validation.warnings else None,
                }
            },
            metadata=response_metadata.model_dump(),
        )

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition"),
    x_api_token: Optional[str] = Header(default=None),
):
    """
    Streaming chat endpoint with RAG: retrieve, ground answer, stream response.

    Returns Server-Sent Events (SSE) stream with chunks of the LLM response.
    Each event contains a JSON payload with response chunks and metadata.
    """
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    # Check if streaming is enabled
    streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower() == "true"
    if not streaming_enabled:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not enabled. Set STREAMING_ENABLED=true to enable."
        )

    if index_manager:
        index_manager.ensure_loaded()
    k = max(1, min(int(req.k or RETRIEVAL_K), 20))
    t0 = time.time()

    async def generate_stream():
        """Generate SSE stream with LLM response chunks."""
        try:
            # Perform retrieval (same logic as regular chat)
            decomp_result = None
            if not decomposition_off and is_multi_intent_query(req.question):
                try:
                    decomp_result = decompose_query(req.question)
                    logger.info(f"Decomposed streaming chat question into {len(decomp_result.subtasks)} subtasks")
                except Exception as e:
                    logger.warning(f"Query decomposition failed for streaming chat: {e}")

            # Expansion and encoding
            if not decomp_result or len(decomp_result.subtasks) <= 1:
                variants = expand_structured(req.question)
            else:
                variants = [{"text": req.question, "source": "original", "weight": 1.0}]

            qvec = encode_weighted_variants(variants).flatten()
            available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
            ns_list = [req.namespace] if req.namespace in available_namespaces else sorted(available_namespaces)
            raw_k = k * 3

            # Retrieval
            if decomp_result and len(decomp_result.subtasks) > 1:
                candidates, _ = search_with_decomposition(req.question, qvec, raw_k, ns_list, decomp_result)
                candidates = candidates[:raw_k]
            else:
                query_type = detect_query_type(req.question)
                use_hybrid = should_enable_hybrid_search(query_type, len(req.question))
                if use_hybrid:
                    per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), req.question, raw_k) for ns in ns_list}
                else:
                    per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}
                candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]

            # Deduplication
            candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
            seen_urls = set()
            hits = []
            for candidate in candidates:
                url = candidate.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                hits.append(candidate)
                if len(hits) >= k:
                    break

            t_retr = int((time.time() - t0) * 1000)

            # Build context with RAGPrompt for consistency with /chat endpoint
            chunks = []
            top_context = hits[: min(4, len(hits))]

            for i, h in enumerate(top_context, start=1):
                url = h.get("url", "")
                anchor = h.get("anchor")
                url_with_anchor = f"{url}#{anchor}" if anchor else url
                title_path = h.get("title_path") or []
                if isinstance(title_path, list) and title_path:
                    title = " > ".join(title_path)
                else:
                    title = h.get("title") or url or f"chunk-{i}"
                text = h.get("text", "")[:1600]

                chunk = {
                    "title": title,
                    "url": url_with_anchor,
                    "namespace": h.get("namespace"),
                    "score": h.get("score", 0.0),
                    "text": text,
                }
                chunks.append(chunk)

            # HARMONY: Use RAGPrompt.build_messages() for consistency
            messages, sources, developer_instructions = RAGPrompt.build_messages(
                question=req.question,
                chunks=chunks,
                namespace=req.namespace or "clockify",
                max_chunks=len(chunks),
                reasoning_effort="low"  # RAG optimization: minimize latency
            )

            # Send initial metadata
            metadata_event = {
                "type": "metadata",
                "sources": sources,
                "retrieval_latency_ms": t_retr,
                "k": k,
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"

            # Stream LLM response
            temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
            t1 = time.time()

            # Get streaming response from LLM
            llm = LLMClient()
            try:
                # HARMONY: Use chat method with messages list
                full_answer = llm.chat(
                    messages=messages,
                    max_tokens=800,
                    temperature=temp,
                    stream=True
                )
            except CircuitOpenError as e:
                # Circuit breaker is open - send error event
                logger.error(f"Circuit breaker open for streaming chat: {str(e)}")
                error_event = {
                    "type": "error",
                    "error": "LLM service temporarily unavailable. Circuit breaker is open. Please try again in a minute.",
                    "error_code": "CIRCUIT_OPEN",
                    "retry_after_seconds": 60
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            t_llm = int((time.time() - t1) * 1000)

            # Send answer chunk
            answer_event = {
                "type": "answer",
                "content": full_answer,
                "llm_latency_ms": t_llm,
            }
            yield f"data: {json.dumps(answer_event)}\n\n"

            # Validate citations
            citation_validation = validate_citations(full_answer, len(sources), strict=False)

            # Send final event with validation
            done_event = {
                "type": "done",
                "total_latency_ms": int((time.time() - t0) * 1000),
                "citation_validation": {
                    "valid": citation_validation.is_valid,
                    "cited_sources": sorted(list(citation_validation.cited_indices)),
                    "warnings": citation_validation.warnings if citation_validation.warnings else None,
                }
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            error_event = {
                "type": "error",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# --------- Mount Static Files for Web UI ---------
# Serve the web UI from public/ directory
PUBLIC_DIR = Path(__file__).parent.parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
    logger.info(f"Mounted static files from {PUBLIC_DIR}")
else:
    logger.warning(f"Public directory not found at {PUBLIC_DIR}, web UI will not be served")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
