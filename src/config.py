#!/usr/bin/env python3
"""Centralized configuration with validation and sensible defaults."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT / "codex"


def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    return val if val is not None else (default or "")


# Namespaces
NAMESPACE: str = _get_env("NAMESPACE", "clockify")
NAMESPACES: list[str] = [s.strip() for s in _get_env("NAMESPACES", NAMESPACE).split(",") if s.strip()]

# Embeddings
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
try:
    EMBEDDING_DIM: int = int(_get_env("EMBEDDING_DIM", "768"))
except ValueError:
    raise RuntimeError("Invalid EMBEDDING_DIM; must be integer.")

# Retrieval hybrid weight (lexical share)
def _parse_float(name: str, default: float) -> float:
    try:
        return float(_get_env(name, str(default)))
    except ValueError:
        return default


SEARCH_LEXICAL_WEIGHT: float = _parse_float("SEARCH_LEXICAL_WEIGHT", 0.35)
if not (0.0 <= SEARCH_LEXICAL_WEIGHT <= 1.0):
    raise RuntimeError("SEARCH_LEXICAL_WEIGHT must be in [0,1].")

# Chunking strategy
CHUNK_STRATEGY: str = _get_env("CHUNK_STRATEGY", "url_level").strip()
CHUNK_SIZE: int = int(_get_env("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP: int = int(_get_env("CHUNK_OVERLAP", "200"))

# Policy paths
ALLOWLIST_PATH: Path = Path(_get_env("ALLOWLIST_PATH", str(CODEX / "ALLOWLIST.txt")))
DENYLIST_PATH: Path = Path(_get_env("DENYLIST_PATH", str(CODEX / "DENYLIST.txt")))

# Enriched corpus path
ENRICHED_LINKS_JSON: Path = Path(_get_env("ENRICHED_LINKS_JSON", str(CODEX / "CRAWLED_LINKS_enriched.json")))

# Index root
INDEX_ROOT: Path = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))


def health_summary() -> dict:
    """Return a health summary for /healthz and /readyz."""
    digest_file = CODEX / "INDEX_DIGEST.txt"
    digest: Optional[str] = None
    if digest_file.exists():
        try:
            lines = digest_file.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.startswith("combined="):
                    digest = line.split("=", 1)[-1].strip()
        except Exception:
            digest = None
    return {
        "namespace": NAMESPACE,
        "index_root": str(INDEX_ROOT),
        "search_lexical_weight": SEARCH_LEXICAL_WEIGHT,
        "chunk_strategy": CHUNK_STRATEGY,
        "embedding_dim": EMBEDDING_DIM,
        "index_digest": digest,
    }

# Back-compat shim for server references
class _Compat:
    MAX_CONTEXT_CHUNKS: int = int(_get_env("MAX_CONTEXT_CHUNKS", "8"))
    CONTEXT_CHAR_LIMIT: int = int(_get_env("CONTEXT_CHAR_LIMIT", "1200"))
    ANSWERABILITY_THRESHOLD: float = _parse_float("ANSWERABILITY_THRESHOLD", 0.18)


CONFIG = _Compat()
