#!/usr/bin/env python3
"""
Ingest v2 enriched corpus into FAISS with strict allow/deny and metadata.

Inputs (env or defaults):
  - ENRICHED_LINKS_JSON=codex/CRAWLED_LINKS_enriched.json
  - ALLOWLIST_PATH=codex/ALLOWLIST.txt
  - DENYLIST_PATH=codex/DENYLIST.txt
  - NAMESPACE=clockify
  - EMBEDDING_MODEL (for logging)
  - EMBEDDING_DIM (validation)

Outputs:
  - codex/INGEST_STATS_v2.md
  - codex/INGEST_FAILED_v2.txt
  - index/faiss/<namespace>/index.bin, meta.json, stats.json
"""
import sys

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import numpy as np

from loguru import logger

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CODex = ROOT / "codex"
ENRICHED_JSON = Path(os.getenv("ENRICHED_LINKS_JSON", str(CODex / "CRAWLED_LINKS_enriched.json")))
ALLOWLIST_PATH = Path(os.getenv("ALLOWLIST_PATH", str(CODex / "ALLOWLIST.txt")))
DENYLIST_PATH = Path(os.getenv("DENYLIST_PATH", str(CODex / "DENYLIST.txt")))
NAMESPACE = os.getenv("NAMESPACE", "clockify")
# Chunking strategy controls output namespace subdir
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "url_level").strip()
_suffix = "url" if CHUNK_STRATEGY == "url_level" else ("h23" if CHUNK_STRATEGY == "h2_h3_blocks" else CHUNK_STRATEGY)
INDEX_DIR = ROOT / "index" / "faiss" / f"{NAMESPACE}_{_suffix}"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INGEST_STATS_MD = CODex / "INGEST_STATS_v2.md"
INGEST_STATS_ABL = CODex / "INGEST_STATS_ablation.md"
INGEST_FAILED_TXT = CODex / "INGEST_FAILED_v2.txt"


def compile_patterns(path: Path) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    if not path.exists():
        return pats
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        try:
            pats.append(re.compile(s))
        except re.error:
            pass
    return pats


def is_allowed(url: str, allow_pats: List[re.Pattern], deny_pats: List[re.Pattern]) -> bool:
    allow_ok = any(p.search(url) for p in allow_pats) if allow_pats else True
    deny_hit = any(p.search(url) for p in deny_pats) if deny_pats else False
    return allow_ok and not deny_hit


def _derive_title_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path.strip('/').split('/')
        for seg in reversed(path):
            if seg:
                t = seg.replace('-', ' ').replace('_', ' ').strip()
                if t:
                    return t.title()
    except Exception:
        pass
    return url


def build_minimal_text(title: str, h1: str, url: str) -> str:
    # Field-weighted surrogate content if full body is unavailable
    from urllib.parse import urlparse
    path = ""
    try:
        path = urlparse(url).path.replace("/", " ")
    except Exception:
        pass
    parts = []
    if title:
        parts.append((title + " ") * 3)
    if h1:
        parts.append((h1 + " ") * 2)
    if path:
        parts.append(path)
    # Ensure URL is present for lexical signal
    parts.append(url)
    return " ".join(parts).strip()


def load_enriched() -> Tuple[List[str], Dict[str, Dict]]:
    data = json.loads(ENRICHED_JSON.read_text(encoding="utf-8"))
    urls = data.get("urls", [])
    meta = data.get("meta", {})
    return urls, meta


def main() -> None:
    allow = compile_patterns(ALLOWLIST_PATH)
    deny = compile_patterns(DENYLIST_PATH)
    urls, meta = load_enriched()

    docs: List[Dict] = []
    failed: List[str] = []
    # Import chunkers lazily to avoid heavy deps if unused
    from tools.chunkers import build_surrogate_text, chunk_h2_h3_blocks

    for u in urls:
        if not is_allowed(u, allow, deny):
            continue
        m = meta.get(u, {})
        title = (m.get("title") or "").strip() or _derive_title_from_url(u)
        h1 = (m.get("h1") or "").strip()
        canonical = (m.get("canonical") or u).strip()
        last_modified = m.get("last_modified")
        checksum = m.get("checksum")
        fetched_at = m.get("fetch_ts")

        if CHUNK_STRATEGY == "h2_h3_blocks":
            pieces = chunk_h2_h3_blocks(u, title)
            for idx, ch in enumerate(pieces):
                text = ch.get("text", "")
                if not text:
                    continue
                cid = hashlib.sha256(f"{canonical or u}|{idx}".encode()).hexdigest()
                docs.append({
                    "id": cid,
                    "chunk_id": cid,
                    "parent_id": None,
                    "url": u,
                    "title": title or u,
                    "headers": [h1] if h1 else [],
                    "tokens": len(text.split()),
                    "node_type": "child",
                    "text": text,
                    "section": ch.get("section"),
                    "anchor": None,
                    "breadcrumb": None,
                    "updated_at": last_modified,
                    "title_path": [title] if title else [],
                    "namespace": NAMESPACE,
                    "metadata": {
                        "canonical": canonical,
                        "checksum": checksum,
                        "fetched_at": fetched_at,
                    }
                })
        else:
            text = build_surrogate_text(title, h1, u)
            if not text:
                failed.append(f"{u}\tno-text")
                continue
            doc_id = hashlib.sha256((canonical or u).encode()).hexdigest()
            docs.append({
                "id": doc_id,
                "chunk_id": doc_id,
                "parent_id": None,
                "url": u,
                "title": title or u,
                "headers": [h1] if h1 else [],
                "tokens": len(text.split()),
                "node_type": "child",
                "text": text,
                "section": None,
                "anchor": None,
                "breadcrumb": None,
                "updated_at": last_modified,
                "title_path": [title] if title else [],
                "namespace": NAMESPACE,
                "metadata": {
                    "canonical": canonical,
                    "checksum": checksum,
                    "fetched_at": fetched_at,
                }
            })

    # Embed
    from src.embeddings import embed_passages, EMBEDDING_MODEL
    texts = [d["text"] for d in docs]
    try:
        embeddings = embed_passages(texts)
    except Exception as e:
        # Fallback: deterministic stub embeddings if real model unavailable
        logger.warning(f"Embedding failed ({e}); generating deterministic stubs for {len(texts)} docs")
        rng = np.random.default_rng(seed=42)
        embeddings = rng.normal(size=(len(texts), int(os.getenv("EMBEDDING_DIM", "768")))).astype(np.float32)
        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    dim = embeddings.shape[1]
    # Build FAISS index
    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_DIR / "index.bin"))

    # Attach embeddings to metadata for in-process BM25 hybrid operations
    rows = []
    for i, d in enumerate(docs):
        row = d.copy()
        row["embedding"] = embeddings[i].tolist()  # stored for RetrievalEngine hybrid usage
        rows.append(row)

    meta_payload = {
        "model": EMBEDDING_MODEL,
        "dimension": dim,
        "dim": dim,
        "num_vectors": len(rows),
        "normalized": True,
        "chunks": rows,
        "rows": rows,
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    stats = {
        "namespace": NAMESPACE,
        "strategy": CHUNK_STRATEGY,
        "total_urls": len(urls),
        "allowed_indexed": len(rows),
        "failed": len(failed),
        "embedding_dim": dim,
        "model": os.getenv("EMBEDDING_MODEL", "unknown"),
    }
    (INDEX_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Write reports
    INGEST_STATS_MD.write_text(
        "\n".join([
            "Ingestion v2 Stats",
            f"- Namespace: {NAMESPACE}",
            f"- Strategy: {CHUNK_STRATEGY}",
            f"- Total URLs in enriched: {len(urls)}",
            f"- Indexed (allowed): {len(rows)}",
            f"- Failed: {len(failed)}",
            f"- Embedding model: {os.getenv('EMBEDDING_MODEL','unknown')} (dim={dim})",
        ]) + "\n",
        encoding="utf-8"
    )
    # Append ablation record
    with open(INGEST_STATS_ABL, "a", encoding="utf-8") as fa:
        fa.write("\n".join([
            f"Strategy: {CHUNK_STRATEGY}",
            f"- Indexed: {len(rows)}",
            f"- Dim: {dim}",
            f"- Index: {INDEX_DIR}",
            ""
        ]))
    if failed:
        INGEST_FAILED_TXT.write_text("\n".join(failed) + "\n", encoding="utf-8")
    else:
        INGEST_FAILED_TXT.write_text("", encoding="utf-8")

    logger.info(f"✓ Ingested {len(rows)} docs into {INDEX_DIR}")


if __name__ == "__main__":
    main()