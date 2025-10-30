#!/usr/bin/env python3
"""Build multi-namespace FAISS vector indexes."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("index/faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


class EmbeddingBuilder:
    """Build per-namespace FAISS indexes."""

    def __init__(self):
        if not SentenceTransformer:
            raise ImportError("sentence-transformers required")
        if not faiss:
            raise ImportError("faiss-cpu required")

        logger.info(f"Loading model: {EMBEDDING_MODEL}")
        # SECURITY: Do not use trust_remote_code=True (RCE risk)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512

    def build_index_for_namespace(self, namespace: str) -> bool:
        """Build index for a single namespace."""
        chunks_file = CHUNKS_DIR / f"{namespace}.jsonl"

        if not chunks_file.exists():
            logger.warning(f"Chunks file not found: {chunks_file}")
            return False

        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            logger.warning(f"No chunks in {chunks_file}")
            return False

        logger.info(f"Building index for {namespace}: {len(chunks)} chunks")

        # Embed with E5 prompt format for passages
        texts = [c["text"] for c in chunks]
        embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # E5 format: prefix each passage with "passage: "
            batch_with_prefix = [f"passage: {text}" for text in batch]
            batch_emb = self.model.encode(batch_with_prefix, convert_to_numpy=True)
            embeddings.append(batch_emb)
            if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0:
                logger.info(f"  → {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

        embeddings = np.vstack(embeddings)
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

        # L2-normalize for cosine similarity with inner product
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        embeddings = embeddings.astype(np.float32)

        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info(f"✓ Index built: {index.ntotal} vectors")

        # Save
        ns_dir = INDEX_DIR / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(ns_dir / "index.bin"))
        def _meta_entry(c: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": c.get("id"),
                "chunk_id": c.get("chunk_id"),
                "parent_id": c.get("parent_id"),
                "url": c.get("url"),
                "title": c.get("title"),
                "headers": c.get("headers"),
                "tokens": c.get("tokens"),
                "node_type": c.get("node_type", "child"),
                "text": c.get("text", ""),
                "section": c.get("section"),
                "anchor": c.get("anchor"),
                "breadcrumb": c.get("breadcrumb"),
                "updated_at": c.get("updated_at"),
                "title_path": c.get("title_path"),
            }

        meta_payload = {
            "model": EMBEDDING_MODEL,
            "dimension": dim,
            "dim": dim,
            "num_vectors": index.ntotal,
            "normalized": True,
            "chunks": [_meta_entry(c) for c in chunks],
            "rows": [_meta_entry(c) for c in chunks],
        }

        with open(ns_dir / "meta.json", "w") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"✓ Saved index for {namespace} to {ns_dir}")
        return True


async def main():
    """Build indexes for all namespaces."""
    builder = EmbeddingBuilder()

    for chunks_file in CHUNKS_DIR.glob("*.jsonl"):
        namespace = chunks_file.stem
        if namespace.startswith("."):
            continue
        builder.build_index_for_namespace(namespace)

    logger.info("✓ Embedding complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
