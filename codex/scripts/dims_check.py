#!/usr/bin/env python3
"""
Embedding dimension consistency checker.

Checks:
- Parses index/faiss/*/meta.json and collects dimensions.
- Verifies all namespaces share the same dimension.
- Compares against configured embedding dim (env) and the code constant if importable.
- Ensures meta.json does not contain raw 'embedding' entries.

Exits with non-zero if severe inconsistencies are detected.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path("~/Downloads/rag").expanduser()
INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT", ROOT / "index/faiss"))


def load_dims() -> Dict[str, int]:
    dims: Dict[str, int] = {}
    if INDEX_ROOT.exists():
        for ns_dir in INDEX_ROOT.iterdir():
            if not ns_dir.is_dir():
                continue
            meta = ns_dir / "meta.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text())
                dim = int(data.get("dim") or data.get("dimension"))
                dims[ns_dir.name] = dim
            except Exception:
                pass
    return dims


def any_meta_contains_embeddings() -> List[str]:
    offenders: List[str] = []
    if INDEX_ROOT.exists():
        for ns_dir in INDEX_ROOT.iterdir():
            meta = ns_dir / "meta.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text())
                # meta.json should not store embeddings
                txt = meta.read_text()
                if '"embedding"' in txt:
                    offenders.append(ns_dir.name)
            except Exception:
                continue
    return offenders


def configured_dim() -> int:
    # Prefer importing code constant
    try:
        from src.embeddings import EMBEDDING_DIM  # type: ignore
        return int(EMBEDDING_DIM)
    except Exception:
        return int(os.getenv("EMBEDDING_DIM", "768"))


def main() -> None:
    dims = load_dims()
    conf_dim = configured_dim()

    print("Index namespace dims:")
    for ns, d in sorted(dims.items()):
        print(f"- {ns}: {d}")
    if not dims:
        print("No meta.json files found.")

    unique = sorted(set(dims.values()))
    if len(unique) > 1:
        print(f"SEVERE: Mixed dimensions across namespaces: {unique}")
    else:
        print(f"All namespaces dim={unique[0] if unique else 'unknown'}; configured={conf_dim}")
        if unique and unique[0] != conf_dim:
            print("MISMATCH: Index dim != configured EMBEDDING_DIM")

    offenders = any_meta_contains_embeddings()
    if offenders:
        print(f"ERROR: meta.json files contain raw 'embedding' entries: {offenders}")
        sys.exit(2)

    # Non-fatal exit code unless mixed dimensions detected
    if len(unique) > 1:
        sys.exit(1)


if __name__ == "__main__":
    main()

