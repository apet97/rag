#!/usr/bin/env python3
"""
Hybrid weight sweep helper.

Runs retrieval with different SEARCH_LEXICAL_WEIGHT values and reports Hit@{1,3,5} and MRR.

Designed to degrade gracefully when no index is present: emits a note and exits 0.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure repo root is on path to import src.* modules
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _index_present(index_root: str, namespace: str) -> bool:
    meta = os.path.join(index_root, namespace, "meta.json")
    idx1 = os.path.join(index_root, namespace, "index.faiss")
    idx2 = os.path.join(index_root, namespace, "index.bin")
    return os.path.exists(meta) and (os.path.exists(idx1) or os.path.exists(idx2))


def _load_eval(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def _canon(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    # Lowercase, drop fragment and trailing slash
    from urllib.parse import urlparse
    p = urlparse(url)
    path = p.path.rstrip('/')
    return f"{p.scheme}://{p.netloc}{path}"


def _evaluate_real(weights: List[float], eval_set: List[Dict[str, Any]], index_root: str, namespace: str) -> List[Dict[str, Any]]:
    import os
    from pathlib import Path
    # Force stub embeddings to avoid heavy model imports during sweep
    os.environ.setdefault('EMBEDDINGS_BACKEND', 'stub')
    from src import embeddings
    from src import server
    from src import config as CFG

    # Prepare server index manager for the given namespace dir
    os.environ['RAG_INDEX_ROOT'] = index_root
    os.environ['NAMESPACES'] = namespace
    server.index_manager = server.IndexManager(Path(index_root), [namespace])
    server.index_manager.ensure_loaded()

    results: List[Dict[str, Any]] = []
    for w in weights:
        # Set lexical weight dynamically
        CFG.SEARCH_LEXICAL_WEIGHT = str(w)
        total = len(eval_set)
        hit1 = hit3 = hit5 = 0
        mrr_sum = 0.0
        for item in eval_set:
            q = item.get('question', '')
            expected: List[str] = [
                _canon(u) for u in item.get('expected', [])
            ]
            if not q or not expected:
                total -= 1
                continue
            qvec = embeddings.embed_query(q)
            try:
                res = server.search_ns_hybrid(namespace, qvec, q, k=5)
            except Exception:
                res = []
            cand_urls = [_canon(r.get('url','')) for r in res[:5]]
            # rank of first match
            rank: Optional[int] = None
            for idx, cu in enumerate(cand_urls, start=1):
                if any(cu.startswith(e) or e.startswith(cu) for e in expected):
                    rank = idx
                    break
            if rank is not None:
                if rank == 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 5:
                    hit5 += 1
                mrr_sum += 1.0 / rank
        if total <= 0:
            results.append({"weight": w, "hit_at_1": None, "hit_at_3": None, "hit_at_5": None, "mrr": None, "note": "no valid eval items"})
        else:
            results.append({
                "weight": w,
                "hit_at_1": hit1/total,
                "hit_at_3": hit3/total,
                "hit_at_5": hit5/total,
                "mrr": mrr_sum/total,
            })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", default=os.getenv("RAG_INDEX_ROOT", "index/faiss"))
    ap.add_argument("--namespace", default=os.getenv("NAMESPACE", "clockify"))
    ap.add_argument("--eval", default="codex/RAG_EVAL_TASKS.jsonl")
    ap.add_argument("--out", default="codex/HYBRID_TUNING.md")
    args = ap.parse_args()

    if not os.path.exists(args.eval):
        # Write placeholder report even when eval is missing
        header = [
            "# Hybrid Tuning",
            "",
            f"- Namespace: {args.namespace}",
            f"- Index root: {args.index_root}",
            f"- Eval: missing at {args.eval}",
            "",
            "Evaluation set not found. Provide codex/RAG_EVAL_TASKS.jsonl and re-run.",
        ]
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(header))
        print(f"Wrote placeholder tuning report to {args.out}")
        return

    weights = [0.20, 0.35, 0.50]
    eval_set = _load_eval(args.eval)

    header = [
        "# Hybrid Tuning",
        "",
        f"- Namespace: {args.namespace}",
        f"- Index root: {args.index_root}",
        f"- Eval size: {len(eval_set)}",
        "",
    ]

    if not _index_present(args.index_root, args.namespace):
        header.append(
            "Index not present. This run records placeholders. Re-run after building indexes (make ingest_v2)."
        )
        rows = _evaluate(weights, eval_set, args.namespace)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(header))
            f.write("\n\n")
            f.write("| weight | Hit@1 | Hit@3 | Hit@5 | MRR |\n")
            f.write("|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['weight']:.2f} | - | - | - | - |\n")
        print(f"Wrote placeholder tuning report to {args.out}")
        return

    # Real evaluation using in-process retrieval
    rows = _evaluate_real(weights, eval_set, args.index_root, args.namespace)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n\n")
        f.write("| weight | Hit@1 | Hit@3 | Hit@5 | MRR |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r.get('hit_at_1') is None:
                f.write(f"| {r['weight']:.2f} | - | - | - | - |\n")
            else:
                f.write(
                    f"| {r['weight']:.2f} | {r['hit_at_1']:.2f} | {r['hit_at_3']:.2f} | {r['hit_at_5']:.2f} | {r['mrr']:.3f} |\n"
                )
    print(f"Wrote tuning report to {args.out}")


if __name__ == "__main__":
    main()
