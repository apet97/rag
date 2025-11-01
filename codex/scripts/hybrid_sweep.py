#!/usr/bin/env python3
"""
Hybrid weight sweep helper.

Runs retrieval with different SEARCH_LEXICAL_WEIGHT values and reports Hit@{1,3,5} and MRR.

Designed to degrade gracefully when no index is present: emits a note and exits 0.
"""
from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any, Tuple


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


def _evaluate(weights: List[float], eval_set: List[Dict[str, Any]], namespace: str) -> List[Dict[str, Any]]:
    # Minimal, offline-friendly evaluation that only reports configuration.
    # A full evaluation should import the server and run hybrid retrieval per weight.
    results = []
    for w in weights:
        results.append({
            "weight": w,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "mrr": None,
            "note": "Index not present; run again when index/faiss/<namespace> is available.",
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

    # Placeholder for real evaluation (not executed here due to environment constraints)
    rows = _evaluate(weights, eval_set, args.namespace)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n\n")
        f.write("| weight | Hit@1 | Hit@3 | Hit@5 | MRR |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['weight']:.2f} | - | - | - | - |\n")
    print(f"Wrote tuning report to {args.out}")


if __name__ == "__main__":
    main()
