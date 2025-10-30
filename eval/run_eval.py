#!/usr/bin/env python3
"""Clockify RAG evaluation harness.

Computes retrieval and mock-generation metrics against a gold set.
Falls back to offline FAISS evaluation when the HTTP API is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

# Ensure project root is importable when executed as a script
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import requests
except Exception:  # pragma: no cover - requests is optional for offline mode
    requests = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - validated at runtime
    raise SystemExit("faiss is required for offline evaluation. Install faiss-cpu.") from exc

from src.embeddings import encode_texts, embed_query
from src.query_expand import expand
from src.rerank import rerank, is_available as rerank_available
from src.query_decomposition import decompose_query, is_multi_intent_query
from src.server import detect_query_type, should_enable_hybrid_search


INDEX_ROOT = Path("index/faiss")
DEFAULT_NAMESPACES = ["clockify"]


@dataclass
class GoldItem:
    """Single gold-set entry."""

    id: str
    question: str
    answer_regex: str
    source_urls: List[str]


@dataclass
class DecompositionHitInfo:
    """Per-query decomposition and hit tracking."""

    strategy: str  # "none", "heuristic", or "llm"
    subtask_count: int
    subtasks: List[str]
    subtask_hits: Dict[int, List[str]]  # subtask_index -> list of retrieved URLs
    llm_used: bool


def load_goldset(path: Path) -> List[GoldItem]:
    """Load CSV goldset with required columns."""
    items: List[GoldItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "answer_regex", "source_url"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"goldset is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            src_urls = [u.strip() for u in row["source_url"].split("|") if u.strip()]
            items.append(
                GoldItem(
                    id=row["id"].strip(),
                    question=row["question"].strip(),
                    answer_regex=row["answer_regex"].strip(),
                    source_urls=src_urls,
                )
            )
    return items


def normalize_url(url: str) -> str:
    """Strip anchors and trailing slashes for stable comparison."""
    if not url:
        return url
    base = url.split("#", 1)[0]
    return base.rstrip("/")


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for g in gold if any(normalize_url(g) == normalize_url(r) for r in top))
    return hits / len(gold)


def mrr_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 1.0
    top = retrieved[:k]
    for rank, url in enumerate(top, start=1):
        if any(normalize_url(url) == normalize_url(g) for g in gold):
            return 1.0 / rank
    return 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    rank = (len(vals) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return vals[int(rank)]
    return vals[lower] + (vals[upper] - vals[lower]) * (rank - lower)


def fuse_results(by_ns: Dict[str, List[Dict]], cap: int) -> List[Dict]:
    """Reciprocal-rank fusion mirroring server logic."""
    scores: Dict[Tuple[str, str], float] = {}
    payloads: Dict[Tuple[str, str], Dict] = {}
    C = 60.0
    for ns, docs in by_ns.items():
        for rank, doc in enumerate(docs, start=1):
            url = doc.get("url", "")
            chunk_id = doc.get("chunk_id", doc.get("id", f"{ns}-{rank}"))
            key = (url, chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (C + rank)
            payloads[key] = doc
    merged = sorted(
        payloads.values(),
        key=lambda doc: scores[(doc.get("url", ""), doc.get("chunk_id", doc.get("id", "")))],
        reverse=True,
    )
    return merged[:cap]


class HttpRetriever:
    """Wrapper over the running FastAPI /search endpoint."""

    def __init__(self, base_url: str, api_token: Optional[str] = None, disable_decomposition: bool = False):
        if requests is None:  # pragma: no cover - handled in load
            raise RuntimeError("requests is required for HTTP evaluation")
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-api-token": api_token or "change-me"}
        self.disable_decomposition = disable_decomposition

    def healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int) -> Tuple[List[Dict], float, Optional[DecompositionHitInfo]]:
        """Retrieve results and optionally return decomposition hit info.

        Returns:
            (results, latency_ms, decomposition_info)
        """
        t0 = time.perf_counter()
        try:
            params = {"q": question, "k": top_k}
            if self.disable_decomposition:
                params["decomposition_off"] = "true"
            resp = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=self.headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            # Extract decomposition metadata if available
            decomp_info = None
            metadata = data.get("metadata", {})
            if metadata and "subtasks" in metadata and not self.disable_decomposition:
                decomp_info = self._build_decomp_hit_info(question, metadata)
        except Exception as exc:
            raise RuntimeError(f"HTTP retrieval failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms, decomp_info

    def _build_decomp_hit_info(self, question: str, metadata: Dict) -> Optional[DecompositionHitInfo]:
        """Build decomposition hit info from server metadata."""
        try:
            subtasks = metadata.get("subtasks", [])
            if not subtasks:
                return None

            subtask_texts = [st.get("text", "") for st in subtasks]
            subtask_hits: Dict[int, List[str]] = {}

            # Extract which subtasks hit which URLs from fused docs metadata
            for st_idx in range(len(subtasks)):
                subtask_hits[st_idx] = []

            # If server provides per-doc hit mapping, parse it
            fused_docs = metadata.get("fused_docs", 0)
            llm_used = metadata.get("llm_used", False)

            return DecompositionHitInfo(
                strategy=metadata.get("decomposition_strategy", "none"),
                subtask_count=len(subtasks),
                subtasks=subtask_texts,
                subtask_hits=subtask_hits,
                llm_used=llm_used,
            )
        except Exception:
            return None


class OfflineRetriever:
    """FAISS-based retriever mirroring server semantics."""

    def __init__(self, namespaces: List[str], oversample: int = 60, enable_rerank: bool = True, disable_decomposition: bool = False):
        self.oversample = oversample
        self.enable_rerank = enable_rerank and rerank_available()
        self.disable_decomposition = disable_decomposition
        self.namespaces = namespaces or DEFAULT_NAMESPACES
        self._indexes: Dict[str, Dict[str, object]] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        for ns in self.namespaces:
            root = INDEX_ROOT / ns
            if not root.exists():
                raise RuntimeError(f"Missing index directory for namespace '{ns}' under {root}")
            idx_path = root / "index.faiss"
            if not idx_path.exists():
                idx_path = root / "index.bin"
            if not idx_path.exists():
                raise RuntimeError(f"Missing FAISS index for namespace '{ns}'")
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text())
            metas = meta.get("chunks") or meta.get("rows") or []
            index = faiss.read_index(str(idx_path))
            self._indexes[ns] = {"index": index, "metas": metas}

    def retrieve(self, question: str, top_k: int) -> Tuple[List[Dict], float, Optional[DecompositionHitInfo]]:
        """Retrieve results with optional decomposition.

        Returns:
            (results, latency_ms, decomposition_info)
        """
        t0 = time.perf_counter()

        # Check if decomposition should be used
        decomp_info = None
        if not self.disable_decomposition and is_multi_intent_query(question):
            decomp_info = self._retrieve_with_decomposition(question, top_k)
            if decomp_info:
                deduped, latency_ms = self._fuse_decomposed_results(decomp_info, question, top_k, t0)
                return deduped, latency_ms, decomp_info

        # Standard (non-decomposed) retrieval
        expansions = expand(question)
        vectors = encode_texts(expansions)
        qvec = np.mean(vectors, axis=0)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-8)
        qvec = qvec.astype(np.float32)

        by_ns: Dict[str, List[Dict]] = {}
        for ns, bundle in self._indexes.items():
            index: faiss.Index = bundle["index"]  # type: ignore[assignment]
            metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
            raw_k = min(max(top_k * 6, self.oversample), index.ntotal)
            distances, indices = index.search(qvec.reshape(1, -1), raw_k)
            ns_results: List[Dict] = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(metas):
                    continue
                meta = dict(metas[idx])
                meta.update({
                    "namespace": ns,
                    "score": float(score),
                    "rank": len(ns_results) + 1,
                })
                ns_results.append(meta)
            by_ns[ns] = ns_results

        fused = fuse_results(by_ns, max(top_k * 3, top_k))
        fused.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))

        deduped: List[Dict] = []
        seen_urls: set[str] = set()
        for candidate in fused:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(candidate)
            if len(deduped) >= top_k:
                break

        if self.enable_rerank and deduped:
            reranked = rerank(question, deduped, top_k)
            if reranked:
                deduped = reranked

        latency_ms = (time.perf_counter() - t0) * 1000
        return deduped, latency_ms, decomp_info

    def _retrieve_with_decomposition(self, question: str, top_k: int) -> Optional[DecompositionHitInfo]:
        """Perform per-subtask retrieval and track hits."""
        try:
            decomp_result = decompose_query(question)
            if not decomp_result or len(decomp_result.subtasks) <= 1:
                return None

            subtask_texts = [st.text for st in decomp_result.subtasks]
            subtask_hits: Dict[int, List[str]] = {}

            # Per-subtask retrieval
            for st_idx, subtask in enumerate(decomp_result.subtasks):
                expansions = expand(subtask.text, boost_terms=subtask.boost_terms)
                vectors = encode_texts(expansions)
                qvec = np.mean(vectors, axis=0)
                qvec = qvec / (np.linalg.norm(qvec) + 1e-8)
                qvec = qvec.astype(np.float32)

                hit_urls = []
                for ns, bundle in self._indexes.items():
                    index: faiss.Index = bundle["index"]  # type: ignore[assignment]
                    metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
                    raw_k = min(max(top_k * 2, 20), index.ntotal)
                    distances, indices = index.search(qvec.reshape(1, -1), raw_k)
                    for score, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(metas):
                            continue
                        url = metas[idx].get("url", "")
                        if url and url not in hit_urls:
                            hit_urls.append(url)
                subtask_hits[st_idx] = hit_urls

            return DecompositionHitInfo(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=subtask_texts,
                subtask_hits=subtask_hits,
                llm_used=decomp_result.llm_used,
            )
        except Exception:
            return None

    def _fuse_decomposed_results(self, decomp_info: DecompositionHitInfo, question: str, top_k: int, t0: float) -> Tuple[List[Dict], float]:
        """Fuse per-subtask results into final ranking."""
        fused_docs: Dict[Tuple[str, str], Dict] = {}

        # Aggregate hits across subtasks
        for st_idx, urls in decomp_info.subtask_hits.items():
            for url in urls:
                key = (url, str(st_idx))  # Use subtask index as chunk_id for deduping
                if key not in fused_docs:
                    fused_docs[key] = {"url": url, "hits": 0, "subtasks": []}
                fused_docs[key]["hits"] += 1
                if st_idx not in fused_docs[key]["subtasks"]:
                    fused_docs[key]["subtasks"].append(st_idx)

        # Sort by hit count (additive fusion)
        sorted_docs = sorted(fused_docs.values(), key=lambda d: (-d["hits"], d["url"]))

        # Load full doc metadata from indexes
        deduped: List[Dict] = []
        seen_urls: set[str] = set()
        for doc in sorted_docs:
            url = doc["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Find full metadata from indexes
            for ns, bundle in self._indexes.items():
                metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
                for meta in metas:
                    if meta.get("url") == url:
                        full_doc = dict(meta)
                        full_doc["namespace"] = ns
                        deduped.append(full_doc)
                        break
                if len(deduped) >= top_k:
                    break
            if len(deduped) >= top_k:
                break

        if self.enable_rerank and deduped:
            reranked = rerank(question, deduped, top_k)
            if reranked:
                deduped = reranked

        latency_ms = (time.perf_counter() - t0) * 1000
        return deduped, latency_ms


def evaluate(goldset: List[GoldItem], retriever, top_k: int = 5, context_k: int = 4, log_decomposition: bool = False) -> Dict[str, object]:
    metrics = {
        "cases": len(goldset),
        "recall_at_5": [],
        "mrr_at_5": [],
        "answer_hits": [],
        "retrieval_latencies": [],
        "full_latencies": [],
        "case_details": [],
        "decomposition_stats": {  # A/B comparison stats
            "none": {"count": 0, "recall_sum": 0.0, "misses": []},
            "heuristic": {"count": 0, "recall_sum": 0.0, "misses": []},
            "multi_part": {"count": 0, "recall_sum": 0.0, "misses": []},
            "comparison": {"count": 0, "recall_sum": 0.0, "misses": []},
            "llm": {"count": 0, "recall_sum": 0.0, "misses": []},
        },
    }

    # Setup decomposition logging if requested
    decomp_log_file = None
    if log_decomposition:
        decomp_log_file = Path("logs/decomposition_eval.jsonl")
        decomp_log_file.parent.mkdir(parents=True, exist_ok=True)
        decomp_log_file.write_text("")  # Clear previous logs

    for item in goldset:
        try:
            result = retriever.retrieve(item.question, max(top_k, context_k))
            if len(result) == 3:
                results, retr_ms, decomp_info = result
            else:
                # Fallback for retrievers that don't return decomp_info
                results, retr_ms = result
                decomp_info = None
        except Exception as exc:
            results, retr_ms, decomp_info = [], float("inf"), None
            detail = {
                "id": item.id,
                "question": item.question,
                "error": str(exc),
            }
            metrics["case_details"].append(detail)
            metrics["recall_at_5"].append(0.0)
            metrics["mrr_at_5"].append(0.0)
            metrics["answer_hits"].append(False)
            metrics["retrieval_latencies"].append(retr_ms)
            metrics["full_latencies"].append(retr_ms)
            continue

        retrieved_urls = [r.get("url", "") for r in results]
        recall = recall_at_k(retrieved_urls, item.source_urls, 5)
        mrr = mrr_at_k(retrieved_urls, item.source_urls, 5)

        metrics["recall_at_5"].append(recall)
        metrics["mrr_at_5"].append(mrr)

        # Track decomposition strategy for A/B analysis
        decomp_strategy = "none"
        subtask_hit_details = {}
        if decomp_info:
            decomp_strategy = decomp_info.strategy
            metrics["decomposition_stats"][decomp_strategy]["count"] += 1
            metrics["decomposition_stats"][decomp_strategy]["recall_sum"] += recall

            # Log per-subtask hits if recall is 0 (miss)
            if recall == 0.0:
                miss_entry = {
                    "id": item.id,
                    "question": item.question,
                    "decomposition_strategy": decomp_strategy,
                    "subtasks": decomp_info.subtasks,
                    "subtask_hits": decomp_info.subtask_hits,
                    "llm_used": decomp_info.llm_used,
                    "gold_urls": item.source_urls,
                }
                metrics["decomposition_stats"][decomp_strategy]["misses"].append(miss_entry)

            # Build subtask hit summary for logging
            for st_idx, urls in decomp_info.subtask_hits.items():
                st_text = decomp_info.subtasks[st_idx] if st_idx < len(decomp_info.subtasks) else f"subtask_{st_idx}"
                hit_count = len([u for u in urls if any(normalize_url(u) == normalize_url(g) for g in item.source_urls)])
                subtask_hit_details[st_idx] = {
                    "text": st_text,
                    "retrieved_urls": urls,
                    "gold_hits": hit_count,
                }

        answer_start = time.perf_counter()
        context_blocks = results[: min(context_k, len(results))]
        context_text = "\n\n".join(r.get("text", "") for r in context_blocks)
        try:
            pattern = re.compile(item.answer_regex)
            answer_hit = bool(pattern.search(context_text))
        except re.error:
            answer_hit = False
        answer_latency_ms = (time.perf_counter() - answer_start) * 1000

        metrics["answer_hits"].append(answer_hit)
        metrics["retrieval_latencies"].append(retr_ms)
        metrics["full_latencies"].append(retr_ms + answer_latency_ms)

        # Log detailed decomposition info if enabled
        if log_decomposition and decomp_info:
            try:
                decomp_entry = {
                    "id": item.id,
                    "question": item.question,
                    "decomposition_strategy": decomp_strategy,
                    "subtask_count": decomp_info.subtask_count,
                    "subtasks": decomp_info.subtasks,
                    "llm_used": decomp_info.llm_used,
                    "subtask_hits": subtask_hit_details,
                    "retrieved_urls": retrieved_urls[:top_k],
                    "gold_urls": item.source_urls,
                    "recall@5": recall,
                    "answer_hit": answer_hit,
                }
                with decomp_log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(decomp_entry) + "\n")
            except Exception:
                # Log error but don't fail evaluation
                pass

        metrics["case_details"].append(
            {
                "id": item.id,
                "question": item.question,
                "decomposition_strategy": decomp_strategy,
                "retrieved_urls": retrieved_urls,
                "recall@5": recall,
                "answer_hit": answer_hit,
                "retrieval_ms": retr_ms,
                "full_ms": retr_ms + answer_latency_ms,
            }
        )

    # Build A/B comparison summary
    strategy_summary = {}
    for strategy in ["none", "heuristic", "llm"]:
        stats = metrics["decomposition_stats"][strategy]
        if stats["count"] > 0:
            strategy_summary[strategy] = {
                "count": stats["count"],
                "recall_at_5": round(stats["recall_sum"] / stats["count"], 3),
                "miss_count": len(stats["misses"]),
            }

    summary = {
        "cases": metrics["cases"],
        "recall_at_5": round(statistics.fmean(metrics["recall_at_5"]) if metrics["cases"] else 0.0, 3),
        "mrr_at_5": round(statistics.fmean(metrics["mrr_at_5"]) if metrics["cases"] else 0.0, 3),
        "answer_accuracy": round(sum(metrics["answer_hits"]) / metrics["cases"] if metrics["cases"] else 0.0, 3),
        "retrieval_latency_ms": {
            "p50": round(percentile(metrics["retrieval_latencies"], 0.5), 1),
            "p95": round(percentile(metrics["retrieval_latencies"], 0.95), 1),
        },
        "full_latency_ms": {
            "p50": round(percentile(metrics["full_latencies"], 0.5), 1),
            "p95": round(percentile(metrics["full_latencies"], 0.95), 1),
        },
        "decomposition_ab_summary": strategy_summary,
        "details": metrics["case_details"],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Clockify RAG retrieval and grounding.")
    parser.add_argument("--goldset", default="eval/goldset.csv", type=Path, help="Path to goldset CSV")
    parser.add_argument("--base-url", default="http://localhost:7000", help="Search API base URL")
    parser.add_argument("--api-token", default="change-me", help="API token for HTTP requests")
    parser.add_argument("--namespaces", default="clockify", help="Comma-separated namespaces for offline mode")
    parser.add_argument("--k", default=5, type=int, help="Top-k results to evaluate")
    parser.add_argument("--context-k", default=4, type=int, help="Chunks to pack into mock answer")
    parser.add_argument("--json", action="store_true", help="Dump summary as JSON only")
    parser.add_argument("--log-decomposition", action="store_true", help="Log query decomposition metadata to JSONL file")
    parser.add_argument("--decomposition-off", action="store_true", help="Disable query decomposition for A/B baseline comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goldset = load_goldset(args.goldset)

    retriever: object
    use_http = False
    if requests is not None and args.base_url:
        http_retriever = HttpRetriever(args.base_url, args.api_token, disable_decomposition=args.decomposition_off)
        if http_retriever.healthy():
            retriever = http_retriever
            use_http = True
        else:
            try:
                retriever = OfflineRetriever(
                    [ns.strip() for ns in args.namespaces.split(",") if ns.strip()],
                    disable_decomposition=args.decomposition_off,
                )
            except RuntimeError as exc:
                # FAISS indexes not available in CI environment
                if args.json:
                    error_summary = {
                        "error": "OfflineRetriever initialization failed",
                        "message": str(exc),
                        "details": "FAISS indexes not found. This is expected in CI where indexes are not persisted.",
                        "cases": 0,
                        "recall_at_5": 0.0,
                        "mrr_at_5": 0.0,
                        "answer_accuracy": 0.0,
                        "retrieval_latency_ms": {"p50": 0.0, "p95": 0.0},
                        "full_latency_ms": {"p50": 0.0, "p95": 0.0},
                        "decomposition_ab_summary": {},
                        "details": [],
                    }
                    print(json.dumps(error_summary, indent=2))
                    return
                else:
                    raise
    else:
        try:
            retriever = OfflineRetriever(
                [ns.strip() for ns in args.namespaces.split(",") if ns.strip()],
                disable_decomposition=args.decomposition_off,
            )
        except RuntimeError as exc:
            # FAISS indexes not available
            if args.json:
                error_summary = {
                    "error": "OfflineRetriever initialization failed",
                    "message": str(exc),
                    "details": "FAISS indexes not found. This is expected in CI where indexes are not persisted.",
                    "cases": 0,
                    "recall_at_5": 0.0,
                    "mrr_at_5": 0.0,
                    "answer_accuracy": 0.0,
                    "retrieval_latency_ms": {"p50": 0.0, "p95": 0.0},
                    "full_latency_ms": {"p50": 0.0, "p95": 0.0},
                    "decomposition_ab_summary": {},
                    "details": [],
                }
                print(json.dumps(error_summary, indent=2))
                return
            else:
                raise

    summary = evaluate(goldset, retriever, top_k=args.k, context_k=args.context_k, log_decomposition=args.log_decomposition)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    mode = "HTTP" if use_http else "offline"
    decomp_status = "DISABLED (A/B Baseline)" if args.decomposition_off else "ENABLED"
    print("\n" + "=" * 80)
    print(f"CLOCKIFY RAG EVAL ({mode.upper()} mode) | Decomposition: {decomp_status}")
    print("=" * 80)
    print(f"Cases: {summary['cases']}")
    print(f"Recall@5: {summary['recall_at_5']:.3f}")
    print(f"MRR@5: {summary['mrr_at_5']:.3f}")
    print(f"Answer accuracy: {summary['answer_accuracy']:.3f}")
    print(
        "Retrieval latency p50/p95 (ms): "
        f"{summary['retrieval_latency_ms']['p50']} / {summary['retrieval_latency_ms']['p95']}"
    )
    print(
        "Full pipeline latency p50/p95 (ms): "
        f"{summary['full_latency_ms']['p50']} / {summary['full_latency_ms']['p95']}"
    )

    # Print A/B comparison table
    if summary.get("decomposition_ab_summary"):
        print("\n" + "-" * 80)
        print("DECOMPOSITION A/B COMPARISON")
        print("-" * 80)
        print(f"{'Strategy':<15} {'Count':<8} {'Recall@5':<12} {'Misses':<8}")
        print("-" * 80)
        for strategy, stats in sorted(summary["decomposition_ab_summary"].items()):
            print(
                f"{strategy:<15} {stats['count']:<8} {stats['recall_at_5']:<12.3f} {stats['miss_count']:<8}"
            )
        print("-" * 80)

    print("-" * 80)
    for detail in summary["details"]:
        qid = detail.get("id")
        recall = detail.get("recall@5", 0.0)
        answer_hit = detail.get("answer_hit", False)
        decomp_strat = detail.get("decomposition_strategy", "none")
        print(
            f"[{qid}] R@5={recall:.2f} | answer={'✓' if answer_hit else '✗'} | "
            f"decomp={decomp_strat} | urls={detail.get('retrieved_urls', [])[:3]}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
