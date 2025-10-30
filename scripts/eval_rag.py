#!/usr/bin/env python3
"""Evaluate RAG retrieval quality using eval_set.json."""

import json
import time
import sys
from pathlib import Path

import requests

# Configuration
BASE_URL = "http://localhost:7000"
SEARCH_ENDPOINT = f"{BASE_URL}/search"
EVAL_SET_PATH = Path(__file__).parent.parent / "tests" / "eval_set.json"


def load_eval_set():
    """Load evaluation test cases."""
    if not EVAL_SET_PATH.exists():
        print(f"Eval set not found: {EVAL_SET_PATH}")
        return None

    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def check_api_available():
    """Check if API is available."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def evaluate_query(query: str, expected_keywords: list, k: int = 5) -> dict:
    """
    Evaluate a single query.

    Returns:
        dict with hit status and metrics
    """
    t0 = time.time()
    try:
        resp = requests.get(SEARCH_ENDPOINT, params={"q": query, "k": k}, timeout=10)
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            return {"hit": False, "latency_ms": latency_ms, "error": "API error"}

        data = resp.json()
        results = data.get("results", [])

        # Check if expected keywords appear in results
        combined_text = " ".join([r.get("text", "").lower() for r in results])
        hit = any(kw.lower() in combined_text for kw in expected_keywords)

        return {
            "hit": hit,
            "latency_ms": latency_ms,
            "num_results": len(results),
            "keywords_found": [kw for kw in expected_keywords if kw.lower() in combined_text],
        }
    except Exception as e:
        return {"hit": False, "latency_ms": int((time.time() - t0) * 1000), "error": str(e)}


def main():
    """Run evaluation."""
    print("Clockify RAG Evaluation")
    print("=" * 60)

    # Check API
    if not check_api_available():
        print("ERROR: API not available at", BASE_URL)
        print("Please run: make serve")
        sys.exit(1)

    print(f"API available at {BASE_URL}")

    # Load eval set
    eval_set = load_eval_set()
    if not eval_set:
        sys.exit(1)

    cases = eval_set.get("eval_cases", [])
    print(f"Loaded {len(cases)} evaluation cases")
    print()

    # Run evaluations
    results = {"cases": [], "summary": {}}
    latencies = []

    for i, case in enumerate(cases):
        query_id = case.get("id", f"case_{i}")
        query = case.get("query")
        keywords = case.get("expected_keywords", [])
        query_type = case.get("type", "other")

        print(f"[{i+1}/{len(cases)}] {query_type:15s} - {query[:50]:50s}", end=" ... ")

        # Evaluate at k=5
        result_5 = evaluate_query(query, keywords, k=5)
        # Evaluate at k=12
        result_12 = evaluate_query(query, keywords, k=12)

        hit_5 = result_5.get("hit", False)
        hit_12 = result_12.get("hit", False)
        latency = result_5.get("latency_ms", 0)

        latencies.append(latency)
        status = "PASS" if hit_5 else "FAIL"
        print(f"{status:6s} (hit@5:{hit_5}, hit@12:{hit_12}, {latency}ms)")

        results["cases"].append({
            "id": query_id,
            "query": query,
            "type": query_type,
            "hit_at_5": hit_5,
            "hit_at_12": hit_12,
            "latency_ms": latency,
            "keywords_found": result_5.get("keywords_found", []),
        })

    # Summary
    hit_at_5 = sum(1 for c in results["cases"] if c["hit_at_5"])
    hit_at_12 = sum(1 for c in results["cases"] if c["hit_at_12"])
    total = len(cases)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    results["summary"] = {
        "total_cases": total,
        "hit_at_5": hit_at_5,
        "hit_at_5_pct": round(100 * hit_at_5 / total, 1),
        "hit_at_12": hit_at_12,
        "hit_at_12_pct": round(100 * hit_at_12 / total, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 1),
    }

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total cases: {total}")
    print(f"  Hit@5:  {hit_at_5:3d}/{total} ({results['summary']['hit_at_5_pct']:5.1f}%)")
    print(f"  Hit@12: {hit_at_12:3d}/{total} ({results['summary']['hit_at_12_pct']:5.1f}%)")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  P99 latency: {results['summary']['p99_latency_ms']:.0f}ms")
    print()

    # Check against targets
    targets = eval_set.get("baseline_targets", {})
    target_hit5 = targets.get("hit_at_5_pct", 80)
    target_hit12 = targets.get("hit_at_12_pct", 95)

    if results["summary"]["hit_at_5_pct"] >= target_hit5:
        print(f"✓ Hit@5 meets target ({target_hit5}%)")
    else:
        print(f"✗ Hit@5 below target (got {results['summary']['hit_at_5_pct']}%, need {target_hit5}%)")

    if results["summary"]["hit_at_12_pct"] >= target_hit12:
        print(f"✓ Hit@12 meets target ({target_hit12}%)")
    else:
        print(f"✗ Hit@12 below target (got {results['summary']['hit_at_12_pct']}%, need {target_hit12}%)")

    # Output JSON report
    report_file = Path(__file__).parent.parent / "eval_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
