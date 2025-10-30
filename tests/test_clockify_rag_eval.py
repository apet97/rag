"""Evaluation tests for Clockify RAG recall and precision."""
import pytest
import os
import json
from typing import Any


# Eval cases: (query, expected_terms_in_result)
# These are representative queries with expected content
EVAL_CASES = [
    ("How do I submit my weekly timesheet?", ["submit", "timesheet"]),
    ("Set billable rates per workspace member", ["billable rate", "rate", "member"]),
    ("Enable time rounding to 15 minutes", ["rounding", "time", "round"]),
    ("What is SSO?", ["sso", "single", "sign"]),
    ("How do I approve timesheets as a manager?", ["approve", "timesheet", "manager"]),
    ("What is a project budget?", ["project", "budget"]),
    ("How to enable time tracking?", ["time", "track"]),
    ("What are user roles?", ["role", "user", "permission"]),
]


def test_eval_cases_structure():
    """Verify eval cases are well-formed."""
    for q, terms in EVAL_CASES:
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(terms, list) and len(terms) > 0
        for t in terms:
            assert isinstance(t, str) and len(t) > 0


def get_search_results(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Query the /search endpoint and return results."""
    import requests

    api_host = os.getenv("API_HOST", "localhost")
    api_port = int(os.getenv("API_PORT", "7000"))
    base_url = f"http://{api_host}:{api_port}"

    try:
        response = requests.get(
            f"{base_url}/search",
            params={"q": query, "k": k},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"Error querying /search: {e}")
        return []


def hit_at_k(results: list[dict[str, Any]], expected_terms: list[str], k: int = 5) -> bool:
    """
    Check if at least one of the expected terms appears in top-k results.

    A hit is when ANY expected term is found (case-insensitive) in the text.
    """
    results_text = " ".join([r.get("text", "").lower() for r in results[:k]])
    for term in expected_terms:
        if term.lower() in results_text:
            return True
    return False


@pytest.mark.skipif(
    os.getenv("SKIP_API_EVAL") == "true",
    reason="Requires running API server; set SKIP_API_EVAL=false to run"
)
class TestRetrievalEval:
    """Retrieval evaluation tests that require a running API."""

    def test_search_endpoint_available(self):
        """Verify the search endpoint is available."""
        import requests
        api_host = os.getenv("API_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", "7000"))
        base_url = f"http://{api_host}:{api_port}"

        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data.get("ok") is not None
        except Exception as e:
            pytest.skip(f"API server not available: {e}")

    @pytest.mark.parametrize("query,expected_terms", EVAL_CASES)
    def test_hit_at_k_5(self, query: str, expected_terms: list[str]):
        """Test that at least one expected term appears in top-5 results."""
        results = get_search_results(query, k=5)
        assert len(results) > 0, f"No results for query: {query}"

        hit = hit_at_k(results, expected_terms, k=5)
        if not hit:
            # Print debug info
            text_sample = " ".join([r.get("text", "")[:100] for r in results[:2]])
            print(f"\nQuery: {query}")
            print(f"Expected terms: {expected_terms}")
            print(f"Results text sample: {text_sample}")

        assert hit, f"None of {expected_terms} found in top-5 for: {query}"

    @pytest.mark.parametrize("query,expected_terms", EVAL_CASES)
    def test_hit_at_k_12(self, query: str, expected_terms: list[str]):
        """Test that at least one expected term appears in top-12 results."""
        results = get_search_results(query, k=12)
        assert len(results) > 0, f"No results for query: {query}"

        hit = hit_at_k(results, expected_terms, k=12)
        assert hit, f"None of {expected_terms} found in top-12 for: {query}"


def compute_eval_metrics() -> dict[str, Any]:
    """
    Compute comprehensive eval metrics across all test cases.

    Returns dict with hit@5, hit@12, coverage, etc.
    """
    metrics = {
        "total_cases": len(EVAL_CASES),
        "hit_at_5": 0,
        "hit_at_12": 0,
        "cases": [],
    }

    for query, expected_terms in EVAL_CASES:
        results_5 = get_search_results(query, k=5)
        results_12 = get_search_results(query, k=12)

        hit_5 = hit_at_k(results_5, expected_terms, k=5)
        hit_12 = hit_at_k(results_12, expected_terms, k=12)

        metrics["hit_at_5"] += 1 if hit_5 else 0
        metrics["hit_at_12"] += 1 if hit_12 else 0

        metrics["cases"].append({
            "query": query,
            "expected_terms": expected_terms,
            "hit_at_5": hit_5,
            "hit_at_12": hit_12,
        })

    # Compute percentages
    metrics["hit_at_5_pct"] = round(100 * metrics["hit_at_5"] / metrics["total_cases"], 1)
    metrics["hit_at_12_pct"] = round(100 * metrics["hit_at_12"] / metrics["total_cases"], 1)

    return metrics


@pytest.mark.skipif(
    os.getenv("SKIP_API_EVAL") == "true",
    reason="Requires running API server; set SKIP_API_EVAL=false to run"
)
def test_eval_report(capsys):
    """Generate and print comprehensive eval report."""
    import requests

    # Check API availability
    api_host = os.getenv("API_HOST", "localhost")
    api_port = int(os.getenv("API_PORT", "7000"))
    base_url = f"http://{api_host}:{api_port}"

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"API server not available: {e}")

    metrics = compute_eval_metrics()

    # Print report
    print("\n" + "="*60)
    print("CLOCKIFY RAG RETRIEVAL EVAL REPORT")
    print("="*60)
    print(f"Total test cases: {metrics['total_cases']}")
    print(f"Hit@5:  {metrics['hit_at_5']}/{metrics['total_cases']} ({metrics['hit_at_5_pct']}%)")
    print(f"Hit@12: {metrics['hit_at_12']}/{metrics['total_cases']} ({metrics['hit_at_12_pct']}%)")
    print("-"*60)

    for case in metrics["cases"]:
        status_5 = "✓" if case["hit_at_5"] else "✗"
        status_12 = "✓" if case["hit_at_12"] else "✗"
        print(f"{status_5} {status_12}  {case['query'][:40]:40s} {case['expected_terms']}")

    print("="*60)
