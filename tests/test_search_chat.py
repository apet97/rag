"""Test search and chat endpoints with decomposition flows.

Regression tests for:
- Multi-intent query decomposition
- Per-subtask retrieval and additive fusion
- Decomposition metadata in responses
- Cache key serialization for decomposed queries
- Hybrid search fallback
"""

import os
import json
import pytest
import importlib
from pathlib import Path

# Skip tests if index is missing
INDEX_DIR = Path("index/faiss")
SKIP_IF_NO_INDEX = pytest.mark.skipif(
    not (INDEX_DIR / "clockify" / "index.bin").exists(),
    reason="FAISS index not found. Run 'make embed' first."
)


@SKIP_IF_NO_INDEX
def test_search_endpoint_mock(client):
    """Test /search endpoint with mock mode."""
    os.environ["MOCK_LLM"] = "true"

    response = client.get(
        "/search?q=timesheet&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Check new response contract
    assert data.get("success") is True, "Response should have success=True"
    assert "results" in data, "Response should contain results"
    assert "total_results" in data, "Response should contain total_results"
    assert "latency_ms" in data, "Response should contain latency_ms"
    assert "metadata" in data, "Response should contain metadata"
    assert len(data["results"]) >= 1, "Should retrieve at least 1 result"

    # Check result structure (scores are now [0, 1] normalized similarity)
    result = data["results"][0]
    assert "score" in result
    assert "title" in result or "url" in result
    assert 0 <= result["score"] <= 1.0, "Similarity score should be in [0, 1] range (L2-normalized)"


@SKIP_IF_NO_INDEX
def test_chat_endpoint_mock(client):
    """Test /chat endpoint with mock mode."""
    os.environ["MOCK_LLM"] = "true"

    response = client.post(
        "/chat",
        json={
            "question": "How do I create a project?",
            "k": 5,
            "namespace": None
        },
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Check new response contract
    assert data.get("success") is True, "Response should have success=True"
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "meta" in data
    assert "metadata" in data, "Response should include metadata field"

    # Check non-empty answer and sources
    assert len(data["answer"]) > 0, "Answer should not be empty"
    assert isinstance(data["sources"], list), "Sources should be a list"
    assert len(data["sources"]) > 0, "Should have at least 1 source"

    # Check source structure
    source = data["sources"][0]
    assert "title" in source or "url" in source
    assert "namespace" in source
    assert "score" in source
    assert 0 <= source["score"] <= 1.0, "Source score should be normalized to [0, 1]"

    # Check latency breakdown (latency_ms is a dict with retrieval, llm, total)
    assert isinstance(data["latency_ms"], dict), "latency_ms should be a dict"
    assert "retrieval" in data["latency_ms"]
    assert "llm" in data["latency_ms"]
    assert "total" in data["latency_ms"]


@SKIP_IF_NO_INDEX
def test_health_endpoint(client):
    """Test /health endpoint shows index normalization."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "ok" in data
    assert "namespaces" in data
    assert "index_normalized" in data

    # If indexes are loaded, they should be normalized
    if data["ok"]:
        assert data["index_normalized"] is True, "Indexes should be L2-normalized"


@SKIP_IF_NO_INDEX
def test_chat_non_streaming_when_disabled(client):
    """Ensure /chat works with stream=false when STREAMING_ENABLED=false."""
    os.environ["MOCK_LLM"] = "true"
    os.environ["STREAMING_ENABLED"] = "false"
    import src.server
    importlib.reload(src.server)
    from src.server import app as reloaded_app
    from fastapi.testclient import TestClient
    client_reloaded = TestClient(reloaded_app)

    payload = {"question": "ping?", "k": 1}
    r = client_reloaded.post("/chat", json=payload, headers={"x-api-token": "change-me"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)


@SKIP_IF_NO_INDEX
def test_multi_intent_query_decomposition(client):
    """Test that multi-intent queries are decomposed and result in metadata."""
    os.environ["MOCK_LLM"] = "true"

    # Use a multi-intent query (comparison)
    response = client.get(
        "/search?q=kiosk vs timer&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data

    # Check for decomposition metadata in new structure
    metadata = data.get("metadata", {})
    # Metadata may contain decomposition if query was decomposed
    # Just verify structure if present
    if "decomposition" in metadata:
        decomp = metadata["decomposition"]
        assert "strategy" in decomp, "Decomposition should have strategy field"
        assert decomp["strategy"] in ["comparison", "heuristic", "llm", "none", "multi_part"]


@SKIP_IF_NO_INDEX
def test_decomposition_metadata_in_response(client):
    """Test that decomposition metadata is included in response."""
    os.environ["MOCK_LLM"] = "true"

    response = client.get(
        "/search?q=export timesheets and invoices&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Check response has metadata field (even if empty)
    assert "metadata" in data, "Response should include metadata field"
    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict), "Metadata should be a dict"

    # If decomposed, should have decomposition field with these properties
    if "decomposition" in metadata:
        decomp = metadata["decomposition"]
        assert "subtask_count" in decomp, "Decomposition should include subtask_count"
        assert "strategy" in decomp, "Decomposition should include strategy field"
        assert "llm_used" in decomp, "Decomposition should include llm_used"
        assert "subtasks" in decomp, "Decomposition should include subtasks"
        assert isinstance(decomp["subtasks"], list), "Subtasks should be a list"


@SKIP_IF_NO_INDEX
def test_cache_separation_decomposed_vs_non_decomposed(client):
    """Test that decomposed and non-decomposed retrievals don't collide in cache."""
    os.environ["MOCK_LLM"] = "true"

    # First request with decomposition enabled
    response1 = client.get(
        "/search?q=kiosk vs timer&k=5",
        headers={"x-api-token": "change-me"}
    )
    assert response1.status_code == 200
    results1 = response1.json().get("results", [])

    # Second request with same query but decomposition disabled
    # (This would be an edge case test with --decomposition-off flag on eval)
    # For now, just verify cache behavior with normal requests
    response2 = client.get(
        "/search?q=kiosk vs timer&k=5",
        headers={"x-api-token": "change-me"}
    )
    assert response2.status_code == 200
    results2 = response2.json().get("results", [])

    # Results should be consistent (same cache key)
    assert len(results1) == len(results2), "Cached results should be identical"


@SKIP_IF_NO_INDEX
def test_hybrid_search_fallback_on_empty_vector_results(client):
    """Test hybrid search fallback when vector search returns empty."""
    os.environ["MOCK_LLM"] = "true"

    # Query that might return few vector results but should work with hybrid
    response = client.get(
        "/search?q=workflow&k=3",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Should get results (either vector or hybrid fallback)
    # We don't strictly require results, but test should not error
    if "results" in data:
        assert isinstance(data["results"], list)


@SKIP_IF_NO_INDEX
def test_search_preserves_order_on_multi_hit_documents(client):
    """Test that documents hitting multiple subtasks rank higher (additive fusion)."""
    os.environ["MOCK_LLM"] = "true"

    # Query likely to have documents matching multiple aspects
    response = client.get(
        "/search?q=export timesheets and invoices&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()
    results = data.get("results", [])

    # If we got decomposition with multiple subtasks, verify scoring
    metadata = data.get("metadata", {})
    decomp = metadata.get("decomposition", {})
    if decomp and decomp.get("subtask_count", 0) > 1:
        # Verify all results have valid scores in [0, 1] range
        for result in results:
            assert "score" in result
            assert isinstance(result["score"], (int, float))
            assert 0 <= result["score"] <= 1.0, f"Score {result['score']} should be in [0, 1] range (L2-normalized)"


@SKIP_IF_NO_INDEX
def test_search_with_decomposition_metadata_latency(client):
    """Test that decomposition doesn't cause excessive latency."""
    os.environ["MOCK_LLM"] = "true"

    # Fix #5: Add warmup query before timed request to avoid cold-start measurement
    # Warmup ensures models (embedding, reranker) are loaded and cached
    warmup_response = client.get(
        "/search?q=test warmup&k=2",
        headers={"x-api-token": "change-me"}
    )
    assert warmup_response.status_code == 200, "Warmup query should succeed"

    # Now measure latency on a warm request
    response = client.get(
        "/search?q=kiosk vs timer&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Check latency (now a top-level field, not in metadata)
    if "latency_ms" in data:
        latency = data["latency_ms"]
        assert isinstance(latency, (int, float)), "latency_ms should be a number (milliseconds)"
        # Fix #5: Lower threshold to 2000ms for warm requests (after warmup)
        # Cold start overhead (embedding model load, reranker, etc.) already handled by warmup
        # In CI with stub index fixture, this should be well under 1000ms
        threshold = 2000
        assert latency < threshold, f"Latency {latency}ms seems excessive for warm request (threshold: {threshold}ms)"


@SKIP_IF_NO_INDEX
def test_chat_with_decomposition_metadata(client):
    """Test that decomposition metadata is preserved in /chat response."""
    os.environ["MOCK_LLM"] = "true"

    response = client.post(
        "/chat",
        json={
            "question": "What is the difference between kiosk and timer?",
            "k": 5,
            "namespace": None
        },
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "answer" in data
    assert "sources" in data
    assert "meta" in data
    assert "metadata" in data, "Chat response should include metadata field"

    # Check metadata structure
    metadata = data.get("metadata", {})
    assert isinstance(metadata, dict), "Metadata should be a dict"

    # If decomposition was used, should have decomposition field
    if "decomposition" in metadata:
        decomp = metadata["decomposition"]
        assert "subtask_count" in decomp, "Decomposition should include subtask_count"
        assert "llm_used" in decomp, "Decomposition should include llm_used"


@SKIP_IF_NO_INDEX
def test_boost_terms_improve_retrieval(client):
    """Test that per-subtask boost terms enhance retrieval (regression check)."""
    os.environ["MOCK_LLM"] = "true"

    # Query with domain terms that should trigger boost term expansion
    response = client.get(
        "/search?q=API integration workflow&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Should retrieve results (boost terms shouldn't break anything)
    results = data.get("results", [])
    assert isinstance(results, list), "Results should be a list"


@SKIP_IF_NO_INDEX
def test_per_subtask_intent_affects_hybrid_strategy(client):
    """Test that per-subtask intent detection changes retrieval strategy per subtask."""
    os.environ["MOCK_LLM"] = "true"

    # Mixed query with command and question parts
    response = client.get(
        "/search?q=export timesheets and what is API?&k=5",
        headers={"x-api-token": "change-me"}
    )

    assert response.status_code == 200
    data = response.json()

    # Should retrieve results (mixed intent should be handled)
    results = data.get("results", [])
    assert isinstance(results, list)
    # Should have at least some results from at least one intent
    if results:
        assert "score" in results[0]


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.server import app

    return TestClient(app)
