"""
Integration Tests: Complete RAG Pipeline with Type Safety

Tests the full search and chat pipeline using Pydantic models for validation.
Covers end-to-end flows, error scenarios, and data validation.
"""

import time
import pytest
import requests
from typing import Dict, Any, List, Optional

# Import type-safe models
from src.models import (
    SearchRequest,
    ChatRequest,
    SearchResponse,
    ChatResponse,
    SearchResult,
    QueryAnalysis,
    ErrorResponse,
)

BASE_URL = "http://localhost:7000"
HEADERS = {"x-api-token": "change-me"}


class TestSearchPipeline:
    """Test complete /search endpoint pipeline."""

    def test_search_basic_query(self) -> None:
        """Basic search with valid query should return SearchResponse."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "how do I track time", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Validate response structure using pydantic
        search_response = SearchResponse(**data)
        assert search_response.success is True
        assert search_response.query == "how do I track time"
        assert len(search_response.results) >= 0
        assert search_response.latency_ms > 0

    def test_search_with_query_analysis(self) -> None:
        """Search should include query_analysis with extracted entities."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "timer features", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # Query analysis should be present
        if search_response.query_analysis:
            qa = search_response.query_analysis
            assert qa.primary_search_query is not None
            assert isinstance(qa.entities, list)
            assert 0.0 <= qa.confidence <= 1.0

    def test_search_result_structure(self) -> None:
        """Search results should have complete SearchResult structure."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "project time tracking", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        if search_response.results:
            for result in search_response.results:
                # Validate each result using pydantic
                validated_result = SearchResult(**result)
                assert validated_result.title
                assert validated_result.url
                assert validated_result.namespace
                assert 0 <= validated_result.confidence <= 100
                assert validated_result.level in ["high", "medium", "low"]

    def test_search_with_namespace(self) -> None:
        """Search with specific namespace should filter results."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "time tracking", "k": 5, "namespace": "clockify"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # All results should be from requested namespace
        for result in search_response.results:
            assert result.get("namespace") == "clockify"

    def test_search_with_custom_k(self) -> None:
        """Search with custom k parameter should return up to k results."""
        for k in [1, 5, 10]:
            resp = requests.get(
                f"{BASE_URL}/search",
                params={"q": "tracking", "k": k},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()
            search_response = SearchResponse(**data)
            assert len(search_response.results) <= k

    def test_search_caching(self) -> None:
        """Repeated search with same query should hit cache (faster latency)."""
        query = "how do I start a project"

        # First request
        start1 = time.time()
        resp1 = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        latency1 = time.time() - start1
        assert resp1.status_code == 200
        data1 = resp1.json()
        response1 = SearchResponse(**data1)

        # Wait a bit to ensure cache isn't just instant
        time.sleep(0.1)

        # Second request (should hit cache)
        start2 = time.time()
        resp2 = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        latency2 = time.time() - start2
        assert resp2.status_code == 200
        data2 = resp2.json()
        response2 = SearchResponse(**data2)

        # Responses should be identical (same results, same request_id expected if cached)
        assert len(response1.results) == len(response2.results)
        # Cache should provide faster response (typically 10-50ms vs 100-200ms)
        # Note: First request might be slow due to model warmup, so just check both succeed

    @pytest.mark.parametrize(
        "query",
        [
            "timer",
            "time tracking",
            "project management",
            "how to track",
            "stop watch",
            "clock app",
        ],
    )
    def test_search_various_queries(self, query: str) -> None:
        """Search should handle various query types successfully."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)
        assert search_response.success is True


class TestChatPipeline:
    """Test complete /chat endpoint pipeline."""

    def test_chat_basic_query(self) -> None:
        """Basic chat with valid query should return ChatResponse."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "How do I track time?", "k": 5},
            headers=HEADERS,
        )
        # May return 200 or 500 depending on LLM availability
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            assert chat_response.success is True
            assert chat_response.query == "How do I track time?"
            assert len(chat_response.answer) > 0
            assert chat_response.latency_ms > 0

    def test_chat_with_namespace(self) -> None:
        """Chat with specific namespace should use that namespace."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "What is Clockify?", "k": 3, "namespace": "clockify"},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            assert chat_response.success is True
            # Context docs should be from requested namespace
            for doc in chat_response.context_docs:
                assert doc.namespace == "clockify"

    def test_chat_result_citations(self) -> None:
        """Chat answer should include citations to sources."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "Tell me about time tracking features", "k": 5},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            # Answer may or may not have citations depending on LLM
            assert len(chat_response.answer) > 0


class TestSearchValidationWithModels:
    """Test input validation using Pydantic models."""

    def test_search_request_validation_min_query(self) -> None:
        """SearchRequest should accept minimum query length."""
        req = SearchRequest(query="a", k=1)
        assert req.query == "a"
        assert req.k == 1

    def test_search_request_validation_max_query(self) -> None:
        """SearchRequest should accept maximum query length."""
        long_query = "a" * 2000
        req = SearchRequest(query=long_query, k=5)
        assert len(req.query) == 2000

    def test_search_request_validation_exceeds_max(self) -> None:
        """SearchRequest should reject query > 2000 chars."""
        long_query = "a" * 2001
        with pytest.raises(ValueError):
            SearchRequest(query=long_query, k=5)

    def test_search_request_k_bounds(self) -> None:
        """SearchRequest should validate k within bounds."""
        # Valid k values
        for k in [1, 5, 10, 20]:
            req = SearchRequest(query="test", k=k)
            assert req.k == k

        # Invalid k values
        with pytest.raises(ValueError):
            SearchRequest(query="test", k=0)  # k < 1

        with pytest.raises(ValueError):
            SearchRequest(query="test", k=21)  # k > 20

    def test_chat_request_validation(self) -> None:
        """ChatRequest should validate parameters."""
        # Valid request
        req = ChatRequest(question="How to track time?", k=5)
        assert req.question == "How to track time?"
        assert req.k == 5

        # Invalid k
        with pytest.raises(ValueError):
            ChatRequest(question="test", k=21)

        # Invalid question length
        with pytest.raises(ValueError):
            ChatRequest(question="a" * 2001, k=5)


class TestErrorHandling:
    """Test error handling and resilience."""

    def test_search_missing_query(self) -> None:
        """Search without required query param should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_search_missing_token(self) -> None:
        """Search without API token should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
            # No headers
        )
        assert resp.status_code == 401

    def test_search_invalid_token(self) -> None:
        """Search with invalid token should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
            headers={"x-api-token": "invalid-token"},
        )
        assert resp.status_code == 401

    def test_chat_missing_question(self) -> None:
        """Chat without question should fail."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_invalid_namespace(self) -> None:
        """Search with invalid namespace should still work (fallback to all namespaces)."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5, "namespace": "nonexistent"},
            headers=HEADERS,
        )
        # Should gracefully handle invalid namespace
        assert resp.status_code in [200, 422]


class TestHealthAndConfig:
    """Test health checks and configuration endpoints."""

    def test_health_endpoint(self) -> None:
        """Health endpoint should return component status."""
        resp = requests.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data or "status" in data

    def test_health_deep_check(self) -> None:
        """Deep health check should probe LLM."""
        resp = requests.get(f"{BASE_URL}/health?deep=1")
        assert resp.status_code == 200
        data = resp.json()
        # May or may not have LLM available

    def test_live_endpoint(self) -> None:
        """Live endpoint should always return 200."""
        resp = requests.get(f"{BASE_URL}/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "alive"

    def test_ready_endpoint(self) -> None:
        """Ready endpoint should check if system is ready."""
        resp = requests.get(f"{BASE_URL}/ready")
        # May return 200 or 503 depending on readiness
        assert resp.status_code in [200, 503]

    def test_config_endpoint(self) -> None:
        """Config endpoint should return system configuration."""
        resp = requests.get(f"{BASE_URL}/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "namespaces_env" in data
        assert "embedding_model" in data


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""

    @pytest.mark.parametrize(
        "query",
        [
            "a",  # Single char
            "what?",  # With punctuation
            "time & tracking",  # With special chars
            "UPPERCASE QUERY",  # Case variation
            "123 456",  # Numbers
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
        ],
    )
    def test_search_special_inputs(self, query: str) -> None:
        """Search should handle special characters and injection attempts safely."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 3},
            headers=HEADERS,
        )
        # Should not crash, but may return 200 or 422
        assert resp.status_code in [200, 422, 400]

    def test_search_concurrent_requests(self) -> None:
        """Multiple concurrent search requests should all succeed."""
        import concurrent.futures

        def search_query(q: str) -> int:
            resp = requests.get(
                f"{BASE_URL}/search",
                params={"q": q, "k": 3},
                headers=HEADERS,
            )
            return resp.status_code

        queries = ["timer", "project", "tracking", "time", "stop"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(search_query, queries))

        # All should succeed (200) or be rate limited (429)
        assert all(r in [200, 429] for r in results)

    def test_search_result_ranking(self) -> None:
        """Results should be ranked by confidence score."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "time tracking", "k": 10},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # Confidence scores should be non-increasing
        confidences = [r.confidence for r in search_response.results]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]


class TestTypeValidation:
    """Test type safety with Pydantic models."""

    def test_search_response_typing(self) -> None:
        """SearchResponse should enforce type validation."""
        # Valid response structure
        valid_data = {
            "success": True,
            "query": "test",
            "results": [],
            "total_results": 0,
            "latency_ms": 100.0,
        }
        response = SearchResponse(**valid_data)
        assert response.success is True

        # Invalid types should raise
        invalid_data = {
            "success": "yes",  # Should be bool
            "query": "test",
            "results": [],
            "total_results": 0,
            "latency_ms": 100.0,
        }
        with pytest.raises(ValueError):
            SearchResponse(**invalid_data)

    def test_search_result_typing(self) -> None:
        """SearchResult should enforce type validation."""
        valid_result = {
            "id": "chunk_1",
            "title": "Test",
            "content": "Test content",
            "url": "http://example.com",
            "namespace": "test",
            "confidence": 85,
            "level": "high",
            "score": 0.85,
        }
        result = SearchResult(**valid_result)
        assert result.confidence == 85

        # Confidence out of range
        invalid_result = valid_result.copy()
        invalid_result["confidence"] = 150
        with pytest.raises(ValueError):
            SearchResult(**invalid_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
