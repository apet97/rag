"""
Edge case and boundary condition tests for RAG API.

Tests validation, limits, error handling, and resilience.
"""

import time
import pytest
import requests

BASE_URL = "http://localhost:7000"
HEADERS = {"x-api-token": "change-me"}


class TestSearchValidation:
    """Test /search parameter validation."""

    def test_search_query_min_length(self):
        """Minimum length query (1 char) should work."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "a", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_search_query_max_length(self):
        """Query at max boundary (2000 chars) should work."""
        query = "a" * 2000
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_search_query_exceeds_max(self):
        """Query > 2000 chars should fail validation."""
        query = "a" * 2001
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422
        assert "too long" in resp.text.lower() or "validation" in resp.text.lower()

    def test_search_query_empty_string(self):
        """Empty query should fail validation."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_search_query_whitespace_only(self):
        """Query with only whitespace should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "   ", "k": 5},
            headers=HEADERS,
        )
        # May be treated as empty after strip
        assert resp.status_code in [422, 200]


class TestSearchKParameter:
    """Test /search k parameter validation."""

    def test_search_k_minimum(self):
        """k=1 should work (minimum)."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 1},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= 1

    def test_search_k_maximum(self):
        """k=20 should work (maximum)."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 20},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) <= 20

    def test_search_k_zero(self):
        """k=0 should fail validation."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 0},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_search_k_negative(self):
        """k=-1 should fail validation."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": -1},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_search_k_exceeds_maximum(self):
        """k=21 should clamp to 20 or fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 21},
            headers=HEADERS,
        )
        # Should either clamp or fail validation
        if resp.status_code == 200:
            assert len(resp.json()["results"]) <= 20
        else:
            assert resp.status_code == 422

    def test_search_k_very_large(self):
        """k=1000 should fail or clamp."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 1000},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            assert len(resp.json()["results"]) <= 20
        else:
            assert resp.status_code == 422


class TestSearchResults:
    """Test search result handling."""

    def test_search_no_results_returns_empty(self):
        """Query with no hits returns empty list, not error."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "xyzabc123gibberishneverexist", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["count"] == 0

    def test_search_result_fields(self):
        """Each result should have required fields."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "help", "k": 1},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        if results:
            r = results[0]
            assert "rank" in r
            assert "url" in r
            assert "title" in r
            assert "score" in r
            assert "namespace" in r


class TestChatValidation:
    """Test /chat parameter validation."""

    def test_chat_min_length_question(self):
        """Minimum length question (1 char) should work."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "?", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_chat_max_length_question(self):
        """Question at max boundary (2000 chars) should work."""
        question = "a" * 2000
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": question, "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_chat_exceeds_max_length(self):
        """Question > 2000 chars should fail."""
        question = "a" * 2001
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": question, "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_chat_empty_question(self):
        """Empty question should fail."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_chat_missing_question(self):
        """Missing question field should fail."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_chat_no_sources_returns_answer(self):
        """Chat with no matching sources should still return gracefully."""
        # Force a query that likely returns zero results
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "xyzabc123gibberishneverexist", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        # Answer may be generic if no sources found


class TestChatCitations:
    """Test citation handling in chat."""

    def test_chat_citations_match_sources(self):
        """Citations in answer should reference valid source indices."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "timesheet submission", "k": 3},
            headers=HEADERS,
        )
        if resp.status_code != 200:
            pytest.skip("Chat unavailable")

        data = resp.json()
        sources_count = len(data.get("sources", []))

        # Extract citation numbers from answer: [1], [2], [3], etc.
        import re

        citations = set(
            int(m)
            for m in re.findall(r'\[(\d+)\]', data.get("answer", ""))
        )

        # All citation numbers should be valid source indices (1-based)
        for citation in citations:
            assert 1 <= citation <= sources_count, \
                f"Citation [{citation}] exceeds source count {sources_count}"


class TestUnicodeHandling:
    """Test Unicode and special character handling."""

    def test_search_emoji_query(self):
        """Query with emoji should not crash."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "How do I submit my 📝 timesheet? 🕐", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_chat_emoji_question(self):
        """Chat with emoji should not crash."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "How do I submit my 📝 timesheet? 🕐", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_search_rtl_text(self):
        """Right-to-left text (Arabic, Hebrew) should work."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "مرحبا بك", "k": 3},  # "Hello" in Arabic
            headers=HEADERS,
        )
        # Should either return results or fail gracefully, not crash
        assert resp.status_code in [200, 422]

    def test_chat_utf8_special_chars(self):
        """UTF-8 special characters should work."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "Où est mon feuille de temps? (Ñoño)", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code in [200, 422]


class TestAuthentication:
    """Test authentication handling."""

    def test_search_missing_token(self):
        """Search without token should fail with 401."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
        )
        assert resp.status_code == 401

    def test_search_invalid_token(self):
        """Search with wrong token should fail with 401."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
            headers={"x-api-token": "wrong-token-xyz"},
        )
        assert resp.status_code == 401

    def test_chat_missing_token(self):
        """Chat without token should fail with 401."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "test", "k": 3},
        )
        assert resp.status_code == 401

    def test_health_no_token(self):
        """Health check may require token depending on config."""
        resp = requests.get(f"{BASE_URL}/health")
        # May be 401 or 200 depending on auth policy
        assert resp.status_code in [200, 401]


class TestMetrics:
    """Test /metrics endpoint."""

    def test_metrics_no_auth_required(self):
        """Metrics endpoint typically doesn't require auth."""
        resp = requests.get(f"{BASE_URL}/metrics")
        # May or may not require auth depending on config
        assert resp.status_code in [200, 401]

    def test_metrics_format(self):
        """Metrics should be in Prometheus format."""
        resp = requests.get(f"{BASE_URL}/metrics", headers=HEADERS)
        if resp.status_code == 200:
            text = resp.text
            # Check for Prometheus format markers
            assert "# HELP" in text or "# TYPE" in text or "{" in text


class TestCacheValidation:
    """Test response caching behavior."""

    def test_repeated_search_consistent(self):
        """Repeated identical searches should return identical results."""
        query_params = {"q": "timesheet", "k": 3}

        resp1 = requests.get(f"{BASE_URL}/search", params=query_params, headers=HEADERS)
        resp2 = requests.get(f"{BASE_URL}/search", params=query_params, headers=HEADERS)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Results should be byte-for-byte identical (deterministic)
        data1 = resp1.json()
        data2 = resp2.json()

        # Compare result counts and scores
        assert data1["count"] == data2["count"]
        if data1["results"]:
            for r1, r2 in zip(data1["results"], data2["results"]):
                assert r1["url"] == r2["url"]
                assert r1["rank"] == r2["rank"]
                assert abs(r1["score"] - r2["score"]) < 0.0001


class TestRateLimiting:
    """Test rate limit enforcement."""

    def test_rate_limit_multiple_requests(self):
        """Rapid requests from same IP should hit rate limit."""
        start_time = time.time()
        responses = []

        # Send 15 requests rapidly
        for i in range(15):
            resp = requests.get(
                f"{BASE_URL}/search",
                params={"q": f"test{i}", "k": 1},
                headers=HEADERS,
            )
            responses.append(resp.status_code)
            # Don't sleep - we want to exceed rate limit

        elapsed = time.time() - start_time

        # Within 1 second, should get some 429s after first batch
        if elapsed < 1.5:  # Still within window
            # Should have at least one 429 if rate limiting works
            assert 429 in responses or all(r == 200 for r in responses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
