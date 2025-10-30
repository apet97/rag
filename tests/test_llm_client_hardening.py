"""Test LLM client hardening: retries, config validation, logging hygiene."""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock, call
from src.llm_client import (
    LLMClient,
    _validate_config,
    _sanitize_url,
    _redact_token,
    _cap_response,
)


class TestConfigValidation:
    """Test configuration validation on startup."""

    def test_validate_config_success_with_ollama_mock(self):
        """Config validation should pass with valid mock Ollama config."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_TIMEOUT_SECONDS"] = "30"
        os.environ["LLM_RETRIES"] = "3"
        os.environ["LLM_BACKOFF"] = "0.75"

        # Should not raise
        _validate_config()

    def test_validate_config_success_with_live_ollama(self):
        """Config validation should pass with valid live Ollama config."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_CHAT_PATH"] = "/api/chat"
        os.environ["LLM_TAGS_PATH"] = "/api/tags"

        _validate_config()

    def test_validate_config_success_with_openai(self):
        """Config validation should pass with valid OpenAI config."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "openai"
        os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["LLM_CHAT_PATH"] = "/chat/completions"

        _validate_config()

    def test_validate_config_fails_invalid_api_type(self):
        """Config validation should fail with invalid API type."""
        os.environ["LLM_API_TYPE"] = "invalid_type"
        os.environ["MOCK_LLM"] = "true"

        with pytest.raises(ValueError, match="LLM_API_TYPE must be"):
            _validate_config()

    def test_validate_config_fails_missing_base_url(self):
        """Config validation should fail when base URL is missing in live mode."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ.pop("LLM_BASE_URL", None)

        with pytest.raises(ValueError, match="LLM_BASE_URL is required"):
            _validate_config()

    def test_validate_config_fails_invalid_url_scheme(self):
        """Config validation should fail with non-http(s) URLs."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "ftp://example.com"

        with pytest.raises(ValueError, match="must be http:// or https://"):
            _validate_config()

    def test_validate_config_fails_path_without_slash(self):
        """Config validation should fail when paths don't start with /."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_CHAT_PATH"] = "api/chat"  # Missing leading /

        with pytest.raises(ValueError, match="must start with '/'"):
            _validate_config()

    def test_validate_config_fails_invalid_timeout(self):
        """Config validation should fail with non-positive timeout."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_TIMEOUT_SECONDS"] = "0"

        with pytest.raises(ValueError, match="must be positive"):
            _validate_config()


class TestLoggingHygiene:
    """Test logging sanitization functions."""

    def test_sanitize_url_removes_token_param(self):
        """_sanitize_url should mask token query parameters."""
        url = "http://example.com/api?token=secret123&other=value"
        sanitized = _sanitize_url(url)

        assert "secret123" not in sanitized
        assert "token=***" in sanitized
        assert "other=value" in sanitized

    def test_sanitize_url_removes_api_key_param(self):
        """_sanitize_url should mask api_key query parameters."""
        url = "https://api.openai.com/v1/chat?api_key=sk-1234567890"
        sanitized = _sanitize_url(url)

        assert "sk-1234567890" not in sanitized
        assert "api_key=***" in sanitized

    def test_sanitize_url_preserves_normal_params(self):
        """_sanitize_url should preserve non-sensitive parameters."""
        url = "http://example.com/api?model=gpt-4&temperature=0.5"
        sanitized = _sanitize_url(url)

        assert "model=gpt-4" in sanitized
        assert "temperature=0.5" in sanitized

    def test_sanitize_url_with_no_query(self):
        """_sanitize_url should handle URLs without query strings."""
        url = "http://example.com/api/chat"
        sanitized = _sanitize_url(url)

        assert sanitized == url

    def test_redact_token_from_log(self):
        """_redact_token should mask Bearer tokens."""
        text = 'Authorization: Bearer sk-1234567890abcdef'
        redacted = _redact_token(text)

        assert "sk-1234567890abcdef" not in redacted
        assert "Bearer ***" in redacted

    def test_redact_token_case_insensitive(self):
        """_redact_token should work case-insensitively."""
        text = "authorization: bearer SECRET_TOKEN_VALUE"
        redacted = _redact_token(text)

        assert "SECRET_TOKEN_VALUE" not in redacted
        assert "Bearer ***" in redacted

    def test_redact_token_multiple_tokens(self):
        """_redact_token should mask multiple Bearer tokens."""
        text = "First: Bearer token1 and Second: Bearer token2"
        redacted = _redact_token(text)

        assert "token1" not in redacted
        assert "token2" not in redacted
        assert redacted.count("Bearer ***") == 2

    def test_cap_response_short(self):
        """_cap_response should not truncate short responses."""
        text = "Short response"
        capped = _cap_response(text, max_len=100)

        assert capped == text
        assert "..." not in capped

    def test_cap_response_long(self):
        """_cap_response should truncate long responses."""
        text = "x" * 500
        capped = _cap_response(text, max_len=200)

        assert len(capped) < len(text)
        assert "..." in capped
        assert "300 more bytes" in capped


class TestRetryLogic:
    """Test retry logic with mocked HTTPX."""

    @patch("src.llm_client._get_http_client")
    def test_retry_on_timeout(self, mock_get_client):
        """LLMClient should retry on timeout exceptions."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "3"

        # Mock client that raises timeout on first 2 attempts, succeeds on 3rd
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            MagicMock(status_code=200, text='{"message": {"content": "ok"}}'),
        ]

        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])

        assert result == "ok"
        assert mock_client.post.call_count == 3

    @patch("src.llm_client._get_http_client")
    def test_no_retry_on_4xx(self, mock_get_client):
        """LLMClient should NOT retry on 4xx errors."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "3"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock 401 Unauthorized
        mock_response = MagicMock(status_code=401)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        llm = LLMClient()
        with pytest.raises(RuntimeError, match="failed after"):
            llm.chat([{"role": "user", "content": "test"}])

        # Should only attempt once (no retries)
        assert mock_client.post.call_count == 1

    @patch("src.llm_client._get_http_client")
    def test_retry_on_5xx(self, mock_get_client):
        """LLMClient should retry on 5xx errors."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "2"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call returns 503, second succeeds
        mock_response_503 = MagicMock(status_code=503)
        mock_response_ok = MagicMock(
            status_code=200,
            text='{"message": {"content": "recovered"}}'
        )

        mock_client.post.side_effect = [
            mock_response_503,
            mock_response_ok,
        ]

        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])

        assert result == "recovered"
        assert mock_client.post.call_count == 2

    @patch("src.llm_client._get_http_client")
    def test_retry_with_jitter(self, mock_get_client):
        """Retry logic should apply exponential backoff with jitter."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "2"
        os.environ["LLM_BACKOFF"] = "0.1"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Fail twice then succeed
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            MagicMock(status_code=200, text='{"message": {"content": "ok"}}'),
        ]

        import time
        start = time.time()
        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])
        elapsed = time.time() - start

        # Should have slept due to backoff (at least 0.1s for first attempt)
        # With jitter and retry, this should be reasonable
        assert result == "ok"
        assert elapsed > 0.05  # Allow some tolerance
        assert mock_client.post.call_count == 3


class TestLiveEndpoints:
    """Test /live and /ready endpoints."""

    def test_live_endpoint_always_up(self):
        """GET /live should always return 200 with status: alive."""
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/live")

        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_ready_endpoint_mock_mode(self):
        """GET /ready in mock mode should return 200."""
        os.environ["MOCK_LLM"] = "true"

        import importlib
        import src.server
        importlib.reload(src.server)
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/ready")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_endpoint_with_bad_llm(self):
        """GET /ready should return 503 if LLM is unhealthy."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"  # Unreachable
        os.environ["LLM_API_TYPE"] = "ollama"

        import importlib
        import src.server
        importlib.reload(src.server)
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/ready")

        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
