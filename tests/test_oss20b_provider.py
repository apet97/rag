"""Tests for oss20b provider adapter with offline mocks."""

import os
import pytest
from src.providers.oss20b_client import OSS20BClient, get_oss20b_client


@pytest.fixture
def mock_client():
    """OSS20B client in mock mode (offline-safe)."""
    return OSS20BClient(mock=True)


@pytest.fixture
def env_clean():
    """Clean up env after test."""
    yield
    # Cleanup singleton
    import src.providers.oss20b_client as oss_mod
    oss_mod._client = None


def test_client_initialization_defaults(env_clean):
    """Client should use sensible defaults from env."""
    # Set env vars
    os.environ["LLM_BASE_URL"] = "http://10.127.0.192:11434"
    os.environ["LLM_MODEL"] = "gpt-oss:20b"
    os.environ["MOCK_LLM"] = "false"

    client = OSS20BClient()

    assert client.base_url == "http://10.127.0.192:11434"
    assert client.model == "gpt-oss:20b"
    assert client.mock is False
    assert client.timeout == 30.0
    assert client.retries == 3


def test_client_initialization_custom():
    """Client should accept custom parameters."""
    client = OSS20BClient(
        base_url="http://custom.local:8080",
        model="custom-model",
        timeout=60,
        retries=5,
        mock=True
    )

    assert client.base_url == "http://custom.local:8080"
    assert client.model == "custom-model"
    assert client.timeout == 60
    assert client.retries == 5
    assert client.mock is True


def test_health_check_mock_mode(mock_client):
    """Health check in mock mode should return ok."""
    result = mock_client.health_check()

    assert result["ok"] is True
    assert result["details"] == "mock mode"
    assert "gpt-oss:20b" in result["models"]


def test_health_check_unreachable_endpoint():
    """Health check should handle unreachable endpoints gracefully."""
    client = OSS20BClient(
        base_url="http://127.0.0.1:9",  # Obviously unreachable
        mock=False,
        timeout=1
    )

    result = client.health_check()

    assert result["ok"] is False
    assert "models" in result
    assert len(result["models"]) == 0


def test_chat_mock_mode_ping(mock_client):
    """Mock mode should handle ping request."""
    messages = [{"role": "user", "content": "ping"}]
    response = mock_client.chat(messages)

    assert isinstance(response, str)
    assert response == "pong"


def test_chat_mock_mode_project_query(mock_client):
    """Mock mode should handle project creation query."""
    messages = [{"role": "user", "content": "How do I create a project in Clockify?"}]
    response = mock_client.chat(messages)

    assert isinstance(response, str)
    assert "project" in response.lower()
    assert "clockify" in response.lower()


def test_chat_mock_mode_generic(mock_client):
    """Mock mode should handle generic queries with fallback."""
    messages = [{"role": "user", "content": "What is the meaning of life?"}]
    response = mock_client.chat(messages)

    assert isinstance(response, str)
    assert "mock" in response.lower() or "offline" in response.lower()


def test_detect_model_mock_mode(mock_client):
    """Model detection in mock mode should return configured model."""
    model = mock_client.detect_model()
    assert model == "gpt-oss:20b"


def test_singleton_client_reuse(env_clean):
    """Singleton pattern should reuse client instance."""
    os.environ["MOCK_LLM"] = "true"

    client1 = get_oss20b_client()
    client2 = get_oss20b_client()

    assert client1 is client2


def test_endpoint_url_construction():
    """URLs should be constructed correctly."""
    client = OSS20BClient(base_url="http://10.127.0.192:11434", mock=True)

    assert client.tags_url == "http://10.127.0.192:11434/api/tags"
    assert client.chat_url == "http://10.127.0.192:11434/api/chat"


def test_endpoint_url_trailing_slash():
    """Base URL with trailing slash should be handled."""
    client = OSS20BClient(base_url="http://example.com/", mock=True)

    assert client.tags_url == "http://example.com/api/tags"
    assert client.chat_url == "http://example.com/api/chat"


def test_mock_mode_env_var(env_clean):
    """MOCK_LLM env var should control mock mode."""
    os.environ["MOCK_LLM"] = "true"
    client = OSS20BClient()
    assert client.mock is True

    os.environ["MOCK_LLM"] = "false"
    client = OSS20BClient()
    assert client.mock is False


def test_streaming_env_var():
    """STREAMING_ENABLED env var should be read."""
    os.environ["STREAMING_ENABLED"] = "true"
    client = OSS20BClient(mock=True)
    assert client.streaming_enabled is True

    os.environ["STREAMING_ENABLED"] = "false"
    client = OSS20BClient(mock=True)
    assert client.streaming_enabled is False


def test_timeout_env_var():
    """LLM_TIMEOUT_SECONDS env var should be used."""
    os.environ["LLM_TIMEOUT_SECONDS"] = "45"
    client = OSS20BClient(mock=True)
    assert client.timeout == 45.0


def test_retries_env_var():
    """LLM_RETRIES env var should be used."""
    os.environ["LLM_RETRIES"] = "5"
    client = OSS20BClient(mock=True)
    assert client.retries == 5


def test_close_client(mock_client):
    """Client should close HTTP client cleanly."""
    # Force client creation
    mock_client._get_client()
    assert mock_client._client is not None

    mock_client.close()
    assert mock_client._client is None


@pytest.mark.skipif(
    os.getenv("VPN_AVAILABLE", "false").lower() != "true",
    reason="Requires VPN connection to internal LLM"
)
def test_health_check_real_vpn():
    """Integration test with real VPN endpoint (skipped by default)."""
    client = OSS20BClient(mock=False)
    result = client.health_check()

    assert result["ok"] is True
    assert len(result["models"]) > 0


@pytest.mark.skipif(
    os.getenv("VPN_AVAILABLE", "false").lower() != "true",
    reason="Requires VPN connection to internal LLM"
)
def test_chat_real_vpn():
    """Integration test with real VPN endpoint (skipped by default)."""
    client = OSS20BClient(mock=False)
    messages = [{"role": "user", "content": "Say 'OK'"}]
    response = client.chat(messages)

    assert isinstance(response, str)
    assert len(response) > 0
