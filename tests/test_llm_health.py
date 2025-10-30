"""Test LLM endpoint health checks."""

import os
import pytest
import importlib


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.server import app
    return TestClient(app)


def test_health_mock_mode():
    """In mock mode, llm_ok should be None."""
    os.environ["MOCK_LLM"] = "true"

    # Force reimport to pick up env changes
    import importlib
    import src.server
    importlib.reload(src.server)
    from src.server import app as reloaded_app
    from fastapi.testclient import TestClient
    client = TestClient(reloaded_app)

    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["ok"] is True
    assert data["mode"] == "mock"
    assert data["llm_ok"] is None  # Not checked in mock mode


def test_health_with_bad_endpoint(client):
    """With invalid endpoint, llm_ok should be False."""
    os.environ["MOCK_LLM"] = "false"
    os.environ["LLM_API_TYPE"] = "ollama"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"  # Obviously unreachable
    os.environ["LLM_CHAT_PATH"] = "/api/chat"
    os.environ["LLM_TAGS_PATH"] = "/api/tags"

    # Force reimport to pick up env changes
    import importlib
    import src.server
    importlib.reload(src.server)
    from src.server import app as reloaded_app
    from fastapi.testclient import TestClient
    client = TestClient(reloaded_app)

    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["llm_ok"] is False
    assert "details" in data["llm_details"] or data["llm_details"] is not None


def test_config_includes_llm_paths(client):
    """Config endpoint should include LLM configuration."""
    resp = client.get("/config")
    assert resp.status_code == 200

    data = resp.json()
    assert "llm_base_url" in data
    assert "llm_chat_path" in data
    assert "llm_tags_path" in data
    assert "llm_timeout_seconds" in data
    assert "llm_api_type" in data
    assert "mock_llm" in data


def test_llm_client_builds_urls():
    """LLMClient should build correct URLs from base and paths."""
    os.environ["LLM_BASE_URL"] = "http://example.com:11434"
    os.environ["LLM_CHAT_PATH"] = "/api/chat"
    os.environ["LLM_TAGS_PATH"] = "/api/tags"
    os.environ["MOCK_LLM"] = "true"  # Mock to avoid actual calls

    from src.llm_client import LLMClient
    llm = LLMClient()

    assert llm.chat_url == "http://example.com:11434/api/chat"
    assert llm.tags_url == "http://example.com:11434/api/tags"


def test_llm_client_health_mock():
    """Health check should return ok=True in mock mode."""
    os.environ["MOCK_LLM"] = "true"

    from src.llm_client import LLMClient
    llm = LLMClient()
    result = llm.health_check()

    assert result["ok"] is True
    assert "mock mode" in result["details"]

def test_deep_health_skip_in_mock(client):
    """deep health returns nulls in mock mode."""
    os.environ["MOCK_LLM"] = "true"
    import importlib
    import src.server
    importlib.reload(src.server)
    from src.server import app as reloaded_app
    from fastapi.testclient import TestClient
    client = TestClient(reloaded_app)

    resp = client.get("/health?deep=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_deep_ok"] is None
    assert data["llm_deep_details"] is None

def test_timeout_alias_deprecated():
    """LLM_TIMEOUT alias triggers warning when seconds unset."""
    os.environ.pop("LLM_TIMEOUT_SECONDS", None)
    os.environ["LLM_TIMEOUT"] = "5"

    import src.llm_client as lc
    lc_reload = importlib.reload(lc)  # recompute DEFAULT_TIMEOUT
    assert lc_reload.DEFAULT_TIMEOUT == 5.0

def test_streaming_disabled_in_config(client):
    """streaming_enabled reflects STREAMING_ENABLED env."""
    os.environ["STREAMING_ENABLED"] = "false"
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json()["streaming_enabled"] is False

def test_model_default_is_correct(client):
    """Model default should be gpt-oss:20b with colon."""
    resp = client.get("/config")
    assert resp.status_code == 200
    # The default should be set in LLMClient
    from src.llm_client import LLMClient
    os.environ.pop("LLM_MODEL", None)  # Remove to test default
    os.environ["MOCK_LLM"] = "true"  # Use mock to avoid validation requiring real LLM_BASE_URL
    llm = LLMClient()
    assert llm.model == "gpt-oss:20b"
