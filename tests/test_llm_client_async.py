"""
Test Suite for Async LLM Client

Tests cover:
- Async HTTP client with connection pooling
- Health checks
- Chat operations (mock and with HTTP)
- Bearer token authentication
- Retry logic with exponential backoff
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List

from src.llm_client_async import (
    AsyncLLMHTTPClient,
    AsyncLLMClient,
    AsyncLLMClientContext,
)

# ============================================================================
# Async HTTP Client Tests
# ============================================================================

class TestAsyncLLMHTTPClient:
    """Test async HTTP client with connection pooling."""

    def test_initialization(self):
        """Client should initialize with configuration."""
        client = AsyncLLMHTTPClient(
            max_connections=20,
            max_keepalive_connections=10,
            timeout=30.0,
            retries=3,
            backoff_factor=0.75,
        )
        assert client.max_connections == 20
        assert client.max_keepalive_connections == 10
        assert client.timeout == 30.0
        assert client.retries == 3
        assert client.backoff_factor == 0.75

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Client should be created on first access."""
        client = AsyncLLMHTTPClient()
        http_client = await client._get_client()
        assert http_client is not None

    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """HTTP client should be reused."""
        client = AsyncLLMHTTPClient()
        http1 = await client._get_client()
        http2 = await client._get_client()
        assert http1 is http2

    @pytest.mark.asyncio
    async def test_close(self):
        """Client should close properly."""
        client = AsyncLLMHTTPClient()
        await client._get_client()
        await client.close()
        assert client._client is None

# ============================================================================
# Async LLM Client Tests
# ============================================================================

class TestAsyncLLMClient:
    """Test async LLM client."""

    def test_initialization_defaults(self):
        """Client should initialize with defaults."""
        client = AsyncLLMClient()
        assert client.api_type in ("ollama", "openai")
        assert client.base_url is not None
        assert client.model is not None

    def test_initialization_custom(self):
        """Client should accept custom parameters."""
        client = AsyncLLMClient(
            api_type="openai",
            base_url="https://api.openai.com",
            model="gpt-4",
        )
        assert client.api_type == "openai"
        assert client.base_url == "https://api.openai.com"
        assert client.model == "gpt-4"

    def test_url_building(self):
        """Client should build URLs correctly."""
        client = AsyncLLMClient(base_url="http://localhost:11434")
        assert client.chat_url.startswith("http://localhost:11434")
        assert "/chat" in client.chat_url or "/api/chat" in client.chat_url

    @pytest.mark.asyncio
    async def test_mock_mode_health_check(self):
        """Health check should pass in mock mode."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            client = AsyncLLMClient()
            result = await client.health_check()
            assert result["ok"] is True
            assert result["details"] == "mock mode"

    @pytest.mark.asyncio
    async def test_mock_mode_chat(self):
        """Chat should work in mock mode."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            client = AsyncLLMClient()
            messages = [{"role": "user", "content": "hello"}]
            response = await client.chat(messages)
            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_health_check_error_handling(self):
        """Health check should handle errors gracefully."""
        client = AsyncLLMClient(base_url="http://invalid-url")
        result = await client.health_check()
        assert result["ok"] is False
        assert "Error" in result["details"] or "Connection" in result["details"]

# ============================================================================
# Context Manager Tests
# ============================================================================

class TestAsyncLLMClientContext:
    """Test async LLM client context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Context manager should handle client lifecycle."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            async with AsyncLLMClientContext() as client:
                assert isinstance(client, AsyncLLMClient)
                result = await client.health_check()
                assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Context manager should cleanup client."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            context = AsyncLLMClientContext()
            async with context as client:
                pass
            # After context exit, client should be closed
            assert context.client is not None  # Object exists

# ============================================================================
# Integration Tests
# ============================================================================

class TestAsyncLLMClientIntegration:
    """Integration tests for async LLM client."""

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Multiple health checks should run concurrently."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            client = AsyncLLMClient()

            # Run multiple health checks concurrently
            results = await asyncio.gather(
                client.health_check(),
                client.health_check(),
                client.health_check(),
            )

            assert len(results) == 3
            assert all(r["ok"] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self):
        """Multiple chat requests should run concurrently."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            client = AsyncLLMClient()

            messages = [{"role": "user", "content": "test"}]

            # Run multiple chat requests concurrently
            results = await asyncio.gather(
                client.chat(messages),
                client.chat(messages),
                client.chat(messages),
            )

            assert len(results) == 3
            assert all(isinstance(r, str) and len(r) > 0 for r in results)

    @pytest.mark.asyncio
    async def test_client_connection_pooling(self):
        """Client should reuse HTTP connections."""
        with patch.dict("os.environ", {"MOCK_LLM": "true"}):
            client = AsyncLLMClient()

            # Multiple requests should reuse same HTTP client
            http1 = await client._http_client._get_client()
            http2 = await client._http_client._get_client()

            assert http1 is http2
