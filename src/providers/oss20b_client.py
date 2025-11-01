#!/usr/bin/env python3
"""
oss20b Provider Adapter - VPN-only LLM with offline-safe mocks.

This provider targets the company's internal Ollama server at http://10.127.0.192:11434
with the gpt-oss:20b model. It provides:

1. VPN-only access (BASE env var, default http://10.127.0.192:11434)
2. /api/tags for model detection
3. /api/chat for inference
4. No API key required
5. Timeouts, non-stream and stream support
6. Offline mock mode for CI (MOCK_LLM=true)

Environment variables:
    LLM_BASE_URL: Base URL (default: http://10.127.0.192:11434)
    LLM_MODEL: Model name (default: gpt-oss:20b)
    LLM_TIMEOUT_SECONDS: Request timeout (default: 30)
    LLM_RETRIES: Retry attempts (default: 3)
    MOCK_LLM: Enable mock mode (default: false)
    STREAMING_ENABLED: Enable streaming (default: false)

Usage:
    from src.providers.oss20b_client import OSS20BClient

    # Production mode (requires VPN)
    client = OSS20BClient()
    response = client.chat([{"role": "user", "content": "ping"}])

    # Mock mode (offline-safe for CI)
    client = OSS20BClient(mock=True)
    response = client.chat([{"role": "user", "content": "test"}])
"""
from __future__ import annotations

import os
import time
import json
import random
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin

import httpx
from loguru import logger


class OSS20BClient:
    """Provider adapter for internal oss20b model (VPN-only Ollama endpoint)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        mock: Optional[bool] = None,
    ) -> None:
        """Initialize oss20b client.

        Args:
            base_url: Ollama base URL (default: env LLM_BASE_URL or http://10.127.0.192:11434)
            model: Model name (default: env LLM_MODEL or gpt-oss:20b)
            timeout: Request timeout in seconds (default: env LLM_TIMEOUT_SECONDS or 30)
            retries: Retry attempts (default: env LLM_RETRIES or 3)
            mock: Enable mock mode (default: env MOCK_LLM or false)
        """
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-oss:20b")
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.retries = retries if retries is not None else int(os.getenv("LLM_RETRIES", "3"))
        self.mock = mock if mock is not None else os.getenv("MOCK_LLM", "false").lower() == "true"
        self.streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower() == "true"

        # Build endpoint URLs
        self.tags_url = urljoin(self.base_url + "/", "api/tags")
        self.chat_url = urljoin(self.base_url + "/", "api/chat")

        # HTTP client with production-grade config
        self._client: Optional[httpx.Client] = None

        logger.info(f"OSS20B client initialized: mock={self.mock}, base={self.base_url}, model={self.model}")

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client with proper timeout/connection pooling."""
        if self._client is None:
            timeout_config = httpx.Timeout(
                connect=5.0,
                read=self.timeout,
                write=10.0,
                pool=5.0
            )
            limits = httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
            self._client = httpx.Client(
                timeout=timeout_config,
                verify=False,  # Company internal SSL may be self-signed
                limits=limits,
                follow_redirects=True
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health via /api/tags.

        Returns:
            {"ok": bool, "details": str, "models": list}
        """
        if self.mock:
            return {
                "ok": True,
                "details": "mock mode",
                "models": ["gpt-oss:20b", "gpt-oss:13b"]
            }

        try:
            resp = self._get_client().get(self.tags_url)

            if resp.status_code == 404:
                return {"ok": False, "details": f"404 on {self.tags_url}", "models": []}

            if resp.status_code == 403:
                return {"ok": False, "details": f"403 forbidden - check VPN/auth", "models": []}

            if resp.status_code != 200:
                return {"ok": False, "details": f"HTTP {resp.status_code}", "models": []}

            # Parse model list
            data = resp.json()
            models = []
            if isinstance(data, dict) and "models" in data:
                models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
            elif isinstance(data, list):
                models = [m.get("name", "") for m in data if isinstance(m, dict)]

            models = [m for m in models if m]
            return {"ok": True, "details": "OK", "models": models}

        except httpx.TimeoutException:
            return {"ok": False, "details": "Timeout - check VPN connection", "models": []}
        except httpx.ConnectError:
            return {"ok": False, "details": "Connection failed - check VPN", "models": []}
        except Exception as e:
            return {"ok": False, "details": f"Error: {e}", "models": []}

    def detect_model(self) -> str:
        """Auto-detect available model from /api/tags.

        Returns configured model if available, otherwise first gpt-oss* model,
        or first available model as fallback.

        Returns:
            Model name to use
        """
        if self.mock:
            return self.model

        health = self.health_check()
        if not health["ok"]:
            logger.warning(f"Model detection failed: {health['details']}")
            return self.model

        models = health["models"]
        if not models:
            return self.model

        # Prefer configured model
        if self.model in models:
            return self.model

        # Prefer gpt-oss* models
        for m in models:
            if m.lower().startswith("gpt-oss"):
                logger.info(f"Auto-detected model: {m}")
                return m

        # Fallback to first available
        logger.info(f"Auto-detected model: {models[0]}")
        return models[0]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 800,
        stream: bool = False,
    ) -> str:
        """Generate chat completion.

        Args:
            messages: Chat messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Max tokens to generate (ignored by Ollama)
            stream: Enable streaming (requires STREAMING_ENABLED=true)

        Returns:
            Generated text

        Raises:
            RuntimeError: On request failure
        """
        if self.mock:
            return self._mock_response(messages)

        # Auto-detect model if needed
        if not hasattr(self, "_model_detected"):
            self.model = self.detect_model()
            self._model_detected = True

        # Handle streaming
        if stream and self.streaming_enabled:
            return self._chat_streaming(messages, temperature)

        # Non-streaming request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }

        return self._post_with_retry(self.chat_url, payload)

    def _chat_streaming(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Handle streaming chat request.

        Args:
            messages: Chat messages
            temperature: Sampling temperature

        Returns:
            Complete generated text
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }

        chunks = []
        try:
            with self._get_client().stream("POST", self.chat_url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        msg = obj.get("message", {})
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            if content:
                                chunks.append(content)
                        if obj.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            return "".join(chunks)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise RuntimeError(f"Streaming chat failed: {e}") from e

    def _post_with_retry(self, url: str, payload: Dict[str, Any]) -> str:
        """POST with exponential backoff retry.

        Retries only on transient errors (timeout, connection, 5xx).
        Fails fast on permanent errors (4xx, invalid JSON).

        Args:
            url: Request URL
            payload: JSON payload

        Returns:
            Response text

        Raises:
            RuntimeError: On permanent failure or after all retries
        """
        delay = 0.75
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = self._get_client().post(url, json=payload)

                # Retry on 5xx
                if 500 <= resp.status_code < 600:
                    raise httpx.HTTPStatusError(
                        f"Server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                resp.raise_for_status()

                # Parse Ollama response
                data = resp.json()
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                    if isinstance(content, str):
                        return content.strip()

                # Fallback to raw text
                return resp.text

            except (httpx.TimeoutException, httpx.ConnectError, OSError) as e:
                # Transient network errors - retry
                last_error = e
                if attempt == self.retries:
                    break

                jitter = random.uniform(0.0, 0.1 * delay)
                sleep_time = delay + jitter
                logger.debug(f"Attempt {attempt}/{self.retries} failed: {type(e).__name__}, retrying in {sleep_time:.2f}s")
                time.sleep(sleep_time)
                delay *= 2

            except httpx.HTTPStatusError as e:
                # Retry 5xx, fail fast on 4xx
                if 500 <= e.response.status_code < 600:
                    last_error = e
                    if attempt == self.retries:
                        break
                    jitter = random.uniform(0.0, 0.1 * delay)
                    sleep_time = delay + jitter
                    logger.debug(f"Attempt {attempt}/{self.retries} failed: HTTP {e.response.status_code}, retrying in {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    delay *= 2
                else:
                    # 4xx - permanent error
                    raise RuntimeError(f"HTTP {e.response.status_code}: {e}") from e

            except Exception as e:
                # JSON decode or other permanent errors - fail fast
                raise RuntimeError(f"Request failed: {e}") from e

        # All retries exhausted
        raise RuntimeError(f"Request failed after {self.retries} attempts: {last_error}") from last_error

    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response for offline testing.

        Args:
            messages: Chat messages

        Returns:
            Mock response text
        """
        # Extract last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Simple mock based on query patterns
        user_lower = user_msg.lower()

        if "ping" in user_lower or "hello" in user_lower:
            return "pong"

        if "create" in user_lower and "project" in user_lower:
            return """To create a project in Clockify:
1. Navigate to Projects tab
2. Click "Create new project"
3. Enter project details
4. Save

[Mock response - offline mode]"""

        if "report" in user_lower or "timesheet" in user_lower:
            return """To generate a timesheet report:
1. Go to Reports section
2. Select Timesheet type
3. Choose date range
4. Export as needed

[Mock response - offline mode]"""

        # Generic fallback
        return f"Mock response for: {user_msg[:100]}...\n\n[Offline mock - VPN required for real LLM]"


# Module-level singleton
_client: Optional[OSS20BClient] = None


def get_oss20b_client(mock: Optional[bool] = None) -> OSS20BClient:
    """Get or create singleton oss20b client.

    Args:
        mock: Override mock mode (default: env MOCK_LLM)

    Returns:
        OSS20BClient instance
    """
    global _client
    if _client is None:
        _client = OSS20BClient(mock=mock)
    return _client


def close_oss20b_client() -> None:
    """Close singleton client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
