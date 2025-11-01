from __future__ import annotations

import os
import time
import json
import random
import re
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin, urlparse, parse_qs

import httpx
from loguru import logger
from src.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
from src.config import CONFIG

def _env_bool(val: Optional[str]) -> Optional[bool]:
    """Parse environment boolean, return None if ambiguous."""
    if val is None:
        return None
    v = val.strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    return None

# Back-compat: LLM_TIMEOUT deprecated in favor of LLM_TIMEOUT_SECONDS
_timeout_seconds = os.getenv("LLM_TIMEOUT_SECONDS")
_timeout_alias = os.getenv("LLM_TIMEOUT")
if _timeout_alias and not _timeout_seconds:
    logger.warning("LLM_TIMEOUT is deprecated. Use LLM_TIMEOUT_SECONDS instead.")
    DEFAULT_TIMEOUT = float(_timeout_alias)
else:
    DEFAULT_TIMEOUT = float(_timeout_seconds or "30")

RETRIES = int(os.getenv("LLM_RETRIES", "3"))
BACKOFF = float(os.getenv("LLM_BACKOFF", "0.75"))
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "false").lower() == "true"

def _validate_config() -> None:
    """Validate LLM configuration on startup. Raises ValueError if invalid."""
    # FIX MAJOR #3: Align validation with runtime default
    # Runtime defaults to VPN LLM if LLM_BASE_URL not set
    # Validation should use the same default to avoid startup crashes
    base_url = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434").strip()
    chat_path = os.getenv("LLM_CHAT_PATH", "").strip()
    tags_path = os.getenv("LLM_TAGS_PATH", "").strip()
    api_type = os.getenv("LLM_API_TYPE", "ollama").strip().lower()
    mock_llm = os.getenv("MOCK_LLM", "false").lower() == "true"

    # Validate API type
    if api_type not in ("ollama", "openai"):
        raise ValueError(f"LLM_API_TYPE must be 'ollama' or 'openai', got: {api_type}")

    # If not in mock mode, validate base URL format is http(s)
    # Note: We no longer require it to be explicitly set since we have a safe default
    if not mock_llm:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"LLM_BASE_URL must be http:// or https://, got: {base_url}")

    # Validate paths start with /
    for name, path in [("LLM_CHAT_PATH", chat_path), ("LLM_TAGS_PATH", tags_path)]:
        if path and not path.startswith("/"):
            raise ValueError(f"{name} must start with '/', got: {path}")

    # Validate timeouts are positive
    if DEFAULT_TIMEOUT <= 0:
        raise ValueError(f"LLM_TIMEOUT_SECONDS must be positive, got: {DEFAULT_TIMEOUT}")
    if RETRIES < 0:
        raise ValueError(f"LLM_RETRIES must be non-negative, got: {RETRIES}")
    if BACKOFF <= 0:
        raise ValueError(f"LLM_BACKOFF must be positive, got: {BACKOFF}")

    logger.info("LLM config validation passed")

def _sanitize_url(url: str) -> str:
    """Remove or mask sensitive query parameters from URL for logging."""
    try:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        # Parse query params
        params = parse_qs(parsed.query, keep_blank_values=True)
        # Mask sensitive params
        for sensitive_key in ("token", "key", "api_key", "password", "secret"):
            if sensitive_key in params:
                params[sensitive_key] = ["***"]
        # Reconstruct query string
        sanitized_qs = "&".join(f"{k}={v[0]}" for k, v in params.items())
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sanitized_qs}" if sanitized_qs else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to sanitize URL: {e}")
        return url

def _redact_token(text: str) -> str:
    """Redact Bearer token values from log text."""
    return re.sub(r'Bearer\s+[^\s]+', 'Bearer ***', text, flags=re.IGNORECASE)

def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Create a redacted copy of headers for logging. Masks Authorization headers."""
    redacted = {}
    for key, value in headers.items():
        if key.lower() == "authorization":
            redacted[key] = "Bearer ***"
        else:
            redacted[key] = value
    return redacted

def _cap_response(text: str, max_len: int = 200) -> str:
    """Cap response body length for logging."""
    if len(text) > max_len:
        return text[:max_len] + f"... ({len(text)-max_len} more bytes)"
    return text

def compute_answerability_score(answer: str, context: str) -> tuple[bool, float]:
    """
    PHASE 5: Compute answerability score to prevent hallucination.

    Uses Jaccard overlap between answer and context tokens.
    If score < threshold (0.25), indicates answer may not be grounded in context.

    Args:
        answer: Generated answer text
        context: Concatenated context from retrieval

    Returns:
        (is_answerable, score) - is_answerable=True if score >= 0.25
    """
    if not answer or not context:
        return False, 0.0

    # Tokenize (simple lowercase word split)
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    # Remove common stop words to avoid inflating overlap
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "will", "would", "should", "could", "may"
    }
    answer_tokens = {t for t in answer_tokens if t not in stop_words and len(t) > 2}
    context_tokens = {t for t in context_tokens if t not in stop_words and len(t) > 2}

    if not answer_tokens or not context_tokens:
        return False, 0.0

    # Jaccard similarity: |intersection| / |union|
    intersection = len(answer_tokens & context_tokens)
    union = len(answer_tokens | context_tokens)
    jaccard_score = intersection / union if union > 0 else 0.0

    # Threshold from CONFIG (default 18% overlap) - configurable via ANSWERABILITY_THRESHOLD env var
    is_answerable = jaccard_score >= CONFIG.ANSWERABILITY_THRESHOLD

    logger.debug(f"Answerability: {jaccard_score:.3f} (answer_tokens={len(answer_tokens)}, "
                f"context_tokens={len(context_tokens)}, intersection={intersection})")

    return is_answerable, jaccard_score

# Module-level HTTP client (reused across instances, thread-safe singleton)
HTTP_CLIENT: Optional[httpx.Client] = None
_http_client_lock = threading.Lock()

def _get_http_client() -> httpx.Client:
    """Get or create module-level HTTP client with production-grade config.

    Uses double-check locking pattern for thread-safe initialization.
    """
    global HTTP_CLIENT

    # First check (no lock - fast path)
    if HTTP_CLIENT is not None:
        return HTTP_CLIENT

    # Second check with lock (slow path - only on first access)
    with _http_client_lock:
        # Double-check pattern: another thread may have initialized while waiting
        if HTTP_CLIENT is None:
            base_url = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434").strip()
            verify_env = _env_bool(os.getenv("LLM_VERIFY_SSL"))
            # Auto-detect SSL verification: default to True for https://, False for http://
            if verify_env is not None:
                verify = verify_env
            else:
                verify = base_url.startswith("https://")

            # Production-grade timeout and connection pooling
            timeout = httpx.Timeout(connect=5.0, read=DEFAULT_TIMEOUT, write=10.0, pool=5.0)
            limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

            HTTP_CLIENT = httpx.Client(
                timeout=timeout,
                verify=verify,
                limits=limits,
                follow_redirects=True
            )

    return HTTP_CLIENT

def close_http_client() -> None:
    """Called by FastAPI on shutdown."""
    global HTTP_CLIENT
    try:
        if HTTP_CLIENT is not None:
            HTTP_CLIENT.close()
    finally:
        HTTP_CLIENT = None

class LLMClient:
    def __init__(self) -> None:
        # Validate configuration early (on first instantiation)
        _validate_config()

        self.api_type = os.getenv("LLM_API_TYPE", "ollama").strip().lower()
        # Default to VPN LLM; use LLM_BASE_URL env to override
        self.base_url = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434").strip()
        self.chat_path = os.getenv("LLM_CHAT_PATH", "/api/chat").strip()
        self.tags_path = os.getenv("LLM_TAGS_PATH", "/api/tags").strip()
        self.model = os.getenv("LLM_MODEL", "gpt-oss:20b").strip()
        self.mock = os.getenv("MOCK_LLM", "false").lower() == "true"

        # Build resolved URLs
        self.chat_url = self._build_url(self.chat_path)
        self.tags_url = self._build_url(self.tags_path)

        # Initialize circuit breaker for LLM calls
        # Moderate config: 3 failures, 30s timeout, 2 successes in half-open (user preference)
        breaker_config = CircuitBreakerConfig(
            name=f"llm_{self.model}",
            failure_threshold=int(os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "3")),
            recovery_timeout_seconds=float(os.getenv("LLM_CIRCUIT_BREAKER_TIMEOUT", "30")),
            success_threshold=int(os.getenv("LLM_CIRCUIT_BREAKER_SUCCESS", "2"))
        )
        self.circuit_breaker = get_circuit_breaker(f"llm_{self.model}", breaker_config)
        logger.info(f"LLM circuit breaker initialized: {self.model} (threshold=3, timeout=30s)")

    def _build_url(self, path: str) -> str:
        """Build full URL from base and path using urljoin."""
        base = self.base_url.rstrip("/")
        path_part = path.lstrip("/")
        return urljoin(base + "/", path_part)

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
        """POST with retries, exponential backoff with jitter, and auth. Returns response text or raises.

        FIX CRITICAL #6: Only retries on transient errors (timeout, connection, 5xx).
        Permanent errors (4xx, JSON decode, etc.) fail fast without retries.

        PRODUCTION FIX: All HTTP calls wrapped in circuit breaker for fault tolerance.
        """
        def _make_request() -> str:
            """Make HTTP POST request (wrapped by circuit breaker)."""
            headers_to_use = {"Content-Type": "application/json", **(headers or {})}

            # Add Bearer token if configured
            bearer_token = os.getenv("LLM_BEARER_TOKEN", "").strip()
            if bearer_token:
                headers_to_use["Authorization"] = f"Bearer {bearer_token}"

            delay = BACKOFF
            last_error: Optional[Exception] = None
            sanitized_url = _sanitize_url(url)

            for attempt in range(1, RETRIES + 1):
                try:
                    resp = _get_http_client().post(url, json=payload, headers=headers_to_use)
                    # Treat 5xx as retryable; skip retry logic for 4xx
                    if 500 <= resp.status_code < 600:
                        raise httpx.HTTPStatusError(
                            f"server {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    return str(resp.text)
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    # Retry on network timeouts and connection errors (transient)
                    last_error = e
                    if attempt == RETRIES:
                        break
                    # Jittered exponential backoff
                    jitter = random.uniform(0.0, 0.1 * delay)
                    sleep_time = delay + jitter
                    logger.debug(f"LLM POST attempt {attempt}/{RETRIES} failed to {sanitized_url}: {type(e).__name__}; backing off {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    delay *= 2
                except httpx.HTTPStatusError as e:
                    # For 5xx errors (already caught above, but keeping for clarity)
                    # Other status errors (4xx) are permanent - fail fast
                    if 500 <= e.response.status_code < 600:
                        last_error = e
                        if attempt == RETRIES:
                            break
                        # Jittered exponential backoff for 5xx
                        jitter = random.uniform(0.0, 0.1 * delay)
                        sleep_time = delay + jitter
                        logger.debug(f"LLM POST attempt {attempt}/{RETRIES} failed to {sanitized_url}: HTTP {e.response.status_code}; backing off {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                        delay *= 2
                    else:
                        # 4xx errors are permanent - fail immediately
                        error_msg = _redact_token(str(e))
                        raise RuntimeError(f"LLM POST failed with permanent error (HTTP {e.response.status_code}): {error_msg}") from e
                except (IOError, OSError) as e:
                    # Network-level errors - transient, retry
                    last_error = e
                    if attempt == RETRIES:
                        break
                    jitter = random.uniform(0.0, 0.1 * delay)
                    sleep_time = delay + jitter
                    logger.debug(f"LLM POST attempt {attempt}/{RETRIES} failed to {sanitized_url}: {type(e).__name__}; backing off {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                    delay *= 2
                except Exception as e:
                    # All other exceptions (JSON decode, etc.) are permanent - fail immediately
                    error_msg = _redact_token(str(e))
                    raise RuntimeError(f"LLM POST failed with permanent error: {error_msg}") from e

            # Ensure error message doesn't leak any tokens
            error_msg = str(last_error) if last_error else "Unknown error"
            error_msg = _redact_token(error_msg)
            raise RuntimeError(f"LLM POST failed after {RETRIES} attempts: {error_msg}") from last_error

        # Execute HTTP request through circuit breaker for fault tolerance
        return self.circuit_breaker.call(_make_request)

    def health_check(self) -> Dict[str, Any]:
        """Check LLM endpoint health. Returns {'ok': bool, 'details': str}.

        GET call is wrapped through circuit breaker for consistent fault tolerance.
        """
        if self.mock:
            return {"ok": True, "details": "mock mode"}

        def _check_health() -> Dict[str, Any]:
            """Check health through GET request (wrapped by circuit breaker)."""
            try:
                # Use module-level HTTP client
                resp = _get_http_client().get(self.tags_url)
                if resp.status_code == 404:
                    return {
                        "ok": False,
                        "details": f"404 on {self.tags_url} - endpoint not exposed (UI-only URL?)",
                    }
                if resp.status_code == 403:
                    return {
                        "ok": False,
                        "details": f"403 on {self.tags_url} - forbidden (check auth, VPN, firewall)",
                    }
                if resp.status_code != 200:
                    return {
                        "ok": False,
                        "details": f"HTTP {resp.status_code} on {self.tags_url}",
                    }

                # Verify JSON and contains models/tags
                data = resp.json()
                if isinstance(data, dict):
                    if "models" in data or "tags" in data:
                        return {"ok": True, "details": f"OK: {self.api_type} at {self.base_url}"}
                elif isinstance(data, list) and len(data) > 0:
                    return {"ok": True, "details": f"OK: {self.api_type} at {self.base_url}"}

                return {"ok": False, "details": f"Unexpected response from {self.tags_url}"}

            except Exception as e:
                return {"ok": False, "details": f"Error contacting {self.tags_url}: {str(e)}"}

        # Execute health check through circuit breaker for consistent fault tolerance
        try:
            return self.circuit_breaker.call(_check_health)
        except Exception as e:
            # Circuit breaker is open - fast fail
            logger.warning(f"Health check failed due to circuit breaker: {e}")
            return {"ok": False, "details": f"LLM circuit breaker is OPEN - service unavailable"}

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 800, temperature: float = 0.2, stream: bool = False) -> str:
        if self.mock:
            # Fabricate concise, grounded response for offline dev
            ctx = "\n".join(m["content"] for m in messages if m["role"] == "user")[:1200]
            return f"{ctx.splitlines()[-1]}\n\n[1]\n\nSources:\n[1] See provided context."

        # Wrap LLM call with circuit breaker protection
        def _chat_protected():
            if self.api_type == "ollama":
                # Check if streaming is requested and enabled
                if stream and STREAMING_ENABLED:
                    payload = {"model": self.model, "messages": messages, "stream": True}
                    chunks = []
                    try:
                        with _get_http_client().stream("POST", self.chat_url, json=payload) as resp:
                            for line in resp.iter_lines():
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    msg = obj.get("message", {})
                                    if isinstance(msg, dict):
                                        part = msg.get("content", "")
                                        if isinstance(part, str):
                                            chunks.append(part)
                                    if obj.get("done"):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        return "".join(chunks)
                    except Exception as e:
                        # Streaming failed - re-raise to avoid duplicate request
                        logger.error(f"Streaming request failed: {e}")
                        raise

                # Non-streaming (default)
                payload = {"model": self.model, "messages": messages, "stream": False}
                text = self._post_json(self.chat_url, payload)
                # Parse Ollama response: {"message": {"role":"assistant","content":"..."}}
                try:
                    data = json.loads(text)
                    if "message" in data and isinstance(data["message"], dict):
                        return data["message"].get("content", "").strip()
                except json.JSONDecodeError:
                    pass
                return text

            elif self.api_type == "openai":
                payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": False}
                headers = {}
                key = os.getenv("OPENAI_API_KEY", "").strip()
                if key:
                    headers["Authorization"] = f"Bearer {key}"
                text = self._post_json(self.chat_url, payload, headers=headers)
                # Parse OpenAI response: {"choices":[{"message":{"content":"..."}}]}
                try:
                    data = json.loads(text)
                    if "choices" in data and data["choices"]:
                        return data["choices"][0]["message"]["content"].strip()
                except json.JSONDecodeError:
                    pass
                return text
            else:
                raise RuntimeError(f"Unsupported LLM_API_TYPE: {self.api_type}")

        # Execute with circuit breaker protection
        from typing import cast
        return cast(str, self.circuit_breaker.call(_chat_protected))
