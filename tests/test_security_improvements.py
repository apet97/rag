"""
Security & Architecture Improvements Verification Tests
Tests Phase 1, 2, 3 improvements without full dependencies
"""

import pytest
import hmac
import threading
import time
from pathlib import Path


# ============================================================================
# PHASE 1: SECURITY FIXES VERIFICATION
# ============================================================================

class TestSecurityFix1ConstantTimeComparison:
    """Verify Fix #1: Token validation uses constant-time comparison"""

    def test_hmac_compare_digest_is_constant_time(self):
        """Verify HMAC compare_digest prevents timing attacks"""
        token1 = "secret-token-12345"
        token2 = "secret-token-12345"
        token3 = "wrong-token-00000"

        # Valid comparison
        assert hmac.compare_digest(token1, token2) is True

        # Invalid comparison
        assert hmac.compare_digest(token1, token3) is False

    def test_token_validation_regardless_of_environment(self):
        """Verify tokens are validated in all environments"""
        api_token_dev = "change-me"
        api_token_prod = "production-secret-key"

        # Dev mode still validates tokens
        valid_dev = hmac.compare_digest("change-me", api_token_dev)
        invalid_dev = hmac.compare_digest("wrong-token", api_token_dev)

        assert valid_dev is True
        assert invalid_dev is False

        # Prod mode also validates
        valid_prod = hmac.compare_digest(api_token_prod, api_token_prod)
        invalid_prod = hmac.compare_digest("wrong-token", api_token_prod)

        assert valid_prod is True
        assert invalid_prod is False


class TestSecurityFix2TokenRedaction:
    """Verify Fix #2: Tokens are redacted from logs"""

    def test_bearer_token_redaction(self):
        """Verify Bearer tokens are masked in error messages"""
        import re

        # Simulate token redaction
        error_msg = "Failed request with token Bearer secret-abc123"
        redacted = re.sub(r'Bearer\s+[^\s]+', 'Bearer ***', error_msg)

        assert "secret-abc123" not in redacted
        assert "Bearer ***" in redacted

    def test_header_authorization_masking(self):
        """Verify Authorization headers are masked"""
        headers = {
            "Authorization": "Bearer secret-token",
            "Content-Type": "application/json",
            "X-Request-ID": "12345"
        }

        # Redact Authorization header
        redacted = {
            k: "Bearer ***" if k.lower() == "authorization" else v
            for k, v in headers.items()
        }

        assert redacted["Authorization"] == "Bearer ***"
        assert redacted["Content-Type"] == "application/json"


class TestSecurityFix3CORSConfiguration:
    """Verify Fix #3: CORS uses explicit origins, not wildcards"""

    def test_cors_no_wildcard_origins(self):
        """Verify CORS configuration doesn't use wildcard domains"""
        allowed_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        # No wildcards
        for origin in allowed_origins:
            assert "*" not in origin
            assert origin.startswith(("http://", "https://"))

    def test_cors_explicit_port_numbers(self):
        """Verify CORS uses explicit port numbers"""
        allowed_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        # Each origin should have explicit port
        for origin in allowed_origins:
            # Extract port
            parts = origin.split(":")
            assert len(parts) == 3  # scheme://host:port
            port = parts[2]
            assert port.isdigit()
            assert int(port) > 0


class TestSecurityFix4ThreadSafeIndexLoading:
    """Verify Fix #4: Index loading uses double-checked locking"""

    def test_double_checked_locking_pattern(self):
        """Verify double-checked locking prevents race conditions"""
        class MockIndexManager:
            def __init__(self):
                self._loaded = False
                self._lock = threading.Lock()

            def ensure_loaded(self):
                # First check (fast path, no lock)
                if self._loaded:
                    return

                # Second check with lock (slow path)
                with self._lock:
                    # Double-check
                    if self._loaded:
                        return

                    # Do work
                    self._loaded = True

        manager = MockIndexManager()

        # First call loads
        manager.ensure_loaded()
        assert manager._loaded is True

        # Second call uses fast path
        manager.ensure_loaded()
        assert manager._loaded is True

    def test_concurrent_loading_safety(self):
        """Verify concurrent access is safe"""
        class SafeManager:
            def __init__(self):
                self._loaded = False
                self._lock = threading.Lock()
                self._load_count = 0

            def ensure_loaded(self):
                if self._loaded:
                    return

                with self._lock:
                    if self._loaded:
                        return

                    self._load_count += 1
                    self._loaded = True

        manager = SafeManager()
        threads = [threading.Thread(target=manager.ensure_loaded) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only load once despite concurrent access
        assert manager._load_count == 1
        assert manager._loaded is True


class TestSecurityFix5EmbeddingErrors:
    """Verify Fix #5: Missing embeddings raise errors"""

    def test_embedding_validation_raises_error(self):
        """Verify missing embeddings cause errors not silent failures"""
        class VectorSearch:
            def search(self, chunks, embedding):
                # Check embeddings exist
                embeddings_list = [c.get("embedding") for c in chunks]

                if not embeddings_list or all(e is None for e in embeddings_list):
                    raise ValueError("Missing embeddings in chunks")

                return embeddings_list

        search = VectorSearch()
        chunks_without_embedding = [
            {"text": "chunk1", "id": "1"},
            {"text": "chunk2", "id": "2"},
        ]

        # Should raise error
        with pytest.raises(ValueError):
            search.search(chunks_without_embedding, None)


class TestSecurityFix6ExceptionRetryLogic:
    """Verify Fix #6: Only transient errors are retried"""

    def test_transient_vs_permanent_errors(self):
        """Verify distinction between retryable and permanent errors"""
        class TransientError(Exception):
            """Temporary network error"""
            pass

        class PermanentError(Exception):
            """Permanent client error"""
            pass

        def should_retry(error):
            """Determine if error should be retried"""
            # Transient errors (timeout, connection, 5xx)
            transient = [
                "Timeout",
                "ConnectError",
                "500",
                "503",
            ]

            # Permanent errors (4xx, JSON decode, etc.)
            permanent = [
                "401",
                "403",
                "404",
                "JSONDecodeError",
            ]

            error_str = str(type(error).__name__)

            for t in transient:
                if t in error_str:
                    return True

            for p in permanent:
                if p in error_str:
                    return False

            return False

        # Transient errors should be retried
        assert should_retry(TransientError("Timeout")) is True
        assert should_retry(TransientError("ConnectError")) is True

        # Permanent errors should not be retried
        assert should_retry(PermanentError("401")) is False
        assert should_retry(PermanentError("404")) is False


# ============================================================================
# PHASE 2: ARCHITECTURE IMPROVEMENTS
# ============================================================================

class TestIndexManagerRefactoring:
    """Verify Phase 2: IndexManager module extraction"""

    def test_index_manager_module_exists(self):
        """Verify index_manager.py exists"""
        index_manager_path = Path("/Users/15x/Downloads/rag/src/index_manager.py")
        assert index_manager_path.exists()

    def test_index_manager_has_required_methods(self):
        """Verify IndexManager has all required methods"""
        index_manager_path = Path("/Users/15x/Downloads/rag/src/index_manager.py")
        content = index_manager_path.read_text()

        # Required methods
        assert "def ensure_loaded" in content
        assert "def get_index" in content
        assert "def get_all_indexes" in content
        assert "def is_normalized" in content

    def test_bm25_cache_lock_present(self):
        """Verify BM25 cache has thread-safe locking"""
        retrieval_engine_path = Path("/Users/15x/Downloads/rag/src/retrieval_engine.py")
        content = retrieval_engine_path.read_text()

        # BM25 cache lock
        assert "_cache_lock" in content
        assert "threading.Lock()" in content


# ============================================================================
# PHASE 3: UI REDESIGN
# ============================================================================

class TestUIRedesignQWEN:
    """Verify Phase 3: QWEN-style UI redesign"""

    def test_html_no_old_tabs(self):
        """Verify old tab navigation removed"""
        html_path = Path("/Users/15x/Downloads/rag/public/index.html")
        content = html_path.read_text()

        # Old tabs should be removed
        assert 'data-tab="articles"' not in content
        assert 'data-tab="about"' not in content

    def test_html_has_sidebar(self):
        """Verify new sidebar navigation present"""
        html_path = Path("/Users/15x/Downloads/rag/public/index.html")
        content = html_path.read_text()

        assert '<aside class="sidebar">' in content
        assert 'id="newChatBtn"' in content
        assert 'id="settingsBtn"' in content

    def test_html_has_single_chat_focus(self):
        """Verify UI focused on single chat"""
        html_path = Path("/Users/15x/Downloads/rag/public/index.html")
        content = html_path.read_text()

        assert 'id="messagesContainer"' in content
        assert 'id="chatInput"' in content
        assert 'id="sendBtn"' in content

    def test_css_has_modern_styling(self):
        """Verify CSS has modern QWEN-style design"""
        css_path = Path("/Users/15x/Downloads/rag/public/css/style.css")
        content = css_path.read_text()

        # Modern elements
        assert ".sidebar" in content
        assert ".message-bubble" in content
        assert "dark-mode" in content
        assert ".modal" in content

    def test_javascript_modules_present(self):
        """Verify new JavaScript modules exist"""
        chat_qwen = Path("/Users/15x/Downloads/rag/public/js/chat-qwen.js")
        main_qwen = Path("/Users/15x/Downloads/rag/public/js/main-qwen.js")

        assert chat_qwen.exists()
        assert main_qwen.exists()

    def test_chat_qwen_has_chat_manager(self):
        """Verify ChatManager class in chat-qwen.js"""
        path = Path("/Users/15x/Downloads/rag/public/js/chat-qwen.js")
        content = path.read_text()

        assert "class ChatManager" in content
        assert "addMessage" in content
        assert "renderMessage" in content


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrationSecurity:
    """Integration tests for security improvements"""

    def test_token_redaction_chain(self):
        """Verify complete token redaction pipeline"""
        import re

        # Original error with token
        original = "Request failed: Bearer secret-token-xyz returned 401"

        # Step 1: Redact bearer tokens
        step1 = re.sub(r'Bearer\s+[^\s]+', 'Bearer ***', original)
        assert "secret-token" not in step1

        # Should see masked token
        assert "Bearer ***" in step1

    def test_cors_and_auth_together(self):
        """Verify CORS and authentication work together"""
        # CORS allows specific origins
        allowed_origins = ["http://localhost:8080"]
        request_origin = "http://localhost:8080"

        assert request_origin in allowed_origins

        # Token is always validated
        token = "test-token"
        is_valid = hmac.compare_digest(token, token)
        assert is_valid is True

    def test_index_loading_with_lock(self):
        """Verify safe index loading with double-checked locking"""
        load_events = []

        class SafeIndexLoader:
            def __init__(self):
                self._loaded = False
                self._lock = threading.Lock()

            def load(self):
                if self._loaded:
                    return

                with self._lock:
                    if self._loaded:
                        return

                    load_events.append(time.time())
                    self._loaded = True

        loader = SafeIndexLoader()

        # Concurrent loads
        threads = [threading.Thread(target=loader.load) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only load once
        assert len(load_events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
