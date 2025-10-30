"""Pytest configuration and fixtures for RAG tests."""

import os
import time
import pytest


def pytest_configure(config):
    """Register custom pytest markers for test categorization."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (skip with '-m \"not integration\"')"
    )


@pytest.fixture(autouse=True)
def reset_rate_limiter_and_cache(request):
    """Reset rate limiter and cache state between tests to avoid 429 errors and cache collisions.

    This fixture:
    1. Clears the module-level _last_req dict in server.py (rate-limiter state)
    2. Clears the response cache from cache.py
    3. Sleeps briefly to ensure rate-limiter window has passed

    Skipped for fixture sanity tests that don't need server imports.
    """
    # Skip for fixture sanity tests
    if "test_fixture_sanity" in request.node.nodeid:
        yield
        return

    # Clear rate limiter state before test runs
    from src import server
    server._last_req.clear()

    # Clear response cache before test runs
    from src.cache import get_cache
    cache = get_cache()
    cache.clear()

    yield

    # Cleanup after test (optional, but good practice)
    server._last_req.clear()
    cache.clear()


@pytest.fixture
def ci_environment():
    """Check if running in CI environment.

    Returns True if CI environment variables are detected, allowing tests
    to behave differently in CI (e.g., use lighter fixtures, skip certain operations).
    """
    ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_HOME"]
    return any(os.getenv(var) for var in ci_vars)
