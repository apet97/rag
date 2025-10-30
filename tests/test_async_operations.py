"""
Test Suite for Async Operations and Concurrency Improvements

Tests cover:
- Thread pool operations for FAISS and embeddings
- Async HTTP client with connection pooling
- Batch embedding operations
- Parallel search operations
- Context management and cleanup
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from src.async_operations import (
    _get_faiss_executor,
    _get_embedding_executor,
    run_in_thread_pool,
    async_faiss_search,
    async_batch_embeddings,
    async_parallel_searches,
    AsyncHTTPClientPool,
    AsyncBatcher,
    shutdown_async_operations,
)

# ============================================================================
# Thread Pool Tests
# ============================================================================

class TestThreadPoolManagement:
    """Test thread pool creation and management."""

    def test_get_faiss_executor_creates_executor(self):
        """Thread pool executor should be created on first call."""
        executor = _get_faiss_executor(max_workers=2)
        assert executor is not None
        assert executor._max_workers == 2

    def test_get_faiss_executor_reuses_existing(self):
        """Thread pool should be reused on subsequent calls."""
        executor1 = _get_faiss_executor()
        executor2 = _get_faiss_executor()
        assert executor1 is executor2

    def test_get_embedding_executor_creates_executor(self):
        """Embedding thread pool should be created on first call."""
        executor = _get_embedding_executor(max_workers=2)
        assert executor is not None
        assert executor._max_workers == 2

    def test_executor_configuration(self):
        """Executor should have correct max workers."""
        executor = _get_faiss_executor(max_workers=4)
        assert executor._max_workers == 4

# ============================================================================
# Async Operation Tests
# ============================================================================

class TestRunInThreadPool:
    """Test run_in_thread_pool wrapper."""

    @pytest.mark.asyncio
    async def test_blocking_function_in_thread_pool(self):
        """Blocking function should execute in thread pool."""
        def blocking_func(x: int, y: int) -> int:
            return x + y

        result = await run_in_thread_pool(blocking_func, 5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_function_with_kwargs(self):
        """Function should accept keyword arguments."""
        def multiply(a: int, b: int = 2) -> int:
            return a * b

        result = await run_in_thread_pool(multiply, 3, b=4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Exceptions should be propagated from thread."""
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await run_in_thread_pool(failing_func)

    @pytest.mark.asyncio
    async def test_numpy_operations_in_thread(self):
        """NumPy operations should work in thread pool."""
        def compute_norm(vec: np.ndarray) -> float:
            return float(np.linalg.norm(vec))

        vec = np.array([3.0, 4.0])
        result = await run_in_thread_pool(compute_norm, vec)
        assert result == pytest.approx(5.0)

# ============================================================================
# FAISS Search Tests
# ============================================================================

class TestAsyncFAISSSearch:
    """Test async FAISS search wrapper."""

    @pytest.mark.asyncio
    async def test_async_faiss_search(self):
        """Async FAISS search should execute in thread pool."""
        # Mock FAISS search function
        def mock_search(query_vec: np.ndarray, k: int):
            # Return distances and indices
            distances = np.array([[0.1, 0.2, 0.3]])
            indices = np.array([[0, 1, 2]])
            return distances, indices

        query_vec = np.random.rand(1, 768).astype(np.float32)
        distances, indices = await async_faiss_search(mock_search, query_vec, k=3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
        np.testing.assert_array_equal(indices[0], [0, 1, 2])

    @pytest.mark.asyncio
    async def test_parallel_faiss_searches(self):
        """Multiple FAISS searches should run in parallel."""
        def mock_search1(query_vec, k):
            return np.array([[0.1, 0.2]]), np.array([[0, 1]])

        def mock_search2(query_vec, k):
            return np.array([[0.15, 0.25]]), np.array([[1, 2]])

        query_vec = np.random.rand(1, 768).astype(np.float32)
        searches = [
            (mock_search1, query_vec, 2),
            (mock_search2, query_vec, 2),
        ]

        results = await async_parallel_searches(searches)
        assert len(results) == 2

# ============================================================================
# Batch Embedding Tests
# ============================================================================

class TestAsyncBatchEmbeddings:
    """Test async batch embedding operations."""

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Empty batch should return empty array."""
        async def mock_embed(texts):
            return np.random.rand(len(texts), 768).astype(np.float32)

        result = await async_batch_embeddings(mock_embed, [], batch_size=32)
        assert result.shape == (0, 0) or result.size == 0

    @pytest.mark.asyncio
    async def test_single_batch(self):
        """Single batch smaller than batch_size should process correctly."""
        texts = ["hello", "world"]

        async def mock_embed(texts_batch):
            return np.random.rand(len(texts_batch), 768).astype(np.float32)

        result = await async_batch_embeddings(mock_embed, texts, batch_size=32)
        assert result.shape == (2, 768)

    @pytest.mark.asyncio
    async def test_multiple_batches(self):
        """Multiple batches should concatenate correctly."""
        texts = [f"text {i}" for i in range(100)]

        async def mock_embed(texts_batch):
            return np.random.rand(len(texts_batch), 768).astype(np.float32)

        result = await async_batch_embeddings(mock_embed, texts, batch_size=32)
        assert result.shape == (100, 768)

    @pytest.mark.asyncio
    async def test_batch_embedding_error_handling(self):
        """Batch embedding should raise RuntimeError on failure."""
        def failing_embed(texts):
            raise Exception("Embedding service unavailable")

        with pytest.raises(RuntimeError, match="Failed to embed texts"):
            await async_batch_embeddings(failing_embed, ["text"], batch_size=32)

# ============================================================================
# Async HTTP Client Pool Tests
# ============================================================================

class TestAsyncHTTPClientPool:
    """Test async HTTP client pool management."""

    def test_initialization(self):
        """Pool should initialize with configuration."""
        pool = AsyncHTTPClientPool(
            max_connections=20,
            max_keepalive_connections=10,
            timeout=30.0,
        )
        assert pool.max_connections == 20
        assert pool.max_keepalive_connections == 10
        assert pool.timeout == 30.0

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Pool should create client on first access."""
        pool = AsyncHTTPClientPool()
        client = await pool.get_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """Pool should reuse client on subsequent accesses."""
        pool = AsyncHTTPClientPool()
        client1 = await pool.get_client()
        client2 = await pool.get_client()
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_client_cleanup(self):
        """Pool should properly close client."""
        pool = AsyncHTTPClientPool()
        client = await pool.get_client()
        await pool.close()
        assert pool._client is None

# ============================================================================
# Async Batcher Tests
# ============================================================================

class TestAsyncBatcher:
    """Test async batching functionality."""

    def test_initialization(self):
        """Batcher should initialize with configuration."""
        batcher = AsyncBatcher(batch_size=32, max_wait_ms=100)
        assert batcher.batch_size == 32
        assert batcher.max_wait_seconds == 0.1
        assert batcher.is_empty()

    @pytest.mark.asyncio
    async def test_batch_by_size(self):
        """Batcher should return batch when size reached."""
        batcher = AsyncBatcher(batch_size=2, max_wait_ms=1000)

        # Add items
        await batcher.add("item1")
        await batcher.add("item2")

        # Get batch (should return immediately when size reached)
        batch = await asyncio.wait_for(batcher.get_batch(), timeout=1.0)
        assert len(batch) == 2
        assert batch == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_batch_by_timeout(self):
        """Batcher should return partial batch on timeout."""
        batcher = AsyncBatcher(batch_size=10, max_wait_ms=50)

        await batcher.add("item1")
        await batcher.add("item2")

        # Wait for partial batch
        batch = await asyncio.wait_for(batcher.get_batch(), timeout=1.0)
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_batcher_empty_after_get(self):
        """Batcher should be empty after getting batch."""
        batcher = AsyncBatcher(batch_size=2, max_wait_ms=100)

        await batcher.add("item1")
        await batcher.add("item2")

        await batcher.get_batch()
        assert batcher.is_empty()

# ============================================================================
# Shutdown Tests
# ============================================================================

class TestShutdown:
    """Test async operations shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self):
        """Shutdown should cleanup all resources."""
        # Initialize some resources
        _get_faiss_executor()
        _get_embedding_executor()

        # Shutdown
        await shutdown_async_operations()

        # Should be ready to create new ones
        executor1 = _get_faiss_executor()
        executor2 = _get_embedding_executor()

        assert executor1 is not None
        assert executor2 is not None

# ============================================================================
# Integration Tests
# ============================================================================

class TestAsyncConcurrencyIntegration:
    """Integration tests for async concurrency improvements."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Multiple async operations should run concurrently."""
        async def task(duration: float) -> str:
            await asyncio.sleep(duration)
            return f"completed {duration}"

        # All tasks should complete in ~max_duration, not sum
        import time
        start = time.time()

        tasks = [
            run_in_thread_pool(lambda d=d: __import__("time").sleep(d), d)
            for d in [0.01, 0.01, 0.01]
        ]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start
        # Should be ~0.01s (concurrent) not 0.03s (sequential)
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_embedding_batching_performance(self):
        """Batch embeddings should be more efficient than sequential."""
        texts = [f"text {i}" for i in range(10)]
        call_count = 0

        async def counting_embed(batch):
            nonlocal call_count
            call_count += 1
            return np.random.rand(len(batch), 768).astype(np.float32)

        # With batch_size=5, should take 2 calls
        await async_batch_embeddings(counting_embed, texts, batch_size=5)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_searches_efficiency(self):
        """Parallel searches should execute concurrently."""
        def search_ns1(query, k):
            return np.array([[0.1, 0.2]]), np.array([[0, 1]])

        def search_ns2(query, k):
            return np.array([[0.15, 0.25]]), np.array([[1, 2]])

        query = np.random.rand(1, 768).astype(np.float32)
        searches = [
            (search_ns1, query, 2),
            (search_ns2, query, 2),
        ]

        results = await async_parallel_searches(searches)
        assert len(results) == 2
