"""
Prometheus metrics for RAG API monitoring.

Provides counters, histograms, and gauges for tracking:
- Request counts and status codes
- Request latency distributions
- Cache hit rates
- Index operations
- LLM call statistics
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional

# ============================================================================
# HTTP Request Metrics
# ============================================================================

request_count = Counter(
    'rag_requests_total',
    'Total number of HTTP requests',
    ['endpoint', 'method', 'status']
)

request_latency = Histogram(
    'rag_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint', 'method'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ============================================================================
# Cache Metrics
# ============================================================================

cache_hits = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'rag_cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

cache_size = Gauge(
    'rag_cache_size_entries',
    'Current number of entries in cache',
    ['cache_type']
)

cache_evictions = Counter(
    'rag_cache_evictions_total',
    'Total number of cache evictions',
    ['cache_type']
)

# ============================================================================
# Search & Retrieval Metrics
# ============================================================================

search_query_count = Counter(
    'rag_search_queries_total',
    'Total number of search queries',
    ['namespace', 'query_type']
)

search_results_count = Histogram(
    'rag_search_results_count',
    'Number of results returned per search',
    ['namespace'],
    buckets=(0, 1, 3, 5, 10, 20, 50, 100)
)

search_latency = Histogram(
    'rag_search_duration_seconds',
    'Search operation latency in seconds',
    ['namespace', 'strategy'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

rerank_latency = Histogram(
    'rag_rerank_duration_seconds',
    'Reranking operation latency in seconds',
    ['namespace'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# ============================================================================
# LLM Metrics
# ============================================================================

llm_requests = Counter(
    'rag_llm_requests_total',
    'Total number of LLM requests',
    ['model', 'status']
)

llm_latency = Histogram(
    'rag_llm_duration_seconds',
    'LLM request latency in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
)

llm_tokens = Histogram(
    'rag_llm_tokens_total',
    'Estimated LLM tokens per request',
    ['model', 'type'],  # type: input or output
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000)
)

# ============================================================================
# Index Metrics
# ============================================================================

index_size = Gauge(
    'rag_index_vectors_count',
    'Number of vectors in FAISS index',
    ['namespace']
)

index_load_time = Histogram(
    'rag_index_load_duration_seconds',
    'Time taken to load index on startup',
    ['namespace'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
)

# ============================================================================
# System Metrics
# ============================================================================

system_uptime = Gauge(
    'rag_system_uptime_seconds',
    'System uptime in seconds'
)

active_namespaces = Gauge(
    'rag_active_namespaces_count',
    'Number of active namespaces'
)

# ============================================================================
# Circuit Breaker Metrics
# ============================================================================

circuit_breaker_state = Gauge(
    'rag_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)',
    ['name']
)

circuit_breaker_requests = Counter(
    'rag_circuit_breaker_requests_total',
    'Total requests through circuit breaker',
    ['name', 'status']  # status: success or failure
)

circuit_breaker_failures = Counter(
    'rag_circuit_breaker_failures_total',
    'Total failures in circuit breaker',
    ['name']
)

circuit_breaker_consecutive_failures = Gauge(
    'rag_circuit_breaker_consecutive_failures',
    'Current consecutive failures',
    ['name']
)

circuit_breaker_state_changes = Counter(
    'rag_circuit_breaker_state_changes_total',
    'Circuit breaker state transitions',
    ['name', 'from_state', 'to_state']
)

# ============================================================================
# Helper Functions
# ============================================================================

def track_request(endpoint: str, method: str, status: int, duration: float) -> None:
    """
    Track HTTP request metrics.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        status: HTTP status code
        duration: Request duration in seconds
    """
    request_count.labels(endpoint=endpoint, method=method, status=status).inc()
    request_latency.labels(endpoint=endpoint, method=method).observe(duration)


def track_cache_operation(cache_type: str, hit: bool, size: Optional[int] = None) -> None:
    """
    Track cache hit/miss metrics.

    Args:
        cache_type: Type of cache (response, bm25, vector, etc.)
        hit: True if cache hit, False if miss
        size: Current cache size (optional)
    """
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()

    if size is not None:
        cache_size.labels(cache_type=cache_type).set(size)


def track_search(namespace: str, query_type: str, num_results: int,
                 duration: float, strategy: str = "hybrid") -> None:
    """
    Track search operation metrics.

    Args:
        namespace: Search namespace
        query_type: Type of query (simple, multi-intent, etc.)
        num_results: Number of results returned
        duration: Search duration in seconds
        strategy: Retrieval strategy used
    """
    search_query_count.labels(namespace=namespace, query_type=query_type).inc()
    search_results_count.labels(namespace=namespace).observe(num_results)
    search_latency.labels(namespace=namespace, strategy=strategy).observe(duration)


def track_llm_request(model: str, status: str, duration: float,
                      input_tokens: Optional[int] = None,
                      output_tokens: Optional[int] = None) -> None:
    """
    Track LLM request metrics.

    Args:
        model: LLM model name
        status: Request status (success, error, timeout, etc.)
        duration: Request duration in seconds
        input_tokens: Number of input tokens (optional)
        output_tokens: Number of output tokens (optional)
    """
    llm_requests.labels(model=model, status=status).inc()
    llm_latency.labels(model=model).observe(duration)

    if input_tokens is not None:
        llm_tokens.labels(model=model, type="input").observe(input_tokens)
    if output_tokens is not None:
        llm_tokens.labels(model=model, type="output").observe(output_tokens)


def track_circuit_breaker(name: str, state: str, metrics_data: dict) -> None:
    """
    Track circuit breaker metrics.

    Args:
        name: Circuit breaker name
        state: Current state (closed, half_open, open)
        metrics_data: Dictionary with total_requests, total_failures, total_successes, consecutive_failures
    """
    # Map state to numeric value for Prometheus
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    circuit_breaker_state.labels(name=name).set(state_map.get(state, 0))

    # Update counters (we set them to match the circuit breaker's internal counters)
    circuit_breaker_consecutive_failures.labels(name=name).set(
        metrics_data.get("consecutive_failures", 0)
    )


def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics as bytes
    """
    return generate_latest()


def get_content_type() -> str:
    """
    Get Prometheus metrics content type.

    Returns:
        Content-Type header value
    """
    return CONTENT_TYPE_LATEST
