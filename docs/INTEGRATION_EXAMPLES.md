# Integration Examples

This guide shows how to integrate with the RAG system's new Tier 2 observability features.

## Python Client Integration

### Basic Search with Performance Tracking

```python
import requests
import json
from datetime import datetime

class RAGClient:
    def __init__(self, base_url="http://localhost:7000", api_token="change-me"):
        self.base_url = base_url
        self.headers = {"x-api-token": api_token}

    def search(self, query: str, k: int = 5) -> dict:
        """Search with performance tracking."""
        response = requests.get(
            f"{self.base_url}/search",
            params={"q": query, "k": k, "namespace": "clockify"},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_perf_metrics(self) -> dict:
        """Get current performance metrics."""
        response = requests.get(
            f"{self.base_url}/perf?detailed=true",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Usage
client = RAGClient()
results = client.search("how to track time")
metrics = client.get_perf_metrics()

print(f"Found {len(results['results'])} results")
print(f"Vector search p95: {metrics['by_stage']['vector_search']['p95_ms']}ms")
print(f"Cache hit rate: {metrics['by_stage']['cache_lookup']['count']}")
```

### Performance Monitoring Loop

```python
import time
import statistics
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self, client: RAGClient):
        self.client = client
        self.measurements = defaultdict(list)

    def run_benchmark(self, queries: list[str], iterations: int = 3):
        """Run benchmark and track performance."""
        for i in range(iterations):
            print(f"\n=== Iteration {i+1}/{iterations} ===")

            for query in queries:
                try:
                    start = time.time()
                    results = self.client.search(query)
                    elapsed_ms = (time.time() - start) * 1000

                    self.measurements[query].append(elapsed_ms)
                    print(f"Query: '{query}' - {elapsed_ms:.0f}ms")
                except Exception as e:
                    print(f"Error querying '{query}': {e}")

            time.sleep(1)

        self.print_summary()

    def print_summary(self):
        """Print benchmark summary."""
        print("\n=== Benchmark Summary ===")
        for query, times in self.measurements.items():
            avg = statistics.mean(times)
            p95 = statistics.quantiles(times, n=20)[18] if len(times) > 1 else times[0]
            print(f"Query: '{query}'")
            print(f"  Average: {avg:.0f}ms, P95: {p95:.0f}ms")

# Usage
monitor = PerformanceMonitor(client)
test_queries = [
    "how to track time",
    "timesheet creation",
    "project management"
]
monitor.run_benchmark(test_queries, iterations=3)
```

### Cache Performance Analysis

```python
class CacheAnalyzer:
    def __init__(self, client: RAGClient):
        self.client = client

    def analyze_cache_effectiveness(self) -> dict:
        """Analyze cache performance by namespace."""
        health = requests.get(
            f"{self.client.base_url}/health",
            headers=self.client.headers
        ).json()

        cache_stats = health.get('cache', {}).get('semantic_cache_stats', {})

        analysis = {
            "overall": {
                "hit_rate": cache_stats.get('hit_rate_pct', 0),
                "size": cache_stats.get('size', 0),
                "memory_mb": cache_stats.get('memory_usage_mb', 0),
            },
            "by_namespace": cache_stats.get('by_namespace', {})
        }

        return analysis

    def print_recommendations(self):
        """Print cache tuning recommendations."""
        analysis = self.analyze_cache_effectiveness()
        hit_rate = analysis['overall']['hit_rate']

        print("=== Cache Optimization Recommendations ===")

        if hit_rate < 30:
            print("⚠️  Low cache hit rate (<30%)")
            print("  Action: Increase SEMANTIC_CACHE_TTL_SECONDS")
        elif hit_rate < 50:
            print("⚠️  Moderate cache hit rate (<50%)")
            print("  Action: Consider increasing SEMANTIC_CACHE_MAX_SIZE")
        else:
            print("✅ Good cache hit rate (>50%)")

        for ns, stats in analysis['by_namespace'].items():
            ns_hit_rate = stats.get('hit_rate_pct', 0)
            if ns_hit_rate < 40:
                print(f"  Namespace '{ns}' has low hit rate: {ns_hit_rate}%")

# Usage
analyzer = CacheAnalyzer(client)
analyzer.print_recommendations()
```

## JavaScript/Node.js Integration

```javascript
class RAGClient {
  constructor(baseUrl = 'http://localhost:7000', apiToken = 'change-me') {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
  }

  async search(query, k = 5, namespace = 'clockify') {
    const response = await fetch(
      `${this.baseUrl}/search?q=${encodeURIComponent(query)}&k=${k}&namespace=${namespace}`,
      {
        headers: { 'x-api-token': this.apiToken }
      }
    );
    return response.json();
  }

  async getPerformanceMetrics(detailed = false) {
    const url = detailed
      ? `${this.baseUrl}/perf?detailed=true`
      : `${this.baseUrl}/perf`;

    const response = await fetch(url, {
      headers: { 'x-api-token': this.apiToken }
    });
    return response.json();
  }

  async monitorLatency(stage) {
    const metrics = await this.getPerformanceMetrics(true);
    const stageMetrics = metrics.by_stage[stage];

    return {
      stage: stage,
      mean: stageMetrics.mean_ms,
      p95: stageMetrics.p95_ms,
      p99: stageMetrics.p99_ms,
      samples: stageMetrics.count
    };
  }
}

// Usage
const client = new RAGClient();

// Search
const results = await client.search('how to track time', 5);
console.log(`Found ${results.results.length} results`);

// Monitor performance
const vectorSearchMetrics = await client.monitorLatency('vector_search');
console.log(`Vector search p95: ${vectorSearchMetrics.p95}ms`);
```

## Grafana Integration

### Dashboard JSON Configuration

```json
{
  "dashboard": {
    "title": "RAG System Performance",
    "panels": [
      {
        "title": "Pipeline Latency - P95",
        "targets": [
          {
            "expr": "rag_vector_search_p95_ms",
            "legendFormat": "Vector Search"
          },
          {
            "expr": "rag_bm25_search_p95_ms",
            "legendFormat": "BM25 Search"
          },
          {
            "expr": "rag_llm_generation_p95_ms",
            "legendFormat": "LLM Generation"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rag_cache_hit_rate_pct",
            "legendFormat": "Overall"
          },
          {
            "expr": "rag_cache_hit_rate_by_namespace",
            "legendFormat": "{{ namespace }}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "rag_cache_memory_mb"
          }
        ]
      }
    ]
  }
}
```

### Prometheus Scrape Configuration

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-system'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

## FastAPI Hook Integration

```python
from fastapi import Request
from typing import Callable

class PerformanceMiddleware:
    """Middleware to track request performance."""

    def __init__(self, app, exclude_paths=["/metrics", "/health"]):
        self.app = app
        self.exclude_paths = exclude_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        import time
        from src.performance_tracker import get_performance_tracker, PipelineStage

        tracker = get_performance_tracker()
        query_id = str(scope.get("query_string", ""))

        tracker.start_timer(query_id)

        try:
            await self.app(scope, receive, send)
        finally:
            tracker.end_timer(query_id)

# Usage
from fastapi import FastAPI
app = FastAPI()
app.add_middleware(PerformanceMiddleware)
```

## Logging Integration

```python
import logging
from src.performance_tracker import get_performance_tracker

class PerformanceLogger(logging.Handler):
    """Log performance metrics to standard logging."""

    def emit(self, record):
        tracker = get_performance_tracker()
        stats = tracker.get_stats()

        # Log bottlenecks
        bottlenecks = tracker.get_bottlenecks(top_n=3)
        for bottleneck in bottlenecks:
            logging.info(
                f"Bottleneck: {bottleneck['stage']} "
                f"mean={bottleneck['mean_ms']}ms p95={bottleneck['p95_ms']}ms"
            )

# Setup
logger = logging.getLogger('rag-performance')
logger.addHandler(PerformanceLogger())
logger.setLevel(logging.INFO)
```

## Query Insights Integration

```python
from src.query_insights import QueryInsight, get_query_insight_tracker

def log_search_insight(query: str, results: dict, query_type: str = None):
    """Log query processing insight."""
    insight = QueryInsight(
        query_id=results.get('query_id', ''),
        query_text=query,
        query_type=query_type,
        vector_results_count=len(results.get('vector_results', [])),
        bm25_results_count=len(results.get('bm25_results', [])),
        fused_results_count=len(results.get('results', [])),
        mmr_applied=results.get('mmr_applied', False),
        cache_hit=results.get('cache_hit', False),
    )

    tracker = get_query_insight_tracker()
    tracker.record_query(insight)

# Get insights summary
insights = tracker.get_insights_summary()
print(f"Cache hit rate: {insights['feature_adoption']['cache_hit_rate_pct']}%")
```

## Monitoring Dashboard Script

```python
#!/usr/bin/env python3
import time
import curses
import requests
from tabulate import tabulate

class RAGDashboard:
    def __init__(self, base_url="http://localhost:7000", refresh_interval=5):
        self.base_url = base_url
        self.refresh_interval = refresh_interval
        self.headers = {"x-api-token": "change-me"}

    def get_data(self):
        """Fetch current metrics."""
        try:
            perf = requests.get(
                f"{self.base_url}/perf?detailed=true",
                headers=self.headers,
                timeout=5
            ).json()

            health = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=5
            ).json()

            return perf, health
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return None, None

    def run(self, stdscr):
        """Run dashboard in curses window."""
        curses.curs_set(0)

        while True:
            perf, health = self.get_data()

            if perf and health:
                stdscr.clear()

                # Performance table
                perf_data = []
                for stage, stats in perf['by_stage'].items():
                    perf_data.append([
                        stage,
                        f"{stats['mean_ms']:.0f}",
                        f"{stats['p95_ms']:.0f}",
                        f"{stats['count']}"
                    ])

                stdscr.addstr("=== RAG System Performance Dashboard ===\n\n")
                stdscr.addstr(tabulate(
                    perf_data,
                    headers=['Stage', 'Mean', 'P95', 'Count'],
                    tablefmt='grid'
                ))

                # Cache stats
                cache = health.get('cache', {}).get('semantic_cache_stats', {})
                stdscr.addstr(f"\n\nCache Hit Rate: {cache.get('hit_rate_pct', 0):.1f}%\n")
                stdscr.addstr(f"Memory Usage: {cache.get('memory_usage_mb', 0):.1f}MB\n")
                stdscr.addstr(f"\nRefresh interval: {self.refresh_interval}s (Press Ctrl+C to exit)\n")

                stdscr.refresh()

            time.sleep(self.refresh_interval)

# Run: python3 monitoring_dashboard.py
if __name__ == "__main__":
    import curses
    dashboard = RAGDashboard(refresh_interval=5)
    curses.wrapper(dashboard.run)
```

## Summary

These integration examples show how to:
- ✅ Build Python/JS clients for monitoring
- ✅ Integrate with Grafana for visualization
- ✅ Connect to Prometheus for metrics collection
- ✅ Use FastAPI middleware for performance tracking
- ✅ Build real-time monitoring dashboards
- ✅ Analyze cache effectiveness
- ✅ Log query insights

Use these patterns to integrate RAG observability into your monitoring stack.
