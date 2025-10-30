# Code Part 4

## .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
env.bak/
venv.bak/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs (keep logs/ directory structure but ignore content)
logs/*.txt
logs/**/*.txt
logs/*.json
logs/**/*.json
logs/*.jsonl
logs/**/*.jsonl
logs/*.md
logs/**/*.md
*.log

# Large files
*.bin
*.pkl
*.tar.gz
*.zip

# OS
Thumbs.db
.DS_Store

# Temporary
tmp/
temp/
*.tmp
codex/

# Cache
.cache/
.mypy_cache/

```

## LICENSE

```
MIT License

Copyright (c) 2024 Clockify RAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## QUICKSTART.md

```
# Clockify RAG - Quick Start (3 Commands)

## Production-Ready Out of the Box ✅

**No configuration needed. Just run these 3 commands:**

---

## Step 1: Clone Repository
```bash
git clone <your-repo-url> clockify-rag
cd clockify-rag
```

---

## Step 2: Build Knowledge Base
```bash
make ingest
```
*Takes ~5 minutes. Builds FAISS indexes for Clockify + LangChain docs.*

---

## Step 3: Start Server
```bash
make serve
```
*Server starts on port 7001 with production settings.*

---

## ✅ That's It!

**Your RAG system is now running:**
- **URL:** http://localhost:7001
- **API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`
- **LLM:** Internal gpt-oss:20b (10.127.0.192:11434)

---

## Quick Test

### Health Check
```bash
curl http://localhost:7001/health | python3 -m json.tool
```

### Search Query
```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=how%20to%20track%20time&k=5'
```

### Chat with Citations
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I create a project?", "k": 5}'
```

---

## Production Features (Pre-Configured)

✅ Port 7001 (production)
✅ Secure token authentication
✅ Internal LLM (no API key needed)
✅ Harmony format (gpt-oss:20b optimized)
✅ Hybrid search (BM25 + Vector)
✅ Cross-encoder reranking
✅ Circuit breakers & fault tolerance
✅ Semantic caching (10K queries)
✅ Rate limiting & CORS

**Everything works out of the box!** 🚀

---

For detailed documentation, see: [DEPLOY.md](DEPLOY.md)
```

## docker-compose.yml

```
version: '3.8'

services:
  # RAG API Server
  rag:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-system
    restart: unless-stopped
    ports:
      - "7000:7000"
    environment:
      # Required - Change this in production!
      API_TOKEN: ${API_TOKEN:-change-me}

      # Server Configuration
      API_HOST: 0.0.0.0
      API_PORT: 7000
      ENV: ${ENV:-dev}

      # Ollama Configuration
      LLM_BASE_URL: http://ollama:11434
      EMBEDDING_MODEL: intfloat/multilingual-e5-base
      LLM_MODEL: gpt-oss:20b
      TRANSFORMERS_CACHE: /app/.cache/huggingface

      # Cache Configuration
      RESPONSE_CACHE_SIZE: 1000
      RESPONSE_CACHE_TTL: 3600

      # Rate Limiting
      RATE_LIMIT_RPS: 10

      # Logging
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
      LOG_FILE: /app/logs/rag.log

    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/.cache/huggingface

    networks:
      - rag-network

    depends_on:
      ollama:
        condition: service_healthy

    healthcheck:
      test: ["CMD", "curl", "-f", "-H", "x-api-token: ${API_TOKEN:-change-me}", "http://localhost:7000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 20s

  # Ollama - Local LLM Server
  ollama:
    image: ollama/ollama:latest
    container_name: rag-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"

    environment:
      OLLAMA_HOST: 0.0.0.0:11434

    volumes:
      - ollama_data:/root/.ollama

    networks:
      - rag-network

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

volumes:
  ollama_data:
    driver: local

networks:
  rag-network:
    driver: bridge
```

## docs/INTEGRATION_EXAMPLES.md

```
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
```

## public/js/chat.js

```
/**
 * Chat Interface Controller
 */

const chatState = {
    conversationId: null,
    isLoading: false
};

document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    const chatSendBtn = document.getElementById('chatSendBtn');

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatSendBtn.addEventListener('click', sendMessage);
});

async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();

    if (!message || chatState.isLoading) return;

    chatState.isLoading = true;
    addChatMessage(message, 'user');
    chatInput.value = '';

    // Show loading indicator
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'message assistant';
    loadingMsg.innerHTML = `<div class="message-content"><div class="message-loading"><div class="spinner"></div> Thinking...</div></div>`;
    document.getElementById('chatMessages').appendChild(loadingMsg);

    try {
        const response = await api.chat(message);

        // Remove loading message
        loadingMsg.remove();

        // Add assistant response
        const answer = response.answer || 'No response generated';
        addChatMessage(answer, 'assistant');

        // Display sources if available
        if (response.sources && response.sources.length > 0) {
            displaySources(response.sources);
        }

        chatState.conversationId = response.conversation_id;
    } catch (error) {
        loadingMsg.remove();
        addChatMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        chatState.isLoading = false;
        document.getElementById('chatInput').focus();
    }
}

function addChatMessage(text, sender) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${sender}`;

    const content = document.createElement('div');
    content.className = 'message-content';

    // Sanitize text to prevent XSS while allowing markdown formatting
    const sanitized = sanitizeText(text);

    // Apply safe markdown-like formatting
    let formattedText = sanitized
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');

    content.innerHTML = formattedText;
    messageEl.appendChild(content);

    document.getElementById('chatMessages').appendChild(messageEl);
    document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
}

/**
 * Sanitize text to prevent XSS attacks
 * Escapes HTML entities to prevent script injection
 */
function sanitizeText(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function displaySources(sources) {
    const sourcesPanel = document.getElementById('sourcesPanel');
    const sourcesList = document.getElementById('sourcesList');

    sourcesList.innerHTML = '';

    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.innerHTML = `
            <strong>[${source.id}] ${source.title}</strong>
            <div style="font-size: 0.75rem; color: var(--text-light);">
                ${source.namespace} • <a href="${source.url}" target="_blank" rel="noopener">View article</a>
            </div>
        `;
        sourcesList.appendChild(sourceItem);
    });

    sourcesPanel.style.display = 'block';
}
```

## scripts/coverage_audit.py

```
#!/usr/bin/env python3
"""
Coverage audit script: Compute category distribution, breadcrumb health, and chunk statistics.

Generates a comprehensive analysis of documentation coverage to track quality over time.
Output is JSON for easy parsing and historical tracking.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import sys

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_chunks(namespace: str = "clockify") -> List[Dict]:
    """Load all chunks for a namespace."""
    chunks_file = Path(f"data/chunks/{namespace}.jsonl")
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return chunks


def analyze_breadcrumbs(chunks: List[Dict]) -> Dict:
    """Analyze breadcrumb distribution and health."""
    breadcrumb_lengths = Counter()
    category_chunks = defaultdict(int)
    subcategory_chunks = defaultdict(int)
    fallback_count = 0

    for chunk in chunks:
        bc = chunk.get("breadcrumb", [])
        breadcrumb_lengths[len(bc)] += 1

        # Check for fallback breadcrumbs (just ["Clockify Help Center", title])
        if len(bc) == 2 and bc[0] == "Clockify Help Center":
            fallback_count += 1

        # Count by category (second element if exists)
        if len(bc) > 1:
            category_chunks[bc[1]] += 1

        # Count by subcategory (third element if exists)
        if len(bc) > 2:
            subcategory_chunks[f"{bc[1]} > {bc[2]}"] += 1

    return {
        "total_chunks": len(chunks),
        "breadcrumb_lengths": dict(breadcrumb_lengths),
        "fallback_breadcrumbs": {
            "count": fallback_count,
            "percentage": round(100 * fallback_count / len(chunks), 1) if chunks else 0
        },
        "categories": dict(sorted(category_chunks.items(), key=lambda x: x[1], reverse=True)),
        "subcategories_sample": dict(sorted(subcategory_chunks.items(), key=lambda x: x[1], reverse=True)[:10]),
        "unique_categories": len(category_chunks),
        "unique_subcategories": len(subcategory_chunks),
    }


def analyze_sections(chunks: List[Dict]) -> Dict:
    """Analyze section hierarchy and content distribution."""
    section_stats = defaultdict(lambda: {"chunks": 0, "total_tokens": 0})

    for chunk in chunks:
        section = chunk.get("section", "Unknown")
        tokens = chunk.get("tokens", 0)
        section_stats[section]["chunks"] += 1
        section_stats[section]["total_tokens"] += tokens

    # Sort by chunk count
    sorted_sections = sorted(section_stats.items(), key=lambda x: x[1]["chunks"], reverse=True)

    return {
        "total_sections": len(section_stats),
        "top_sections": [
            {
                "section": sec,
                "chunks": stats["chunks"],
                "total_tokens": stats["total_tokens"],
                "avg_tokens": round(stats["total_tokens"] / stats["chunks"], 1) if stats["chunks"] > 0 else 0
            }
            for sec, stats in sorted_sections[:15]
        ]
    }


def analyze_tokens(chunks: List[Dict]) -> Dict:
    """Analyze token distribution and efficiency."""
    tokens = [chunk.get("tokens", 0) for chunk in chunks]
    tokens = [t for t in tokens if t > 0]  # Filter out zeros

    if not tokens:
        return {
            "total_tokens": 0,
            "avg_tokens": 0,
            "median_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0
        }

    tokens_sorted = sorted(tokens)
    total = sum(tokens)

    return {
        "total_tokens": total,
        "chunks_analyzed": len(tokens),
        "avg_tokens": round(total / len(tokens), 1),
        "median_tokens": round(tokens_sorted[len(tokens) // 2], 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "p95_tokens": round(tokens_sorted[int(0.95 * len(tokens))], 1),
    }


def analyze_urls(chunks: List[Dict]) -> Dict:
    """Analyze URL distribution and unique sources."""
    urls = Counter()
    unique_urls = set()

    for chunk in chunks:
        url = chunk.get("url", "")
        if url:
            urls[url] += 1
            unique_urls.add(url)

    return {
        "total_chunks": len(chunks),
        "unique_urls": len(unique_urls),
        "avg_chunks_per_url": round(len(chunks) / len(unique_urls), 2) if unique_urls else 0,
        "top_urls_by_chunk_count": [
            {"url": url, "chunks": count}
            for url, count in urls.most_common(10)
        ]
    }


def analyze_chunk_types(chunks: List[Dict]) -> Dict:
    """Analyze chunk node types (parent vs child)."""
    node_types = Counter()

    for chunk in chunks:
        node_type = chunk.get("node_type", "unknown")
        node_types[node_type] += 1

    return dict(node_types)


def main():
    """Run comprehensive coverage audit."""
    import argparse

    parser = argparse.ArgumentParser(description="Coverage audit for RAG chunks")
    parser.add_argument("--namespace", default="clockify", help="Namespace to audit")
    parser.add_argument("--output", help="Output JSON file (default: print to stdout)")
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    try:
        print(f"Loading chunks for namespace: {args.namespace}...", file=sys.stderr)
        chunks = load_chunks(args.namespace)
        print(f"✓ Loaded {len(chunks)} chunks", file=sys.stderr)

        # Run all analyses
        print("Running analyses...", file=sys.stderr)
        audit = {
            "namespace": args.namespace,
            "total_chunks": len(chunks),
            "breadcrumbs": analyze_breadcrumbs(chunks),
            "sections": analyze_sections(chunks),
            "tokens": analyze_tokens(chunks),
            "urls": analyze_urls(chunks),
            "chunk_types": analyze_chunk_types(chunks),
        }

        # Output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(audit, f, indent=2)
            print(f"✓ Audit saved to {args.output}", file=sys.stderr)

        if args.summary or not args.output:
            print("\n" + "="*60, file=sys.stderr)
            print(f"COVERAGE AUDIT - {args.namespace.upper()}", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(f"Total chunks: {audit['total_chunks']}", file=sys.stderr)
            print(f"Unique URLs: {audit['urls']['unique_urls']}", file=sys.stderr)
            print(f"Unique categories: {audit['breadcrumbs']['unique_categories']}", file=sys.stderr)
            print(f"Unique subcategories: {audit['breadcrumbs']['unique_subcategories']}", file=sys.stderr)
            print(f"Total tokens: {audit['tokens']['total_tokens']}", file=sys.stderr)
            print(f"Avg tokens/chunk: {audit['tokens']['avg_tokens']}", file=sys.stderr)
            print(f"\nBreadcrumb health:", file=sys.stderr)
            print(f"  - Full breadcrumbs (3-element): {audit['breadcrumbs']['breadcrumb_lengths'].get(3, 0)} chunks", file=sys.stderr)
            print(f"  - Partial breadcrumbs (2-element): {audit['breadcrumbs']['breadcrumb_lengths'].get(2, 0)} chunks", file=sys.stderr)
            print(f"  - Fallbacks: {audit['breadcrumbs']['fallback_breadcrumbs']['percentage']}%", file=sys.stderr)
            print(f"\nTop categories by chunk count:", file=sys.stderr)
            for i, (cat, count) in enumerate(list(audit['breadcrumbs']['categories'].items())[:5], 1):
                print(f"  {i}. {cat}: {count} chunks", file=sys.stderr)
            print(f"\nTop sections by chunk count:", file=sys.stderr)
            for sec in audit['sections']['top_sections'][:3]:
                print(f"  - {sec['section']}: {sec['chunks']} chunks ({sec['avg_tokens']} tokens avg)", file=sys.stderr)
            print(f"\nChunk types:", file=sys.stderr)
            for node_type, count in audit['chunk_types'].items():
                print(f"  - {node_type}: {count}", file=sys.stderr)

        # Output JSON to stdout if no file specified and not summary-only
        if not args.output and not args.summary:
            json.dump(audit, sys.stdout, indent=2)

        print("✓ Coverage audit complete", file=sys.stderr)

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## src/chunk.py

```
#!/usr/bin/env python3
"""Parent-child semantic chunking with token-based packing."""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from src.chunkers.clockify import parse_clockify_html

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_CLEAN_DIR = Path("data/clean")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CHUNK_TOKENS = int(os.getenv("PARENT_CHUNK_TOKENS", "3200"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP_TOKENS", "240"))
CHILD_CHUNK_TOKENS = int(os.getenv("CHILD_CHUNK_TOKENS", "640"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP_TOKENS", "140"))
CHILD_CHUNK_MIN = int(os.getenv("CHILD_CHUNK_MIN_TOKENS", "100"))


class TokenCounter:
    """Simple token estimation."""

    @staticmethod
    def count(text: str) -> int:
        """Estimate tokens roughly as word count."""
        return max(1, len(text.split()))


class ParentChildChunker:
    """Create parent nodes (sections) and child nodes (chunks within sections)."""

    @staticmethod
    def split_by_headers(text: str) -> List[tuple[str, int]]:
        """Split by H2/H3 boundaries into sections."""
        sections = []
        current = []
        depth = 0

        for line in text.split("\n"):
            if re.match(r"^##\s", line):
                if current:
                    sections.append(("\n".join(current), depth))
                    current = []
                current.append(line)
                depth = 2
            elif re.match(r"^###\s", line):
                if depth >= 3 and current:
                    sections.append(("\n".join(current), 3))
                    current = []
                current.append(line)
                depth = 3
            else:
                current.append(line)

        if current:
            sections.append(("\n".join(current), depth))

        return sections

    @staticmethod
    def pack_by_tokens(text: str, target: int, overlap: int) -> List[str]:
        """Pack text into chunks of ~target tokens with overlap."""
        if TokenCounter.count(text) <= target:
            return [text]

        chunks = []
        words = text.split()
        current = []
        overlap_buf = []

        for i, word in enumerate(words):
            current.append(word)
            if TokenCounter.count(" ".join(current)) >= target:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)

                # Overlap
                overlap_words = []
                for w in reversed(current):
                    overlap_words.insert(0, w)
                    if TokenCounter.count(" ".join(overlap_words)) >= overlap:
                        break

                current = overlap_words

        if current and TokenCounter.count(" ".join(current)) >= CHILD_CHUNK_MIN:
            chunks.append(" ".join(current))

        return chunks


class ChunkProcessor:
    """Process markdown files into parent-child chunks."""

    def __init__(self):
        self.chunk_id = 0
        self.parent_id = 0
        self.all_chunks = []

    def process_file(self, md_file: Path, namespace: str) -> List[Dict[str, Any]]:
        """Process a single markdown file."""
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
            if not match:
                logger.warning(f"⊘ Invalid frontmatter: {md_file}")
                return []

            try:
                fm = json.loads(match.group(1))
            except Exception:
                logger.warning(f"⊘ Bad frontmatter: {md_file}")
                return []

            body = match.group(2).strip()
            url = fm.get("url", "")
            title = fm.get("title", md_file.stem)
            breadcrumb = fm.get("breadcrumb", [])
            updated_at = fm.get("updated_at")
            section_meta = fm.get("sections", [])

            # Namespace-specific handling
            if namespace == "clockify":
                raw_html_path = fm.get("raw_html_path")
                if not raw_html_path:
                    logger.warning(f"⊘ Missing raw_html_path for {md_file}")
                    return []

                raw_path = Path(raw_html_path)
                if not raw_path.exists():
                    logger.warning(f"⊘ Raw HTML path missing: {raw_path}")
                    return []

                try:
                    raw_payload = json.loads(raw_path.read_text())
                except Exception as exc:
                    logger.error(f"✗ Failed to load raw HTML {raw_path}: {exc}")
                    return []

                html = raw_payload.get("html", "")
                parsed_sections = parse_clockify_html(
                    html,
                    url=url,
                    title=title,
                    breadcrumb=breadcrumb,
                    updated_at=updated_at,
                )

                section_headers = [title]
                for meta_item in section_meta:
                    if isinstance(meta_item, dict):
                        candidate = meta_item.get("title")
                        if candidate and candidate not in section_headers:
                            section_headers.append(candidate)

                chunks_created = []
                for sec_idx, (doc, meta) in enumerate(parsed_sections):
                    section_text = doc.get("text", "")
                    if not section_text:
                        continue

                    anchor = meta.get("anchor")
                    section_title = meta.get("section", title)
                    if section_title and section_title not in section_headers:
                        section_headers.append(section_title)

                    parent_title_path = breadcrumb[:] if breadcrumb else []
                    if section_title:
                        if not parent_title_path or parent_title_path[-1] != section_title:
                            parent_title_path = parent_title_path + [section_title]
                    else:
                        if not parent_title_path:
                            parent_title_path = [title]

                    parent = {
                        "id": self.parent_id,
                        "url": url,
                        "namespace": namespace,
                        "title": title,
                        "headers": section_headers,
                        "section_index": sec_idx,
                        "text": section_text,
                        "tokens": TokenCounter.count(section_text),
                        "section": section_title,
                        "anchor": anchor,
                        "breadcrumb": breadcrumb,
                        "updated_at": updated_at,
                        "title_path": parent_title_path,
                    }

                    parent_id = self.parent_id
                    self.parent_id += 1

                    child_chunks = ParentChildChunker.pack_by_tokens(
                        section_text, CHILD_CHUNK_TOKENS, CHILD_CHUNK_OVERLAP
                    )

                    for ch_idx, child_text in enumerate(child_chunks):
                        child_tokens = TokenCounter.count(child_text)
                        if child_tokens < CHILD_CHUNK_MIN:
                            continue

                        chunk_uid = hashlib.md5(
                            f"{url}|{anchor or sec_idx}|{ch_idx}".encode("utf-8")
                        ).hexdigest()

                        title_path = breadcrumb[:] if breadcrumb else []
                        if section_title:
                            if not title_path or title_path[-1] != section_title:
                                title_path = title_path + [section_title]
                        else:
                            if not title_path:
                                title_path = [title]

                        child = {
                            "id": self.chunk_id,
                            "chunk_id": chunk_uid,
                            "parent_id": parent_id,
                            "url": url,
                            "namespace": namespace,
                            "title": title,
                            "headers": section_headers,
                            "section_index": sec_idx,
                            "chunk_index": ch_idx,
                            "text": child_text,
                            "tokens": child_tokens,
                            "node_type": "child",
                            "section": section_title,
                            "anchor": anchor,
                            "breadcrumb": breadcrumb,
                            "updated_at": updated_at,
                            "title_path": title_path,
                        }

                        chunks_created.append(child)
                        self.all_chunks.append(child)
                        self.chunk_id += 1

                logger.info(f"✓ {namespace}/{md_file.stem}: {len(chunks_created)} chunks")
                return chunks_created

            # Default markdown-based processing (legacy namespaces)
            headers = [fm.get("h1", "")] + fm.get("h2", [])
            headers = [h for h in headers if h]

            sections = ParentChildChunker.split_by_headers(body)
            chunks_created = []

            for sec_idx, (sec_text, _) in enumerate(sections):
                parent_tokens = TokenCounter.count(sec_text)
                parent = {
                    "id": self.parent_id,
                    "url": url,
                    "namespace": namespace,
                    "title": title,
                    "headers": headers,
                    "section_index": sec_idx,
                    "text": sec_text,
                    "tokens": parent_tokens,
                }

                parent_id = self.parent_id
                self.parent_id += 1

                child_chunks = ParentChildChunker.pack_by_tokens(
                    sec_text, CHILD_CHUNK_TOKENS, CHILD_CHUNK_OVERLAP
                )

                for ch_idx, child_text in enumerate(child_chunks):
                    child_tokens = TokenCounter.count(child_text)
                    if child_tokens < CHILD_CHUNK_MIN:
                        continue

                    chunk_uid = hashlib.md5(
                        f"{url}|{sec_idx}|{ch_idx}".encode("utf-8")
                    ).hexdigest()

                    child = {
                        "id": self.chunk_id,
                        "chunk_id": chunk_uid,
                        "parent_id": parent_id,
                        "url": url,
                        "namespace": namespace,
                        "title": title,
                        "headers": headers,
                        "section_index": sec_idx,
                        "chunk_index": ch_idx,
                        "text": child_text,
                        "tokens": child_tokens,
                        "node_type": "child",
                    }

                    chunks_created.append(child)
                    self.all_chunks.append(child)
                    self.chunk_id += 1

            logger.info(f"✓ {namespace}/{md_file.stem}: {len(chunks_created)} chunks")
            return chunks_created

        except Exception as e:
            logger.error(f"✗ Error processing {md_file}: {e}")
            return []

    def save_chunks(self, namespace: str):
        """Save chunks to JSONL per namespace."""
        ns_chunks = [c for c in self.all_chunks if c.get("namespace") == namespace]
        if not ns_chunks:
            return

        output_file = CHUNKS_DIR / f"{namespace}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in ns_chunks:
                f.write(json.dumps(chunk) + "\n")

        logger.info(f"✓ Saved {len(ns_chunks)} chunks to {output_file}")


async def main():
    """Process all markdown files."""
    logger.info(f"Starting chunking from {DATA_CLEAN_DIR}")

    processor = ChunkProcessor()
    namespaces = set()

    for ns_dir in DATA_CLEAN_DIR.glob("*"):
        if ns_dir.is_dir():
            md_files = list(ns_dir.glob("*.md"))
            logger.info(f"Found {len(md_files)} files in {ns_dir.name}")

            for md_file in sorted(md_files):
                chunks = processor.process_file(md_file, ns_dir.name)
                namespaces.add(ns_dir.name)

    for ns in namespaces:
        processor.save_chunks(ns)

    logger.info(f"✓ Chunking complete: {len(processor.all_chunks)} total chunks")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## src/llm/local_client.py

```
#!/usr/bin/env python3
"""Local LLM client with mock mode and Harmony chat format support."""

import httpx
import json
import time
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.llm.harmony_encoder import get_harmony_encoder

logger = logging.getLogger(__name__)


# P5: Normalize model names to accept multiple formats (oss20b, gpt-oss:20b)
def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to support multiple naming conventions.

    Supports:
    - "oss20b" -> "oss20b" (native Ollama format)
    - "gpt-oss:20b" -> "oss20b" (GPT-style format alias)
    - Other formats -> returned as-is

    Args:
        model_name: Model name in any supported format

    Returns:
        Normalized model name
    """
    if not model_name:
        return "oss20b"

    model_name = str(model_name).strip()

    # Map aliases to canonical names
    model_aliases = {
        "gpt-oss:20b": "oss20b",
        "gpt-oss:13b": "oss13b",
        "gpt-oss:7b": "oss7b",
        "gpt-4": "gpt-4",  # Support GPT-4 if using OpenAI
    }

    return model_aliases.get(model_name, model_name)

class LocalLLMClient:
    """Client for Ollama or OpenAI-compatible LLM endpoint with optional mock mode."""

    def __init__(
        self,
        base_url: str = None,
        model_name: str = "oss20b",
        timeout: int = 60,
        max_retries: int = 3,
        mock_mode: bool = None,
        api_type: str = "ollama",  # "ollama" or "openai"
    ):
        """Initialize LLM client.

        Args:
            base_url: LLM server endpoint (uses env var if None)
            model_name: Model name to use
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            mock_mode: Mock mode flag (None = auto-detect)
            api_type: "ollama" or "openai" - endpoint format
        """
        # Use environment variable if base_url not provided
        if base_url is None:
            base_url = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")

        self.base_url = base_url
        # P5: Normalize model name to support multiple formats
        self.model_name = normalize_model_name(model_name)
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_type = api_type

        # Harmony chat format support for gpt-oss:20b optimal performance
        self.harmony_encoder = get_harmony_encoder(self.model_name)

        # Set endpoint based on API type
        if api_type == "ollama":
            # Ollama API: /api/chat
            self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        else:
            # OpenAI-compatible: /v1/chat/completions
            self.endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # Determine mock mode
        if mock_mode is None:
            # Auto-detect: check if LLM is running
            self.mock_mode = not self._check_connection_silent()
        else:
            self.mock_mode = mock_mode

        mode_str = "MOCK_MODE" if self.mock_mode else "PRODUCTION_MODE"
        api_str = f"({api_type})"
        harmony_str = " [Harmony enabled]" if self.harmony_encoder.use_harmony else ""
        logger.info(f"🚀 LLM Client initialized in {mode_str} {api_str}{harmony_str}")
        logger.info(f"   Endpoint: {self.endpoint}")
        logger.info(f"   Model: {self.model_name}")

    def _check_connection_silent(self) -> bool:
        """Silently check if LLM is available."""
        try:
            response = httpx.post(
                self.endpoint,
                json={"model": self.model_name, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def test_connection(self) -> bool:
        """Test connection to LLM server (or mock mode).

        Returns:
            True if connected/mocked, False otherwise
        """
        if self.mock_mode:
            logger.info("✅ Mock LLM ready")
            return True

        try:
            logger.info(f"Testing connection to {self.base_url}")
            response = httpx.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Say 'OK'"}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                logger.info("✅ LLM connection successful")
                return True
            else:
                logger.error(f"❌ LLM returned status {response.status_code}")
                return False
        except httpx.ConnectError:
            logger.error(f"❌ Cannot connect to {self.base_url}")
            logger.error("   Make sure LLM is running (e.g., ollama serve)")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False

    def _generate_mock_response(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_context: str = "",
    ) -> str:
        """Generate a mock response for testing.

        Args:
            system_prompt: System prompt
            user_prompt: User question
            retrieved_context: Context from retrieval

        Returns:
            Mock response text
        """
        # Template-based mock responses
        if "create" in user_prompt.lower() and "project" in user_prompt.lower():
            return """Based on the Clockify documentation:

To create a project in Clockify:
1. Log in to your Clockify account
2. Click on the "Projects" tab
3. Click the "Create new project" button
4. Enter a project name
5. (Optional) Add a client name
6. (Optional) Set billable rates and budget
7. Click "Save" to create the project

Once created, you can assign team members to the project and start tracking time against it.

[Source: Clockify Project Management Guide]"""

        elif "report" in user_prompt.lower() and "timesheet" in user_prompt.lower():
            return """Based on the Clockify documentation:

To generate a timesheet report in Clockify:
1. Navigate to the "Reports" section
2. Select "Timesheet" from the report type dropdown
3. Choose the date range you want to report on
4. Select team members or leave blank for all
5. Click "Generate Report"
6. Export to Excel or PDF using the export buttons

The timesheet report shows total hours worked, breaks, billable time, and project allocation for the selected period.

[Source: Clockify Reporting Guide]"""

        elif "integration" in user_prompt.lower():
            return """Based on the Clockify documentation:

Clockify integrates with many popular tools:
- **Project Management:** Jira, Asana, Monday.com, ClickUp
- **Communication:** Slack, Microsoft Teams
- **Calendar:** Google Calendar, Outlook
- **Development:** GitHub, GitLab, Bitbucket
- **Productivity:** Google Sheets, Zapier

You can find and configure integrations in your account settings under "Integrations". Each integration can be customized with authentication tokens and mapping preferences.

[Source: Clockify Integrations Documentation]"""

        else:
            # Generic response template
            return f"""Based on the Clockify documentation:

{user_prompt.strip().rstrip('?')}:

Clockify provides a comprehensive time tracking solution. The feature you're asking about is available through our web interface, desktop app, and mobile applications. For detailed instructions, please refer to our help documentation or contact our support team.

Configuration and customization options are available in your account settings.

[Source: Clockify Support Documentation]"""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        top_p: float = 0.9,
        retrieved_context: str = "",
        developer_instructions: Optional[str] = None,
        reasoning_effort: str = "low",
    ) -> Optional[str]:
        """Generate response from LLM or mock, with Harmony format support.

        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            retrieved_context: Context from retrieval (for mock)
            developer_instructions: RAG-specific instructions (Harmony Developer role)
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG)

        Returns:
            Generated text or None if failed
        """
        if self.mock_mode:
            logger.debug("📝 Generating mock response")
            return self._generate_mock_response(system_prompt, user_prompt, retrieved_context)

        # Try Harmony format first if available
        prefill_ids, stop_ids = self.harmony_encoder.render_messages(
            system_prompt=system_prompt,
            developer_instructions=developer_instructions,
            user_message=user_prompt,
            reasoning_effort=reasoning_effort,
        )

        # Use Harmony prefill if available, otherwise fall back to standard messages
        if prefill_ids:
            # For Harmony prefill: pass token IDs directly (Ollama/vLLM compatible)
            # This is experimental; may need adjustment based on server support
            messages = None  # Harmony uses prefill_ids instead
            logger.debug("Using Harmony chat format")
        else:
            # Standard OpenAI format fallback
            messages = self.harmony_encoder.encoding.build_messages_standard(
                system_prompt=system_prompt,
                user_message=user_prompt,
            ) if hasattr(self.harmony_encoder.encoding, 'build_messages_standard') else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            logger.debug("Using standard chat format")

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling LLM (attempt {attempt + 1}/{self.max_retries})")

                # Build request body based on API type
                if self.api_type == "ollama":
                    # Ollama API format
                    payload = {
                        "model": self.model_name,
                        "stream": False,
                        "temperature": temperature,
                    }
                    if messages is not None:
                        payload["messages"] = messages
                    elif prefill_ids:
                        # For Harmony: pass prefill tokens if supported
                        # Fallback: use standard messages if server doesn't support prefill
                        payload["messages"] = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    # Note: Ollama might not support max_tokens, but we add it
                else:
                    # OpenAI-compatible format
                    payload = {
                        "model": self.model_name,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": False,
                    }
                    if messages is not None:
                        payload["messages"] = messages
                    # Add Harmony stop tokens if available (Harmony-aware servers will respect them)
                    if stop_ids:
                        payload["stop"] = stop_ids

                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                    verify=False,  # For self-signed certs from company
                )

                if response.status_code == 200:
                    data = response.json()

                    # Parse response based on API type
                    if self.api_type == "ollama":
                        # Ollama response: {"message": {"role": "assistant", "content": "..."}}
                        if "message" in data:
                            answer = data["message"].get("content", "")
                        elif "choices" in data:
                            # Some Ollama versions use choices format
                            answer = data["choices"][0].get("message", {}).get("content", "")
                        else:
                            logger.warning("Unexpected Ollama response format")
                            return None
                    else:
                        # OpenAI-compatible response: {"choices": [{"message": {"content": "..."}}]}
                        if "choices" in data and len(data["choices"]) > 0:
                            answer = data["choices"][0].get("message", {}).get("content", "")
                        else:
                            logger.warning("Unexpected response format from LLM")
                            return None

                    logger.debug(f"LLM response: {answer[:100]}...")
                    return answer
                else:
                    logger.warning(f"LLM returned status {response.status_code}")
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    continue

            except httpx.TimeoutException:
                logger.warning(f"LLM request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            except httpx.ConnectError:
                logger.error("Cannot connect to LLM server")
                return None
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue

        logger.error(f"Failed to get LLM response after {self.max_retries} attempts")
        return None

    def generate_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """Generate response from LLM with streaming.

        Yields:
            Chunks of generated text
        """
        if self.mock_mode:
            # For mock mode, yield the response in chunks
            response = self._generate_mock_response(system_prompt, user_prompt)
            # Yield in ~50 character chunks to simulate streaming
            for i in range(0, len(response), 50):
                yield response[i:i + 50]
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            with httpx.stream(
                "POST",
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": True,
                },
                timeout=self.timeout,
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str and data_str != "[DONE]":
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    continue
                else:
                    logger.error(f"LLM streaming returned status {response.status_code}")
                    yield f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"LLM streaming failed: {str(e)}")
            yield f"Error: {str(e)}"


# Singleton instance
_client: Optional[LocalLLMClient] = None


def get_llm_client(
    base_url: str = None,
    model_name: str = None,
    mock_mode: bool = None,
    api_type: str = None,
) -> LocalLLMClient:
    """Get or create singleton LLM client.

    Args:
        base_url: LLM server endpoint (uses env var if None)
        model_name: Model name to use (uses env var if None)
        mock_mode: Mock mode flag (uses env var if None)
        api_type: "ollama" or "openai" (uses env var if None)

    Returns:
        LocalLLMClient instance
    """
    global _client
    if _client is None:
        # Use environment variables as defaults
        if base_url is None:
            base_url = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")
        if model_name is None:
            model_name = os.getenv("LLM_MODEL", "oss20b")
        if api_type is None:
            api_type = os.getenv("LLM_API_TYPE", "ollama")
        if mock_mode is None:
            mock_env = os.getenv("MOCK_LLM", "auto").lower()
            if mock_env == "auto":
                mock_mode = None  # Auto-detect
            else:
                mock_mode = mock_env in ("true", "1", "yes")

        _client = LocalLLMClient(
            base_url=base_url,
            model_name=model_name,
            mock_mode=mock_mode,
            api_type=api_type,
        )
    return _client


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Test in mock mode
    print("\n" + "=" * 80)
    print("Testing Mock LLM Client")
    print("=" * 80 + "\n")

    client = LocalLLMClient(mock_mode=True)

    if client.test_connection():
        print("✅ Mock LLM client ready for use\n")
    else:
        print("❌ Mock LLM client failed\n")
        exit(1)
```

## src/server.py

```
from __future__ import annotations

import os
import time
import json
import re
import random
import hmac
from pathlib import Path
from uuid import uuid4
from typing import Optional, Dict, List, Tuple, Any, TypedDict
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from src.llm_client import LLMClient, close_http_client
from src.embeddings import embed_query, encode_weighted_variants, warmup
from src.query_expand import expand_structured
from src.rerank import rerank, warmup_reranker, is_available as reranker_available
from src.cache import init_cache, get_cache
from src.search_improvements import detect_query_type, get_adaptive_k_multiplier, log_query_analysis, should_enable_hybrid_search
from src.query_optimizer import get_optimizer
from src.scoring import get_scorer
from src.retrieval_engine import RetrievalEngine, RetrievalConfig, RetrievalStrategy
from src.query_decomposition import decompose_query, is_multi_intent_query
from src.models import ResponseMetadata, DecompositionMetadata
from src.metrics import track_request, get_metrics, get_content_type, track_circuit_breaker
from src.circuit_breaker import get_all_circuit_breakers
from src.citation_validator import validate_citations, format_validation_report
from src.errors import CircuitOpenError
from src.index_manager import IndexManager, NamespaceIndex
from src.performance_tracker import get_performance_tracker
from src.prompt import RAGPrompt
from src.config import CONFIG

API_TOKEN = os.getenv("API_TOKEN", "change-me")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "7000"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "20"))  # Increased from 5 to 20 for better recall

# CI support: Allow RAG_INDEX_ROOT env var to override default index location (for test fixtures)
INDEX_ROOT = Path(os.getenv("RAG_INDEX_ROOT", "index/faiss"))


def _derive_namespaces(index_root: Path) -> List[str]:
    """
    Auto-derive namespaces from subdirectories containing meta.json.

    If NAMESPACES env var is not set, scans INDEX_ROOT for subdirectories
    with meta.json files and uses those as namespaces. This prevents CI
    failures when only a subset of namespaces are available in test fixtures.

    Args:
        index_root: Root path to search for namespace directories

    Returns:
        Sorted list of namespace names found
    """
    candidates: List[str] = []
    if index_root.exists() and index_root.is_dir():
        for p in index_root.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                candidates.append(p.name)
    return sorted(candidates)


# Namespace configuration: explicit env var or auto-derived from index structure
NAMESPACES = (
    [s.strip() for s in os.getenv("NAMESPACES", "").split(",") if s.strip()]
    or _derive_namespaces(INDEX_ROOT)
    or ["clockify"]  # Final fallback for backwards compatibility
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434")
INDEX_MODE = os.getenv("INDEX_MODE", "single")

# FIX CRITICAL #3: CORS configuration with explicit origins (no wildcards)
# Default to localhost:8080 and 127.0.0.1:8080 for local development
# Production deployments should set CORS_ALLOWED_ORIGINS env var to explicit domains
_default_cors_origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://10.127.0.192:8080",
    "http://10.127.0.192:7001",
    "http://ai.coingdevelopment.com:8080",
    "http://ai.coingdevelopment.com:7001",
]
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _cors_env:
    # Parse comma-separated list of origins
    CORS_ALLOWED_ORIGINS = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    logger.info(f"CORS origins from env: {CORS_ALLOWED_ORIGINS}")
else:
    CORS_ALLOWED_ORIGINS = _default_cors_origins
    logger.info(f"CORS using default origins: {CORS_ALLOWED_ORIGINS}")

MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

# PHASE 2 REFACTOR: Global index manager (initialized in _startup)
index_manager: Optional[IndexManager] = None


async def _startup() -> None:
    """Initialize embeddings and FAISS index on startup with validation."""
    logger.info("RAG System startup: validating index, seeding randomness, warming up embedding model...")

    # Prod guard: API_TOKEN must not be "change-me" in production (AXIOM 0)
    ENV = os.getenv("ENV", "dev")
    if ENV == "prod" and API_TOKEN == "change-me":
        logger.error("API_TOKEN must not be 'change-me' in production")
        raise RuntimeError("Invalid production config: API_TOKEN not configured")

    # PREBUILT INDEX VALIDATION: Ensure index files exist and metadata is valid
    logger.info(f"Validating prebuilt index for namespaces: {NAMESPACES}")
    for ns in NAMESPACES:
        root = INDEX_ROOT / ns
        idx_path_faiss = root / "index.faiss"
        idx_path_bin = root / "index.bin"
        meta_path = root / "meta.json"

        # Check index file exists
        if not idx_path_faiss.exists() and not idx_path_bin.exists():
            raise RuntimeError(
                f"\n❌ STARTUP FAILURE: Missing prebuilt index for namespace '{ns}'\n"
                f"   Expected: {idx_path_faiss} or {idx_path_bin}\n"
                f"   Fix: Run 'make ingest' to build the FAISS index before deployment"
            )

        # Check metadata exists
        if not meta_path.exists():
            raise RuntimeError(
                f"\n❌ STARTUP FAILURE: Missing metadata for namespace '{ns}'\n"
                f"   Expected: {meta_path}\n"
                f"   Fix: Run 'make ingest' to build the FAISS index before deployment"
            )

        # Validate metadata format and model/dimension
        try:
            meta_data = json.loads(meta_path.read_text())
            meta_model = meta_data.get("model")
            meta_dim = meta_data.get("dim") or meta_data.get("dimension")

            logger.info(f"  Namespace '{ns}': model={meta_model}, dim={meta_dim}, vectors={meta_data.get('num_vectors', '?')}")

            if not meta_dim or meta_dim <= 0:
                raise ValueError(f"Invalid embedding dimension in metadata: {meta_dim}")

        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"\n❌ STARTUP FAILURE: Invalid metadata for namespace '{ns}': {e}\n"
                f"   File: {meta_path}\n"
                f"   Fix: Ensure meta.json is valid JSON with 'dim' field"
            )

    # Validate embedding dimension against local encoder
    try:
        probe = embed_query("clockify help health-check")
        encoder_dim = probe.shape[1]
        logger.info(f"✓ Embedding encoder ready: {EMBEDDING_MODEL} (dim={encoder_dim})")

        for ns in NAMESPACES:
            meta_data = json.loads((INDEX_ROOT / ns / "meta.json").read_text())
            meta_dim = meta_data.get("dim") or meta_data.get("dimension", 768)
            if meta_dim != encoder_dim:
                raise RuntimeError(
                    f"\n❌ STARTUP FAILURE: Embedding dimension mismatch for namespace '{ns}'\n"
                    f"   Index built with: dim={meta_dim}\n"
                    f"   Encoder provides: dim={encoder_dim}\n"
                    f"   Fix: Rebuild index with the current EMBEDDING_MODEL or update environment"
                )
    except Exception as e:
        raise RuntimeError(
            f"\n❌ STARTUP FAILURE: Embedding encoder check failed for model '{EMBEDDING_MODEL}'\n"
            f"   Error: {e}"
        )

    if MOCK_LLM:
        logger.info("⚠️  MOCK_LLM mode enabled: skipping live LLM probes, using mock responses")

    # PHASE 2 REFACTOR: Initialize IndexManager for thread-safe index loading
    logger.info("Initializing FAISS index manager...")
    global index_manager
    index_manager = IndexManager(INDEX_ROOT, NAMESPACES)
    index_manager.ensure_loaded()

    # Log vector counts per namespace for observability
    all_indexes = index_manager.get_all_indexes()
    for ns, entry in all_indexes.items():
        vector_count = entry["index"].ntotal
        dim = entry.get("dim", "unknown")
        logger.info(f"  ✓ Namespace '{ns}': {vector_count} vectors (dim={dim})")
    logger.info("✓ FAISS indexes loaded and cached")

    # Seed randomness for deterministic behavior (AXIOM 1)
    logger.info("Seeding randomness for deterministic retrieval...")
    random.seed(0)
    np.random.seed(0)

    # Initialize response cache for /search endpoint (80-90% latency reduction for repeats)
    init_cache()
    cache = get_cache()
    logger.info(f"✓ Response cache initialized: {cache}")

    # Log embedding backend for startup observability
    embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "model")
    if embeddings_backend == "stub":
        logger.info(f"⚠️  Embedding backend: STUB MODE (testing/development only)")
    else:
        logger.info(f"✓ Embedding backend: {embeddings_backend} ({EMBEDDING_MODEL})")

    try:
        warmup()
        logger.info("✓ Embedding model warmed up")
    except Exception as e:
        logger.error(f"Embedding warmup failed: {e}")

    # Log reranker status explicitly
    reranker_enabled = not os.getenv("RERANK_DISABLED", "false").lower() == "true"
    try:
        warmup_reranker()
        if reranker_enabled:
            logger.info("✓ Reranker model warmed up (ENABLED)")
        else:
            logger.info("⊘ Reranker warmup skipped (DISABLED via RERANK_DISABLED=true)")
    except Exception as e:
        if reranker_enabled:
            logger.warning(f"Reranker warmup failed (non-fatal, will disable reranking): {e}")
        else:
            logger.info("Reranker disabled, warmup skipped")

    logger.info("✅ RAG System startup complete: index validated, embedding ready, cache active")


def _shutdown() -> None:
    """Clean up HTTP client on FastAPI shutdown."""
    try:
        close_http_client()
    except Exception as e:
        logger.warning(f"Error closing HTTP client: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown events.

    This replaces the deprecated @app.on_event pattern with a context manager
    that handles both startup and shutdown in a single, modern FastAPI pattern.
    """
    # Startup
    await _startup()
    yield
    # Shutdown
    _shutdown()


app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

# Request logging middleware for observability
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing and status information."""
    request_id = str(uuid4())
    start_time = time.time()

    # Log incoming request
    logger.info(
        f"[{request_id}] → {request.method} {request.url.path} "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    # Process request
    response = await call_next(request)

    # Calculate duration and track metrics
    duration = time.time() - start_time

    # Track metrics (skip /metrics endpoint to avoid self-reference)
    if request.url.path != "/metrics":
        track_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            duration=duration
        )

    # Log response with timing
    logger.info(
        f"[{request_id}] ← {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )

    # Add request ID to response headers for tracing
    response.headers["X-Request-ID"] = request_id

    return response

# FIX CRITICAL #3: CORS middleware with explicit origins (no wildcards)
# Using configurable CORS_ALLOWED_ORIGINS from environment
# allow_credentials=True is only safe because origins are explicitly restricted
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-token"],
)

# --------- Retrieval Engine Management ---------
_retrieval_engine: Optional[RetrievalEngine] = None  # Lazy-initialized hybrid search engine
_retrieval_engine_lock = __import__('threading').Lock()  # Protects retrieval engine initialization

# --------- Retrieval ---------
def search_ns(ns: str, qvec: np.ndarray, k: int) -> List[Dict[str, Any]]:
    """Vector-only search via FAISS (original implementation).

    Converts FAISS L2 distances to similarity scores (0-1 scale).
    FAISS returns distances (lower=better); we convert to similarity: similarity = 1/(1+distance)
    """
    assert index_manager is not None, "IndexManager not initialized"
    entry = index_manager.get_index(ns)
    D, I = entry["index"].search(qvec, k)
    res = []
    for rank, (idx, distance) in enumerate(zip(I[0].tolist(), D[0].tolist()), start=1):
        if idx < 0:
            continue
        meta = entry["metas"][idx]
        # Convert FAISS L2 distance to similarity (0-1 scale)
        # Formula: similarity = 1 / (1 + distance)
        # This ensures: distance=0 -> similarity=1.0, distance=infinity -> similarity=0.0
        similarity = 1.0 / (1.0 + float(distance))
        # Filter out embedding field (not JSON-serializable, internal use only)
        meta_without_embedding = {k: v for k, v in meta.items() if k != "embedding"}
        res.append({
            "namespace": ns,
            "score": similarity,
            "rank": rank,
            **meta_without_embedding
        })
    return res

def search_ns_hybrid(ns: str, qvec: np.ndarray, query_text: str, k: int, alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Hybrid search combining BM25 (keyword) + vector semantic search (AXIOM 1-7)."""
    global _retrieval_engine

    # Fix #6: Thread-safe lazy initialization with double-check locking pattern
    if _retrieval_engine is None:
        with _retrieval_engine_lock:
            # Double-check: another thread may have initialized while we waited
            if _retrieval_engine is None:
                _retrieval_engine = RetrievalEngine(
                    config=RetrievalConfig(
                        strategy=RetrievalStrategy.HYBRID,
                        k_vector=min(k * 6, 100),      # Get more vector candidates before fusion
                        k_bm25=min(k * 6, 100),        # Get more BM25 candidates before fusion
                        k_final=k,
                        hybrid_alpha=alpha,
                        apply_diversity_penalty=True
                    )
                )
                logger.info("Hybrid retrieval engine initialized (BM25 + vector)")

    try:
        assert index_manager is not None, "IndexManager not initialized"
        entry = index_manager.get_index(ns)
        # PERFORMANCE FIX: Chunks now include embeddings (cached at startup)
        # No per-request reconstruction needed - embeddings are pre-populated in IndexManager._load_index_for_ns()
        chunks = entry["metas"]  # Already includes "embedding" field for each chunk

        # Extract 1D embedding if 2D array passed (for RetrievalEngine compatibility)
        embedding_1d = qvec[0] if qvec.ndim == 2 else qvec

        # Call hybrid search engine (chunks already have embeddings)
        results = _retrieval_engine.search_hybrid(
            query_embedding=embedding_1d,
            query_text=query_text,
            chunks=chunks,
            k=k,
            alpha=alpha
        )

        # Convert RetrievalResult objects to dict format (backward compatible)
        res = []
        for rank, result in enumerate(results, start=1):
            # Filter out 'embedding' from metadata (internal use only, not serializable)
            metadata_without_embedding = {
                k: v for k, v in result.metadata.items()
                if k != "embedding"
            }
            res.append({
                "namespace": ns,
                "score": float(result.hybrid_score or result.final_score or 0.0),
                "rank": rank,
                "chunk_id": result.chunk_id,
                "text": result.text,
                "title": result.title,
                "url": result.url,
                # Preserve original metadata (excluding embedding)
                **metadata_without_embedding
            })

        logger.debug(f"Hybrid search for '{query_text}' in {ns}: {len(res)}/{len(chunks)} candidates")
        return res

    except Exception as e:
        logger.warning(f"Hybrid search failed for {ns}: {e}, falling back to vector search")
        return search_ns(ns, qvec, k)

def search_with_decomposition(
    q: str, qvec: np.ndarray, k: int, ns_list: List[str], decomp_result
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve results using query decomposition for multi-intent queries.

    Per-subtask retrieval with score fusion:
    1. For each subtask, recompute intent and retrieve with appropriate strategy
    2. Convert FAISS distances to similarities (score = 1 / (1 + distance))
    3. Fuse by (url, chunk_id): best score + additive hits bonus (+0.05 per hit)
    4. Return merged results with per-doc hit mapping for analysis

    Returns:
        (merged_results, decomposition_metadata)
    """
    # Dict keyed by (url, chunk_id) storing:
    # {best_score, hits, semantic_score, subtasks_hit, payload}
    fused_docs = {}

    logger.debug(f"Starting decomposition-aware retrieval for {len(decomp_result.subtasks)} subtasks")

    for subtask_idx, subtask in enumerate(decomp_result.subtasks):
        subtask_text = subtask.text
        subtask_reason = subtask.reason
        subtask_weight = subtask.weight
        subtask_intent = subtask.intent  # May be None if detection failed

        logger.debug(
            f"Subtask {subtask_idx+1}: '{subtask_text}' "
            f"(reason={subtask_reason}, intent={subtask_intent}, weight={subtask_weight})"
        )

        try:
            # Recompute intent if not already set
            if subtask_intent is None:
                subtask_intent = detect_query_type(subtask_text)

            # Recompute hybrid flag based on subtask's own characteristics
            subtask_query_type = subtask_intent if subtask_intent else detect_query_type(subtask_text)
            should_hybrid = should_enable_hybrid_search(subtask_query_type, len(subtask_text))

            # Expand subtask with boost terms specific to this subtask (structured with weights)
            subtask_variants = expand_structured(subtask_text, boost_terms=subtask.boost_terms)
            logger.debug(f"  Expanded to {len(subtask_variants)} weighted variants")

            # Encode subtask variants using weighted averaging
            # encode_weighted_variants handles the weighting, averaging, and L2 normalization
            subtask_qvec = encode_weighted_variants(subtask_variants).flatten()

            # Retrieve with adaptive k based on subtask intent
            adaptive_k = get_adaptive_k_multiplier(subtask_query_type, k)
            raw_k = min(adaptive_k * 2, 100)

            # Retrieve per namespace for this subtask
            per_ns = {}
            if should_hybrid:
                per_ns = {
                    ns: search_ns_hybrid(ns, subtask_qvec[None,:].astype(np.float32), subtask_text, raw_k)
                    for ns in ns_list
                }
            else:
                per_ns = {
                    ns: search_ns(ns, subtask_qvec[None,:].astype(np.float32), raw_k)
                    for ns in ns_list
                }

            # Fuse across namespaces for this subtask
            subtask_results = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]

            logger.debug(f"  Retrieved {len(subtask_results)} results from {len(ns_list)} namespace(s)")

            # Process subtask results and merge into fused_docs
            for result in subtask_results:
                url = result.get("url", "")
                chunk_id = result.get("chunk_id", result.get("id", ""))
                # All scores are already normalized to [0-1] similarity range:
                # - search_ns: converts distances to similarities before returning
                # - search_ns_hybrid: returns normalized fused scores
                similarity = result.get("score", 0.0)

                doc_key = (url, chunk_id)

                if doc_key not in fused_docs:
                    # First hit for this document
                    fused_docs[doc_key] = {
                        "payload": result,
                        "best_score": similarity,
                        "semantic_score": similarity,
                        "hits": 1,
                        "subtasks_hit": [
                            {
                                "text": subtask_text,
                                "reason": subtask_reason,
                                "intent": subtask_intent,
                                "weight": subtask_weight,
                                "score": similarity,
                            }
                        ],
                    }
                else:
                    # Document already seen in another subtask
                    fused_docs[doc_key]["hits"] += 1
                    fused_docs[doc_key]["semantic_score"] += similarity  # Accumulate similarity
                    fused_docs[doc_key]["best_score"] = max(fused_docs[doc_key]["best_score"], similarity)
                    fused_docs[doc_key]["subtasks_hit"].append(
                        {
                            "text": subtask_text,
                            "reason": subtask_reason,
                            "intent": subtask_intent,
                            "weight": subtask_weight,
                            "score": similarity,
                        }
                    )

        except Exception as e:
            logger.warning(f"Error retrieving for subtask {subtask_idx+1} '{subtask_text}': {e}")
            continue

    # Compute final scores with multi-hit bonus
    for doc_key, doc_data in fused_docs.items():
        # Final score: best individual score + accumulation + hits bonus
        hits = doc_data["hits"]
        best_score = doc_data["best_score"]
        # semantic_score available in doc_data["semantic_score"] if needed for future scoring

        # Additive fusion: reward documents matching multiple subtasks
        # Score = best_score + (0.05 * (hits - 1))
        # This ensures single-hit docs aren't penalized, but multi-hit docs are boosted
        final_score = best_score + (0.05 * (hits - 1))

        # Cap final score at 1.0
        final_score = min(final_score, 1.0)

        # Update payload with fused score and metadata
        doc_data["payload"]["score"] = final_score
        doc_data["payload"]["decomposition_hits"] = hits
        doc_data["payload"]["decomposition_subtasks"] = doc_data["subtasks_hit"]

    # Sort by final score and extract payloads (filter out embeddings)
    sorted_docs = sorted(
        fused_docs.values(),
        key=lambda x: (-x["payload"]["score"], -x["hits"])
    )
    merged_list = [_remove_embedding_from_result(doc["payload"]) for doc in sorted_docs]

    logger.debug(
        f"Fused {len(fused_docs)} unique documents across {len(decomp_result.subtasks)} subtasks"
    )

    # Build metadata with per-subtask info and per-doc hit mapping
    metadata = {
        "decomposition_strategy": decomp_result.strategy,
        "llm_used": decomp_result.llm_used,
        "subtask_count": len(decomp_result.subtasks),
        "subtasks": [st.to_log_payload() for st in decomp_result.subtasks],
        "fused_docs": len(fused_docs),
        "multi_hit_docs": sum(1 for doc in fused_docs.values() if doc["hits"] > 1),
    }

    return merged_list, metadata

def _remove_embedding_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove embedding field from result dict (not serializable, internal use only)."""
    return {k: v for k, v in result.items() if k != "embedding"}

def fuse_results(by_ns: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
    scores: Dict[Tuple[str, str], float] = {}
    payloads: Dict[Tuple[str, str], Dict[str, Any]] = {}
    C = 60.0
    for ns, lst in by_ns.items():
        for r, item in enumerate(lst, start=1):
            # Remove embedding before processing
            item = _remove_embedding_from_result(item)
            key = (item.get("url",""), item.get("chunk_id", item.get("id", r)))
            scores[key] = scores.get(key, 0.0) + 1.0 / (C + r)
            payloads[key] = item
    merged = sorted(payloads.values(), key=lambda x: scores[(x.get("url",""), x.get("chunk_id", x.get("id", 0)))], reverse=True)
    return merged[:k]

# --------- Models ---------
class SearchQuery(BaseModel):
    """Validated search query parameters."""
    q: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    namespace: Optional[str] = Field(default=None, max_length=100)

class SearchResponse(BaseModel):
    """Search response with request tracing and metadata."""
    success: bool = True
    query: str
    results: List[Dict[str, Any]]
    total_results: int = 0
    latency_ms: int = 0
    request_id: str = ""
    metadata: Optional[Dict[str, Any]] = None
    query_decomposition: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Validated chat request."""
    question: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    namespace: Optional[str] = Field(default=None, max_length=100)

class ChatResponse(BaseModel):
    """Chat response with citations and grounding."""
    success: bool = True
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: Dict[str, Any]
    meta: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

# --------- Auth/limits ---------
def require_token(token: Optional[str]):
    """Verify API token using constant-time comparison (AXIOM 0).

    Always validates token in all environments. In dev mode, use API_TOKEN="change-me"
    to require that specific token. In production, set API_TOKEN to actual secret.
    """
    # Token is always required; AXIOM 0 enforcement
    if not token:
        raise HTTPException(status_code=401, detail="unauthorized")

    # Always validate token using constant-time comparison, regardless of environment
    # In dev: token must equal "change-me"
    # In prod: token must equal configured API_TOKEN
    if not hmac.compare_digest(token, API_TOKEN):
        logger.warning("Invalid token attempt")
        raise HTTPException(status_code=401, detail="unauthorized")

_last_req: Dict[str, float] = {}

def rate_limit(ip: str, min_interval: float = 0.1) -> None:
    """Rate limit: enforce minimum interval between requests per IP (AXIOM 0)."""
    now = time.time()

    # Fix #7: Cleanup old entries to prevent memory leak
    # Remove IPs that haven't been seen in 2*min_interval window
    window = 2 * min_interval
    stale_ips = [k for k, v in _last_req.items() if now - v > window]
    for ip_to_remove in stale_ips:
        del _last_req[ip_to_remove]

    t = _last_req.get(ip, 0.0)
    if now - t < min_interval:
        raise HTTPException(status_code=429, detail="rate_limited")
    _last_req[ip] = now

# --------- Routes ---------
@app.get("/health")
def health(deep: int = 0, detailed: int = 0) -> Dict[str, Any]:
    """
    Health endpoint with optional detailed statistics.

    Args:
        deep: Include deep LLM health checks (chat ping)
        detailed: Include detailed cache stats and circuit breaker metrics

    Returns:
        Health status dict with optional detailed observability data
    """
    ok = True
    try:
        if index_manager:
            index_manager.ensure_loaded()
    except Exception as e:
        ok = False
        logger.error(f"Index load error: {e}")

    # Get all indexes and compute metrics (PHASE 2 REFACTOR: use IndexManager)
    index_metrics = {}
    index_normalized = None
    if index_manager:
        all_indexes = index_manager.get_all_indexes()
        if all_indexes:
            index_normalized = all(index_manager.is_normalized(ns) for ns in all_indexes.keys())
            for ns, entry in all_indexes.items():
                index = entry["index"]
                ntotal = index.ntotal
                metas = entry["metas"]
                index_metrics[ns] = {
                    "indexed_vectors": ntotal,
                    "indexed_chunks": len(metas),
                    "vector_dim": entry.get("dim", 768),
                    "normalized": index_manager.is_normalized(ns),
                }

    # Check embedding model health
    embedding_ok = None
    embedding_details = None
    try:
        test_vec = embed_query("health check")
        embedding_ok = test_vec.shape[0] > 0 and test_vec.shape[1] > 0
        embedding_details = f"OK: {EMBEDDING_MODEL} (dim={test_vec.shape[1]})"
    except Exception as e:
        embedding_ok = False
        embedding_details = f"Error: {str(e)}"

    # Check reranker availability
    reranker_ok = reranker_available()
    reranker_details = "Available: BAAI/bge-reranker-base" if reranker_ok else "Not available (FlagEmbedding not installed or disabled)"

    # Check LLM health if not mock mode
    llm_ok = None
    llm_details = None
    llm_deep_ok = None
    llm_deep_details = None

    if not MOCK_LLM:
        try:
            llm = LLMClient()
            llm_check = llm.health_check()
            llm_ok = llm_check.get("ok")
            llm_details = llm_check.get("details")

            # Deep health check: try a lightweight chat ping
            if deep:
                try:
                    result = llm.chat([{"role": "user", "content": "ping"}], stream=False)
                    llm_deep_ok = bool(result)
                    llm_deep_details = "chat ping ok" if llm_deep_ok else "empty response"
                except Exception as e:
                    llm_deep_ok = False
                    llm_deep_details = f"chat ping failed: {str(e)}"
        except Exception as e:
            llm_ok = False
            llm_details = f"Error initializing LLM client: {str(e)}"
            llm_deep_ok = False
            llm_deep_details = "skipped due to init error"
    else:
        # In mock mode, deep checks are skipped
        llm_deep_ok = None
        llm_deep_details = None

    # Get circuit breaker status
    circuit_breakers = get_all_circuit_breakers()

    # PHASE 2 REFACTOR: Get namespaces from IndexManager
    namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
    index_normalized_by_ns = {}
    if index_manager:
        all_indexes = index_manager.get_all_indexes()
        index_normalized_by_ns = {ns: index_manager.is_normalized(ns) for ns in all_indexes.keys()}

    # PHASE 5: Get semantic cache stats for observability
    semantic_cache_stats = None
    try:
        semantic_cache_stats = get_semantic_cache().stats()
    except Exception as e:
        logger.warning(f"Failed to get semantic cache stats: {e}")

    # Build base response
    response = {
        "ok": ok,
        "namespaces": namespaces,
        "mode": "mock" if MOCK_LLM else "live",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_ok": embedding_ok,
        "embedding_details": embedding_details,
        "reranker_ok": reranker_ok,
        "reranker_details": reranker_details,
        "llm_api_type": os.getenv("LLM_API_TYPE","ollama"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-oss:20b"),
        "llm_ok": llm_ok,
        "llm_details": llm_details,
        "llm_deep_ok": llm_deep_ok,
        "llm_deep_details": llm_deep_details,
        "index_normalized": index_normalized,
        "index_normalized_by_ns": index_normalized_by_ns,
        "index_metrics": index_metrics,
        "cache": {
            "semantic_cache_stats": semantic_cache_stats,
        },
    }

    # Add detailed stats if requested
    if detailed:
        response["circuit_breakers"] = circuit_breakers

        # Add cache hit rate summary
        if semantic_cache_stats:
            response["cache_hit_rate_pct"] = semantic_cache_stats.get("hit_rate_pct", 0)
            response["cache_memory_mb"] = semantic_cache_stats.get("memory_usage_mb", 0)
            response["cache_entries"] = semantic_cache_stats.get("size", 0)
            response["cache_max_entries"] = semantic_cache_stats.get("max_size", 0)

        # Add system configuration details
        response["config"] = {
            "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "model"),
            "rerank_disabled": os.getenv("RERANK_DISABLED", "false").lower() == "true",
            "mock_llm": MOCK_LLM,
            "semantic_cache_ttl_seconds": os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"),
            "semantic_cache_max_size": os.getenv("SEMANTIC_CACHE_MAX_SIZE", "10000"),
        }
    else:
        # In non-detailed mode, only include circuit breakers for backward compatibility
        response["circuit_breakers"] = circuit_breakers

    return response

@app.get("/live")
def live() -> Dict[str, str]:
    """Liveness probe: returns 200 if process is alive (no dependencies checked)."""
    return {"status": "alive"}

@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping by monitoring systems.
    Tracks request counts, latency, cache stats, circuit breaker status, and more.
    """
    from fastapi.responses import Response

    # Update circuit breaker metrics before returning
    circuit_breakers = get_all_circuit_breakers()
    for name, status in circuit_breakers.items():
        track_circuit_breaker(
            name=name,
            state=status["state"],
            metrics_data={
                "total_requests": status["total_requests"],
                "total_failures": status["total_failures"],
                "total_successes": status["total_successes"],
                "consecutive_failures": status["consecutive_failures"],
            }
        )

    return Response(content=get_metrics(), media_type=get_content_type())

@app.get("/perf")
def perf(detailed: bool = False) -> Dict[str, Any]:
    """
    Performance metrics endpoint - TIER 2 improvement.

    Shows latency statistics across all pipeline stages.
    Useful for identifying bottlenecks and optimization opportunities.

    Args:
        detailed: If true, return recent samples for analysis

    Returns:
        Performance statistics dictionary
    """
    tracker = get_performance_tracker()

    result = tracker.get_stats()

    if detailed:
        result["recent_samples"] = tracker.get_recent_samples(limit=20)
        result["bottlenecks"] = tracker.get_bottlenecks(top_n=5)

    return result

@app.get("/ready")
def ready() -> Tuple[Dict[str, str], Optional[int]]:
    """Readiness probe: returns 200 only if index loaded and LLM ready."""
    try:
        if index_manager:
            index_manager.ensure_loaded()
        if not MOCK_LLM:
            llm = LLMClient()
            llm_check = llm.health_check()
            if not llm_check.get("ok"):
                logger.warning(f"LLM not ready: {llm_check.get('details')}")
                return {"status": "not_ready", "reason": "llm_unhealthy"}, 503
        return {"status": "ready"}, 200
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return {"status": "not_ready", "reason": str(e)}, 503

@app.get("/config")
def config(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    ENV = os.getenv("ENV", "dev")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")

    # In prod, hide llm_base_url unless admin token matches
    reveal_sensitive = (ENV != "prod") or (ADMIN_TOKEN != "change-me" and x_admin_token == ADMIN_TOKEN)

    # PHASE 5: Enhanced /config endpoint for observability
    # Includes reranker status, embedding backend, actual loaded namespaces, cache stats
    from src.embeddings import EMBEDDING_DIM
    from src.semantic_cache import get_semantic_cache

    out = {
        # Core config
        "namespaces_env": NAMESPACES,
        "actual_namespaces": list(index_manager.get_all_indexes().keys()) if index_manager else [],
        "index_mode": os.getenv("INDEX_MODE","single"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "retrieval_k": RETRIEVAL_K,
        "env": ENV,

        # Feature flags (PHASE 5)
        "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "ollama"),
        "rerank_disabled": os.getenv("RERANK_DISABLED", "false").lower() == "true",
        "streaming_enabled": os.getenv("STREAMING_ENABLED","false").lower()=="true",
        "mock_llm": MOCK_LLM,

        # LLM config
        "llm_chat_path": os.getenv("LLM_CHAT_PATH", "/api/chat"),
        "llm_tags_path": os.getenv("LLM_TAGS_PATH", "/api/tags"),
        "llm_timeout_seconds": int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        "llm_api_type": os.getenv("LLM_API_TYPE", "ollama"),

        # Cache stats (PHASE 5)
        "cache": {
            "response_cache_size": get_cache().size() if get_cache() else 0,
            "semantic_cache_stats": get_semantic_cache().stats(),
        }
    }

    if reveal_sensitive:
        out["llm_base_url"] = os.getenv("LLM_BASE_URL", "http://10.127.0.192:11434")
    else:
        out["llm_base_url"] = "<hidden>"

    return out

@app.get("/search", response_model=SearchResponse)
def search(
    q: str,
    request: Request,
    k: Optional[int] = None,
    namespace: Optional[str] = None,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition for baseline comparison"),
    x_api_token: Optional[str] = Header(default=None),
) -> SearchResponse:
    """Search with query expansion, normalized embeddings, optional reranking (AXIOM 1,3,4,5,7).

    Responses cached for repeated queries (80-90% latency reduction).
    """
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    if index_manager:
        index_manager.ensure_loaded()
    # Sanitize k: clamp to [1, 20] bounds
    k = max(1, min(int(k or RETRIEVAL_K), 20))

    t0 = time.time()

    try:
        # AXIOM 4: Query expansion with glossary synonyms
        logger.debug(f"Search query: '{q}'")

        # NEW: Detect and decompose multi-intent queries (before cache check)
        decomp_result = None
        decomp_metadata: Dict[str, Any] = {}
        cache_query = q  # Default: use original query as cache key

        if not decomposition_off and is_multi_intent_query(q):
            try:
                decomp_result = decompose_query(q)
                logger.info(f"Decomposed query into {len(decomp_result.subtasks)} subtasks "
                           f"(strategy={decomp_result.strategy})")

                # If successful decomposition with >1 subtask, modify cache key to avoid collisions
                if decomp_result and len(decomp_result.subtasks) > 1:
                    # Create unique cache key that includes subtask info
                    # Format: original_query|decomp:subtask1|subtask2|...
                    subtask_str = "|".join([st.text for st in decomp_result.subtasks[1:]])  # Skip original
                    cache_query = f"{q}|decomp:{subtask_str}"
                    logger.debug(f"Using decomposition-aware cache key for multi-intent query")
            except Exception as e:
                logger.warning(f"Query decomposition failed: {e}, continuing with standard expansion")
                decomp_result = None
        elif decomposition_off:
            logger.debug(f"Decomposition disabled via decomposition_off=true")

        # Check response cache with decomposition-aware key
        cache = get_cache()
        cached_response = cache.get(cache_query, k, namespace)
        if cached_response is not None:
            latency_ms = int((time.time() - t0) * 1000)
            logger.info(f"Search cache hit: '{q}' (decomp={decomp_result is not None and len(decomp_result.subtasks) > 1}) k={k} in {latency_ms}ms")
            return cached_response

        # Standard expansion (used for all-in-one retrieval if no decomposition)
        if not decomp_result or len(decomp_result.subtasks) <= 1:
            # Use structured expansion with weights for better embedding fusion
            variants = expand_structured(q)  # [{text, source, weight}, ...]
            logger.debug(f"Query expanded to {len(variants)} weighted variants")
        else:
            # Use original as fallback expansion
            variants = [{"text": q, "source": "original", "weight": 1.0}]
            logger.debug(f"Skipping standard expansion due to decomposition (using original query only)")

        # QUICK WIN: Detect query type for adaptive retrieval strategy
        query_type = detect_query_type(q)
        adaptive_k = get_adaptive_k_multiplier(query_type, k)
        log_query_analysis(q, query_type, adaptive_k)

        # AXIOM 3: Encode variants with weighted averaging and L2 normalization
        # encode_weighted_variants handles weighting, averaging, and normalization
        qvec = encode_weighted_variants(variants).flatten()
        # Verify normalization (AXIOM 3) but do not assert; just log
        actual_norm = np.linalg.norm(qvec)
        if not (0.98 <= actual_norm <= 1.02):
            logger.debug(f"Query vector norm={actual_norm:.6f} (expected ~1.0); reclamping")
            qvec = qvec / (actual_norm + 1e-8)

        # AXIOM 1: Determinism via deterministic namespace-based retrieval
        # Use sorted() for consistent ordering across multiple calls (PHASE 2 REFACTOR: use IndexManager)
        available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
        ns_list = [namespace] if namespace in available_namespaces else sorted(available_namespaces)

        # QUICK WIN: Adaptive k multiplier based on query type for better candidate pool
        # Retrieve with adaptive_k instead of fixed k*6 or k*3
        raw_k = min(adaptive_k, 100)  # Cap at 100 for efficiency

        # NEW: Use decomposition-aware retrieval if query was decomposed
        if decomp_result and len(decomp_result.subtasks) > 1:
            logger.info(f"Using decomposition-aware retrieval for multi-intent query")
            candidates, decomp_metadata = search_with_decomposition(
                q, qvec, raw_k, ns_list, decomp_result
            )
            # Limit to raw_k results
            candidates = candidates[:raw_k]
        else:
            # Standard retrieval (vector or hybrid)
            use_hybrid = should_enable_hybrid_search(query_type, len(q))

            # Retrieve results using hybrid or vector-only search
            if use_hybrid:
                logger.debug(f"Using hybrid search (BM25 + vector) for query type: {query_type}")
                per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), q, raw_k) for ns in ns_list}
            else:
                logger.debug(f"Using vector-only search for query type: {query_type}")
                per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}

            # Fuse and deduplicate by URL (AXIOM 4)
            candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]
        # AXIOM 1: Stable sort on candidates before dedup to ensure deterministic tie-breaking
        candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
        seen_urls = set()
        results_dedup = []
        for candidate in candidates:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results_dedup.append(candidate)
            if len(results_dedup) >= k:
                break

        # AXIOM 5: Optional reranking (silent fallback if not available)
        results = rerank(q, results_dedup, k) if results_dedup else []

        # NEW: Apply query optimization and confidence scoring
        optimizer = get_optimizer()
        scorer = get_scorer()

        # Analyze query for optimization
        query_analysis = optimizer.analyze(q)
        query_entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("type", "general")

        # Score and rank results by confidence
        if results:
            results = scorer.batch_score(results, q, query_entities, query_type)

        # Add sequential 1-based rank to each result
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Search '{q}' k={k} -> {len(results)} results (unique URLs) in {latency_ms}ms")

        request_id = str(uuid4())

        # Build ResponseMetadata with proper Pydantic objects
        decomp_meta = None
        if decomp_result and len(decomp_result.subtasks) > 1:
            decomp_meta = DecompositionMetadata(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=[st.text for st in decomp_result.subtasks],
                llm_used=getattr(decomp_result, "llm_used", False),
                fused_docs=len(results),
                multi_hit_docs=decomp_metadata.get("multi_hit_docs", 0),
            )

        response_metadata = ResponseMetadata(
            cache_hit=False,
            index_normalized=True,
            decomposition=decomp_meta,
        )

        response = {
            "success": True,
            "query": q,
            "results": results,
            "total_results": len(results),
            "latency_ms": latency_ms,
            "request_id": request_id,
            "metadata": response_metadata.model_dump(),
        }

        # Add stub mode disclaimer if using test embeddings
        embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "model")
        if embeddings_backend == "stub":
            response["metadata"]["stub_mode_disclaimer"] = (
                "⚠️  STUB MODE: Using deterministic test embeddings. "
                "Results are NOT semantically meaningful. Use only for testing/CI."
            )

        # Include legacy decomposition metadata if available (for backward compat)
        if decomp_metadata:
            response["query_decomposition"] = decomp_metadata

        # Cache the response for repeated queries (80-90% latency improvement)
        # Use decomposition-aware cache key if applicable
        cache.set(cache_query, k, response, namespace)

        return response

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition for baseline comparison"),
    x_api_token: Optional[str] = Header(default=None),
) -> ChatResponse:
    """Chat with RAG: retrieve, ground answer, cite sources (AXIOM 1,2,3,4,6,7,9)."""
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    if index_manager:
        index_manager.ensure_loaded()
    # Sanitize k: clamp to [1, 20] bounds
    k = max(1, min(int(req.k or RETRIEVAL_K), 20))

    t0 = time.time()

    try:
        # AXIOM 1,3,4: Same retrieval as /search - expanded, normalized, deduplicated
        logger.debug(f"Chat question: '{req.question}'")

        # NEW: Detect and decompose multi-intent queries for chat
        decomp_result = None
        if not decomposition_off and is_multi_intent_query(req.question):
            try:
                decomp_result = decompose_query(req.question)
                logger.info(f"Decomposed chat question into {len(decomp_result.subtasks)} subtasks "
                           f"(strategy={decomp_result.strategy})")
            except Exception as e:
                logger.warning(f"Query decomposition failed for chat: {e}, continuing with standard expansion")
        elif decomposition_off:
            logger.debug(f"Decomposition disabled via decomposition_off=true")

        # Standard expansion (used for all-in-one retrieval if no decomposition)
        if not decomp_result or len(decomp_result.subtasks) <= 1:
            # Use structured expansion with weights for better embedding fusion
            variants = expand_structured(req.question)  # [{text, source, weight}, ...]
            logger.debug(f"Query expanded to {len(variants)} weighted variants")
        else:
            # Use original as fallback expansion
            variants = [{"text": req.question, "source": "original", "weight": 1.0}]
            logger.debug(f"Skipping standard expansion due to decomposition (using original query only)")

        # Encode variants with weighted averaging and L2 normalization
        qvec = encode_weighted_variants(variants).flatten()

        # PHASE 5: Check semantic cache before expensive retrieval
        # Cache key uses query embedding fingerprint + doc IDs + prompt version
        from src.semantic_cache import get_semantic_cache
        semantic_cache = get_semantic_cache()

        # Use sorted() for deterministic namespace ordering (AXIOM 1, PHASE 2 REFACTOR)
        available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
        ns_list = [req.namespace] if req.namespace in available_namespaces else sorted(available_namespaces)
        raw_k = k * 3  # For chat, use slightly smaller retrieval set

        # NEW: Use decomposition-aware retrieval if query was decomposed
        if decomp_result and len(decomp_result.subtasks) > 1:
            logger.info(f"Using decomposition-aware retrieval for multi-intent chat question")
            candidates, _ = search_with_decomposition(
                req.question, qvec, raw_k, ns_list, decomp_result
            )
            candidates = candidates[:raw_k]
        else:
            # Standard retrieval (vector or hybrid)
            query_type = detect_query_type(req.question)
            use_hybrid = should_enable_hybrid_search(query_type, len(req.question))

            if use_hybrid:
                logger.debug(f"Using hybrid search (BM25 + vector) for chat question type: {query_type}")
                per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), req.question, raw_k) for ns in ns_list}
            else:
                logger.debug(f"Using vector-only search for chat question type: {query_type}")
                per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}

            candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]
        # AXIOM 1: Stable sort on candidates before dedup to ensure deterministic tie-breaking
        candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
        seen_urls = set()
        hits = []
        for candidate in candidates:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            hits.append(candidate)
            if len(hits) >= k:
                break

        t_retr = int((time.time() - t0) * 1000)

        # AXIOM 2,6: Build context with citations for grounding
        # Convert hits to chunk format for RAGPrompt.build_messages()
        chunks = []
        source_map = {}

        # Use more chunks for richer context (configurable via MAX_CONTEXT_CHUNKS)
        top_context = hits[: min(CONFIG.MAX_CONTEXT_CHUNKS, len(hits))]

        for i, h in enumerate(top_context, start=1):
            url = h.get("url", "")
            anchor = h.get("anchor")
            url_with_anchor = f"{url}#{anchor}" if anchor else url
            title_path = h.get("title_path") or []
            if isinstance(title_path, list) and title_path:
                title = " > ".join(title_path)
            else:
                title = h.get("title") or url or f"chunk-{i}"
            text = h.get("text", "")[:1600]
            chunk_id = h.get("chunk_id", h.get("id", str(i)))

            # Build chunk dict for RAGPrompt
            chunk = {
                "title": title,
                "url": url_with_anchor,
                "namespace": h.get("namespace"),
                "score": h.get("score", 0.0),
                "text": text,
                "chunk_id": chunk_id,
                "anchor": anchor,
            }
            chunks.append(chunk)
            source_map[str(i)] = chunk_id

        # HARMONY: Use RAGPrompt.build_messages() to get system prompt, user prompt, and developer instructions
        # reasoning_effort="low" is default for RAG to minimize latency per gpt-oss:20b best practices
        messages, sources, developer_instructions = RAGPrompt.build_messages(
            question=req.question,
            chunks=chunks,
            namespace=req.namespace or "clockify",
            max_chunks=len(chunks),
            reasoning_effort="low"  # gpt-oss:20b best practice: minimize latency
        )

        # Update source_map for citation tracking
        for i, source in enumerate(sources, start=1):
            source_map[str(i)] = source.get("chunk_id", "")

        t1 = time.time()
        # AXIOM 1: Determinism via configurable temperature (default 0.0 for strict determinism)
        temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))

        # PHASE 5: Try semantic cache before LLM call
        # Cache key: query embedding + top doc IDs + namespaces + prompt version
        top_doc_ids = [s.get("chunk_id", "") for s in sources]
        cached_result = semantic_cache.get(qvec, top_doc_ids, prompt_version="v1", namespaces=ns_list)
        cache_hit = False

        if cached_result is not None:
            answer = cached_result["answer"]
            cache_hit = True
            logger.info(f"✓ CACHE HIT for question: '{req.question[:50]}...'")
        else:
            try:
                # HARMONY: Use chat method with messages list
                llm = LLMClient()
                answer = llm.chat(
                    messages=messages,
                    max_tokens=800,
                    temperature=temp,
                    stream=False
                )
            except CircuitOpenError as e:
                # Circuit breaker is open - LLM service temporarily unavailable
                logger.error(f"Circuit breaker open for chat request: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service temporarily unavailable. Circuit breaker is open. Please try again in a minute.",
                    headers={"Retry-After": "60"}
                )

        t_llm = int((time.time() - t1) * 1000)

        # PHASE 5: Answerability check - prevent hallucination
        # Validate that answer is grounded in context (Jaccard overlap >= 0.25)
        from src.llm_client import compute_answerability_score
        # Build context string from chunk texts
        # CRITICAL: Use same char truncation that LLM saw in prompt (configurable via CONTEXT_CHAR_LIMIT)
        # If we use full text here but LLM saw truncated text, valid answers get rejected
        context_blocks = [chunk.get("text", "")[:CONFIG.CONTEXT_CHAR_LIMIT] for chunk in chunks]
        context_str = "\n".join(context_blocks)
        is_answerable, answerability_score = compute_answerability_score(answer, context_str)

        # Debug logging: always log answerability score for diagnostics
        logger.info(
            f"Answerability check: score={answerability_score:.3f}, "
            f"threshold={CONFIG.ANSWERABILITY_THRESHOLD}, passed={is_answerable}"
        )

        if not is_answerable:
            # Debug logging: save original LLM answer before replacing
            original_answer = answer
            logger.warning(
                f"Answerability check failed for chat response (score={answerability_score:.3f}). "
                f"Original LLM answer: {original_answer[:200]}... "
                f"Replacing with refusal."
            )
            # Replace with safe refusal instead of returning hallucinated answer
            answer = "I don't have enough information in the documentation to answer that confidently. Could you rephrase your question or ask about a related topic?"

        # Validate citations in response
        citation_validation = validate_citations(answer, len(sources), strict=False)
        if not citation_validation.is_valid:
            logger.warning(
                f"Citation validation failed for chat response: "
                f"missing={citation_validation.missing_citations}, "
                f"invalid={citation_validation.invalid_citations}"
            )

        # AXIOM 2: Extract and validate citations (AXIOM 9: test this grounding)
        # Safe citation parsing: strip URLs first, then extract [1], [2],...[99]
        tmp = re.sub(r'https?://\S+', '<URL>', answer)
        citations_in_answer = re.findall(r'\[(\d{1,2})\]', tmp)
        cited_chunks = []
        for cite_idx_str in set(citations_in_answer):
            try:
                cite_idx = int(cite_idx_str)
                if 1 <= cite_idx <= len(sources):
                    cited_chunks.append(source_map.get(str(cite_idx), sources[cite_idx - 1].get("chunk_id")))
            except (ValueError, IndexError):
                pass

        # AXIOM 2 citation floor: if no citations found but sources exist, append [1]
        citations_found = len(citations_in_answer)
        if citations_found == 0 and sources:
            answer = answer.rstrip() + " [1]"
            citations_found = 1
            cited_chunks = [sources[0].get("chunk_id", "")]
            logger.debug(f"Citation floor applied: appended [1] to answer")

        logger.info(
            f"Chat '{req.question[:50]}...' -> {len(sources)} sources, "
            f"{citations_found} citations (floor applied: {len(citations_in_answer)==0 and bool(sources)}), {t_retr}ms retrieval, {t_llm}ms LLM"
        )

        model_used = os.getenv("LLM_MODEL", "gpt-oss:20b")
        request_id = str(uuid4())
        total_latency = int((time.time() - t0) * 1000)

        # Build ResponseMetadata with proper Pydantic objects
        decomp_meta = None
        if decomp_result and len(decomp_result.subtasks) > 1:
            decomp_meta = DecompositionMetadata(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=[st.text for st in decomp_result.subtasks],
                llm_used=getattr(decomp_result, "llm_used", False),
                fused_docs=len(sources),
                multi_hit_docs=sum(1 for s in sources if s.get("cluster_size", 1) > 1),
            )

        response_metadata = ResponseMetadata(
            cache_hit=cache_hit,
            index_normalized=True,
            decomposition=decomp_meta,
        )

        # PHASE 5: Cache the answer if not already from cache and answerability passed
        if not cache_hit and is_answerable:
            semantic_cache.set(
                query_embedding=qvec,
                top_doc_ids=top_doc_ids,
                answer=answer,
                sources=sources,
                answerability_score=answerability_score,
                prompt_version="v1",
                namespaces=ns_list,
            )
            logger.debug(f"✓ Cached answer for question: '{req.question[:50]}...'")

        return ChatResponse(
            success=True,
            answer=answer,
            sources=sources,
            latency_ms={"retrieval": t_retr, "llm": t_llm, "total": total_latency},
            meta={
                "request_id": request_id,
                "temperature": temp,
                "model": model_used,
                "namespaces_used": ns_list,
                "k": k,
                "api_type": os.getenv("LLM_API_TYPE", "ollama"),
                "cited_chunks": cited_chunks,
                "citations_found": citations_found,
                "citation_validation": {
                    "valid": citation_validation.is_valid,
                    "cited_sources": sorted(list(citation_validation.cited_indices)),
                    "total_citations": citation_validation.total_citations,
                    "warnings": citation_validation.warnings if citation_validation.warnings else None,
                }
            },
            metadata=response_metadata.model_dump(),
        )

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    decomposition_off: bool = Query(default=False, description="Disable query decomposition"),
    x_api_token: Optional[str] = Header(default=None),
):
    """
    Streaming chat endpoint with RAG: retrieve, ground answer, stream response.

    Returns Server-Sent Events (SSE) stream with chunks of the LLM response.
    Each event contains a JSON payload with response chunks and metadata.
    """
    require_token(x_api_token)
    rate_limit(request.client.host if request.client else "unknown")

    # Check if streaming is enabled
    streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower() == "true"
    if not streaming_enabled:
        raise HTTPException(
            status_code=501,
            detail="Streaming is not enabled. Set STREAMING_ENABLED=true to enable."
        )

    if index_manager:
        index_manager.ensure_loaded()
    k = max(1, min(int(req.k or RETRIEVAL_K), 20))
    t0 = time.time()

    async def generate_stream():
        """Generate SSE stream with LLM response chunks."""
        try:
            # Perform retrieval (same logic as regular chat)
            decomp_result = None
            if not decomposition_off and is_multi_intent_query(req.question):
                try:
                    decomp_result = decompose_query(req.question)
                    logger.info(f"Decomposed streaming chat question into {len(decomp_result.subtasks)} subtasks")
                except Exception as e:
                    logger.warning(f"Query decomposition failed for streaming chat: {e}")

            # Expansion and encoding
            if not decomp_result or len(decomp_result.subtasks) <= 1:
                variants = expand_structured(req.question)
            else:
                variants = [{"text": req.question, "source": "original", "weight": 1.0}]

            qvec = encode_weighted_variants(variants).flatten()
            available_namespaces = list(index_manager.get_all_indexes().keys()) if index_manager else []
            ns_list = [req.namespace] if req.namespace in available_namespaces else sorted(available_namespaces)
            raw_k = k * 3

            # Retrieval
            if decomp_result and len(decomp_result.subtasks) > 1:
                candidates, _ = search_with_decomposition(req.question, qvec, raw_k, ns_list, decomp_result)
                candidates = candidates[:raw_k]
            else:
                query_type = detect_query_type(req.question)
                use_hybrid = should_enable_hybrid_search(query_type, len(req.question))
                if use_hybrid:
                    per_ns = {ns: search_ns_hybrid(ns, qvec[None,:].astype(np.float32), req.question, raw_k) for ns in ns_list}
                else:
                    per_ns = {ns: search_ns(ns, qvec[None,:].astype(np.float32), raw_k) for ns in ns_list}
                candidates = fuse_results(per_ns, raw_k) if len(ns_list) > 1 else per_ns[ns_list[0]]

            # Deduplication
            candidates.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))
            seen_urls = set()
            hits = []
            for candidate in candidates:
                url = candidate.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                hits.append(candidate)
                if len(hits) >= k:
                    break

            t_retr = int((time.time() - t0) * 1000)

            # Build context with RAGPrompt for consistency with /chat endpoint
            chunks = []
            top_context = hits[: min(4, len(hits))]

            for i, h in enumerate(top_context, start=1):
                url = h.get("url", "")
                anchor = h.get("anchor")
                url_with_anchor = f"{url}#{anchor}" if anchor else url
                title_path = h.get("title_path") or []
                if isinstance(title_path, list) and title_path:
                    title = " > ".join(title_path)
                else:
                    title = h.get("title") or url or f"chunk-{i}"
                text = h.get("text", "")[:1600]

                chunk = {
                    "title": title,
                    "url": url_with_anchor,
                    "namespace": h.get("namespace"),
                    "score": h.get("score", 0.0),
                    "text": text,
                }
                chunks.append(chunk)

            # HARMONY: Use RAGPrompt.build_messages() for consistency
            messages, sources, developer_instructions = RAGPrompt.build_messages(
                question=req.question,
                chunks=chunks,
                namespace=req.namespace or "clockify",
                max_chunks=len(chunks),
                reasoning_effort="low"  # RAG optimization: minimize latency
            )

            # Send initial metadata
            metadata_event = {
                "type": "metadata",
                "sources": sources,
                "retrieval_latency_ms": t_retr,
                "k": k,
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"

            # Stream LLM response
            temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
            t1 = time.time()

            # Get streaming response from LLM
            llm = LLMClient()
            try:
                # HARMONY: Use chat method with messages list
                full_answer = llm.chat(
                    messages=messages,
                    max_tokens=800,
                    temperature=temp,
                    stream=True
                )
            except CircuitOpenError as e:
                # Circuit breaker is open - send error event
                logger.error(f"Circuit breaker open for streaming chat: {str(e)}")
                error_event = {
                    "type": "error",
                    "error": "LLM service temporarily unavailable. Circuit breaker is open. Please try again in a minute.",
                    "error_code": "CIRCUIT_OPEN",
                    "retry_after_seconds": 60
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            t_llm = int((time.time() - t1) * 1000)

            # Send answer chunk
            answer_event = {
                "type": "answer",
                "content": full_answer,
                "llm_latency_ms": t_llm,
            }
            yield f"data: {json.dumps(answer_event)}\n\n"

            # Validate citations
            citation_validation = validate_citations(full_answer, len(sources), strict=False)

            # Send final event with validation
            done_event = {
                "type": "done",
                "total_latency_ms": int((time.time() - t0) * 1000),
                "citation_validation": {
                    "valid": citation_validation.is_valid,
                    "cited_sources": sorted(list(citation_validation.cited_indices)),
                    "warnings": citation_validation.warnings if citation_validation.warnings else None,
                }
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            error_event = {
                "type": "error",
                "error": str(e),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# --------- Mount Static Files for Web UI ---------
# Serve the web UI from public/ directory
PUBLIC_DIR = Path(__file__).parent.parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
    logger.info(f"Mounted static files from {PUBLIC_DIR}")
else:
    logger.warning(f"Public directory not found at {PUBLIC_DIR}, web UI will not be served")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
```

