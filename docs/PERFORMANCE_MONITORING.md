# Performance Monitoring Guide

## Overview

This guide explains how to use the RAG system's built-in performance monitoring and observability features (Tier 2 improvements) to track, analyze, and optimize system performance.

## Performance Metrics Endpoint

### Basic Usage

Get overall performance statistics:

```bash
curl -H "x-api-token: change-me" http://localhost:7000/perf
```

Response includes:
- Per-stage latency statistics (min, max, mean, median, p95, p99)
- Stage counts and sample sizes
- All 12 pipeline stages tracked

### Detailed Analysis

Get performance data with bottleneck detection:

```bash
curl -H "x-api-token: change-me" http://localhost:7000/perf?detailed=true
```

Returns:
- **by_stage**: Latency stats for each pipeline stage
- **recent_samples**: Last 20 latency measurements with metadata
- **bottlenecks**: Top 5 slowest stages for optimization focus
- **total_samples**: Total measurements collected

## Pipeline Stages

The performance tracker monitors these stages:

### Retrieval Pipeline
1. **query_embedding** - Query embedding generation latency
2. **vector_search** - Vector/semantic search latency
3. **bm25_search** - BM25 keyword search latency
4. **fusion** - RRF fusion latency
5. **diversity_filter** - MMR diversity penalty application
6. **time_decay** - Temporal freshness boost application

### Ranking & Generation
7. **reranking** - Optional reranker latency
8. **llm_generation** - LLM response generation
9. **answerability_check** - Grounding validation latency

### Cache Operations
10. **cache_lookup** - Semantic cache retrieval
11. **cache_store** - Cache storage latency
12. **end_to_end** - Total request latency

## Cache Monitoring

### Basic Cache Stats

Available in `/health` endpoint:

```bash
curl -H "x-api-token: change-me" http://localhost:7000/health | jq '.cache'
```

Shows:
- Semantic cache size and utilization
- TTL configuration
- Current cache statistics

### Detailed Cache Analysis

Response cache stats available in `/health`:

```json
{
  "hits": 1234,
  "misses": 567,
  "hit_rate_pct": 68.5,
  "size": 890,
  "capacity": 1000,
  "evictions": 42,
  "memory_usage_mb": 125.3,
  "avg_entry_size_bytes": 140832,
  "by_namespace": {
    "clockify": { "hits": 800, "misses": 300, "hit_rate_pct": 72.7 },
    "langchain": { "hits": 434, "misses": 267, "hit_rate_pct": 62.0 }
  }
}
```

### Key Metrics

- **hit_rate_pct**: Percentage of requests served from cache
- **memory_usage_mb**: Total cache memory consumption
- **avg_entry_size_bytes**: Average response size
- **by_namespace**: Per-namespace cache effectiveness

## Identifying Bottlenecks

### Step 1: Check Overall Latency

```bash
curl -s http://localhost:7000/perf?detailed=true | jq '.by_stage.end_to_end'
```

If median latency is high, proceed to Step 2.

### Step 2: Find Slowest Stages

```bash
curl -s http://localhost:7000/perf?detailed=true | jq '.bottlenecks'
```

Typical bottlenecks:
- **llm_generation**: 500-2000ms (normal for LLM calls)
- **reranking**: 100-300ms (if enabled)
- **vector_search**: 50-150ms (index size dependent)

### Step 3: Monitor Specific Stage

```bash
curl -s http://localhost:7000/perf?detailed=true | jq '.recent_samples[] | select(.stage=="llm_generation")'
```

### Step 4: Analyze Trends

Monitor the same endpoint periodically:

```bash
watch -n 5 'curl -s http://localhost:7000/perf | jq ".by_stage | to_entries[] | \{stage: .key, mean_ms: .value.mean_ms}"'
```

## Cache Optimization

### Semantic Cache Tuning

Configuration in `src/tuning_config.py`:

```python
SEMANTIC_CACHE_MAX_SIZE = 10000  # Increase for higher hit rate
SEMANTIC_CACHE_TTL_SECONDS = 3600  # Extend for stable queries
```

### Hit Rate Improvement

If cache hit rate < 50%:

1. **Increase TTL**: Extend from 3600s (1h) to 7200s (2h)
   ```python
   SEMANTIC_CACHE_TTL_SECONDS = 7200
   ```

2. **Increase Cache Size**: Grow from 10000 to 20000 entries
   ```python
   SEMANTIC_CACHE_MAX_SIZE = 20000
   ```

3. **Monitor by namespace**: Check if specific namespaces have low hit rates
   - Increase K for that namespace
   - Ensure consistent query patterns

## Query Analysis

### Check Feature Adoption

The system tracks usage of advanced features:

- MMR (Maximal Marginal Relevance) diversity filtering
- Time decay for freshness boosting
- Multi-intent query decomposition
- Cache hit efficiency

## Performance Tuning Parameters

### Fusion Strategy (RRF)

```python
RRF_K_CONSTANT = 60  # Lower = more weight to top results
```

Tuning:
- High K (>100): More balanced ranking, better for diverse results
- Low K (<30): Top results dominate, good for precision-focused tasks

### Diversity (MMR)

```python
MMR_LAMBDA = 0.7  # 70% relevance, 30% diversity
```

Tuning:
- Increase to 0.8-0.9: More relevant results, less diverse
- Decrease to 0.5-0.6: More diverse results, less relevant

### Time Decay

```python
TIME_DECAY_RATE = 0.95  # 5% decay per month
```

Tuning:
- Higher (0.98): Recent documents weighted more lightly
- Lower (0.90): Recent documents weighted more heavily

## Monitoring Checklist

Daily:
- [ ] Check `/perf` endpoint for p95 latency
- [ ] Monitor cache hit rate (should be >50%)
- [ ] Verify no slow stage degradation

Weekly:
- [ ] Review bottleneck analysis
- [ ] Check memory usage trends
- [ ] Validate feature adoption rates

Monthly:
- [ ] Analyze query patterns
- [ ] Review fusion strategy effectiveness
- [ ] Tune hyperparameters based on metrics

## Troubleshooting

### High Latency

**Vector Search Slow (>200ms)?**
- Check index size: `/config` endpoint
- Reduce k values in retrieval config
- Optimize FAISS index parameters

**LLM Slow (>2000ms)?**
- Check LLM server health: `http://10.127.0.192:11434/api/tags`
- Reduce context size (fewer retrieved documents)
- Increase LLM timeout if network is unstable

**Reranking Slow (>500ms)?**
- Consider disabling reranking: `RERANK_DISABLED=true`
- Reduce documents sent to reranker (lower k)
- Use lighter reranker model

### Low Cache Hit Rate

**Below 30%?**
- Increase TTL in tuning_config.py
- Analyze query patterns for similarity
- Check if queries are too diverse
- Verify semantic cache is enabled

**Below 50%?**
- Increase cache size (SEMANTIC_CACHE_MAX_SIZE)
- Review query decomposition logic
- Check for timestamp variations in queries

## Integration with Monitoring Systems

### Prometheus Metrics

Performance stats can be exported to Prometheus:

```python
from prometheus_client import Gauge

perf_tracker = get_performance_tracker()
stats = perf_tracker.get_stats()

vector_search_p95 = Gauge('rag_vector_search_p95_ms', 'Vector search p95 latency')
vector_search_p95.set(stats['by_stage']['vector_search']['p95_ms'])
```

### Grafana Dashboard

Create a dashboard using `/perf` endpoint data:
- X-axis: Time
- Y-axis: Latency (p50, p95, p99)
- Panels: One per stage

### Alert Rules

Suggested alert thresholds:

```yaml
- alert: HighLatency
  expr: rag_end_to_end_p95_ms > 3000
  for: 5m

- alert: LowCacheHitRate
  expr: rag_cache_hit_rate < 30
  for: 10m

- alert: HighMemoryUsage
  expr: rag_cache_memory_mb > 500
  for: 5m
```

## Performance Benchmarking

### Baseline Measurement

Run with production data:

```bash
# Reset and warm up
curl -X POST http://localhost:7000/cache/clear

# Run 100 queries
for i in {1..100}; do
  curl http://localhost:7000/search?q=sample_query_$i&k=5
done

# Check stats
curl http://localhost:7000/perf?detailed=true
```

### Before/After Tuning

1. Record baseline: `perf_baseline.json`
2. Apply tuning changes
3. Record after tuning: `perf_after.json`
4. Compare metrics

## Summary

The performance monitoring system provides:
- ✅ Real-time latency tracking per pipeline stage
- ✅ Cache effectiveness analysis by namespace
- ✅ Bottleneck detection and recommendations
- ✅ Feature adoption and usage tracking
- ✅ Memory usage monitoring
- ✅ Historical sample collection for analysis

Use these tools to continuously optimize and understand your RAG system's behavior.
