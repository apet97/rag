# Troubleshooting Guide

## Quick Diagnosis

### 1. Check System Health

```bash
curl -H "x-api-token: change-me" http://localhost:7000/health | jq '.ok'
```

Should return `true`. If false, check:
- Index is loaded: `jq '.index_metrics'`
- Embedding model works: `jq '.embedding_ok'`
- LLM is available: `jq '.llm_ok'`

### 2. Get Performance Baseline

```bash
curl -H "x-api-token: change-me" http://localhost:7000/perf | jq '.by_stage.end_to_end'
```

Check `mean_ms` and `p95_ms`:
- **<300ms**: Excellent
- **300-1000ms**: Good
- **1000-3000ms**: Acceptable but slow
- **>3000ms**: Investigate bottleneck

### 3. Test Single Query

```bash
time curl -H "x-api-token: change-me" \
  'http://localhost:7000/search?q=test&k=5&namespace=clockify'
```

Measure actual vs reported latency to identify network overhead.

---

## Common Issues & Solutions

### Issue: "Search Timeout"

**Error:** Request takes >30 seconds

**Root Cause Diagnosis:**

```bash
# Check if vector search is slow
curl http://localhost:7000/perf | jq '.by_stage.vector_search'

# Check if LLM is slow
curl http://localhost:7000/perf | jq '.by_stage.llm_generation'

# Check if reranking is slow
curl http://localhost:7000/perf | jq '.by_stage.reranking'
```

**Solution:**

1. **If Vector Search >500ms:**
   ```python
   # In retrieval_engine.py or config
   config.k_vector = 10  # Reduce from default
   ```

2. **If LLM >2000ms:**
   ```bash
   # Check LLM server
   curl http://10.127.0.192:11434/api/tags

   # If LLM is slow, reduce context:
   export RETRIEVAL_K=3  # Use fewer documents
   ```

3. **If Reranking >500ms:**
   ```bash
   # Disable reranking
   export RERANK_DISABLED=true
   ```

---

### Issue: "Low Cache Hit Rate"

**Symptom:** Cache hit rate <30%

**Diagnosis:**

```bash
curl http://localhost:7000/health | jq '.cache.semantic_cache_stats.hit_rate_pct'
```

**Root Causes & Solutions:**

1. **TTL too short:**
   ```python
   # src/tuning_config.py
   SEMANTIC_CACHE_TTL_SECONDS = 7200  # Increase from 3600
   ```

2. **Cache too small:**
   ```python
   # src/tuning_config.py
   SEMANTIC_CACHE_MAX_SIZE = 20000  # Increase from 10000
   ```

3. **Queries too diverse:**
   - Check query patterns: Are queries too different each time?
   - Consider pre-computing common queries
   - Use query normalization (lowercase, remove extra spaces)

4. **Namespace-specific low rate:**
   ```bash
   curl http://localhost:7000/health | jq '.cache.semantic_cache_stats.by_namespace'
   ```
   - If one namespace has low rate, increase its data
   - Consider namespace-specific TTL tuning

---

### Issue: "High Memory Usage"

**Symptom:** Process using >2GB RAM

**Diagnosis:**

```bash
curl http://localhost:7000/health | jq '.cache.semantic_cache_stats.memory_usage_mb'

# Get per-entry size
curl http://localhost:7000/health | jq '.cache.semantic_cache_stats.avg_entry_size_bytes'
```

**Solutions:**

1. **Reduce cache size:**
   ```python
   SEMANTIC_CACHE_MAX_SIZE = 5000  # Reduce from 10000
   ```

2. **Reduce response cache TTL:**
   ```python
   # src/cache.py - LRUResponseCache init
   default_ttl = 1800  # Reduce from 3600 (30 min)
   ```

3. **Compress cache entries:**
   ```python
   # Consider storing compressed JSON:
   import gzip
   import json

   compressed = gzip.compress(json.dumps(response).encode())
   ```

4. **Monitor entry sizes:**
   ```bash
   curl http://localhost:7000/health | jq '.cache.semantic_cache_stats | {
     size: .size,
     memory_mb: .memory_usage_mb,
     avg_entry_bytes: .avg_entry_size_bytes
   }'
   ```

---

### Issue: "High Vector Search Latency"

**Symptom:** Vector search taking >200ms

**Diagnosis:**

```bash
curl http://localhost:7000/perf | jq '.by_stage.vector_search'

# Check index size
curl http://localhost:7000/config | jq '.index_metrics'
```

**Solutions:**

1. **Reduce k_vector:**
   ```python
   # src/retrieval_engine.py - RetrievalConfig
   k_vector = 10  # Reduce from 20
   ```

2. **Use index partitioning:**
   - Split large indexes into multiple namespaces
   - Create separate indexes for different document types

3. **Optimize FAISS index:**
   ```python
   # Use quantized index
   quantizer = faiss.ScalarQuantizer(faiss.ScalarQuantizer.QT_8bit)
   index = faiss.IndexIVFScalarQuantizer(quantizer)
   ```

4. **Upgrade hardware:**
   - FAISS benefits from CPU cores and RAM
   - Use SSD for index files

---

### Issue: "Low Reranker Quality"

**Symptom:** Results after reranking are worse

**Diagnosis:**

```bash
# Check if reranker is enabled
curl http://localhost:7000/config | jq '.rerank_disabled'

# Check reranking latency
curl http://localhost:7000/perf | jq '.by_stage.reranking'
```

**Solutions:**

1. **Disable reranking:**
   ```bash
   export RERANK_DISABLED=true
   ```

2. **Reduce documents sent to reranker:**
   ```python
   # src/rerank.py - adjust k parameter
   rerank(results[:10])  # Rerank top 10 only
   ```

3. **Use better reranker model:**
   ```python
   # src/rerank.py
   model = FlagReranker("BAAI/bge-reranker-large")  # Larger model
   ```

---

### Issue: "Empty Search Results"

**Symptom:** Query returns 0 results even for common questions

**Diagnosis:**

```bash
# Test with index
curl -H "x-api-token: change-me" \
  'http://localhost:7000/search?q=timesheet&k=20&namespace=clockify' \
  | jq '.results | length'

# Check index is loaded
curl http://localhost:7000/config | jq '.index_metrics'
```

**Solutions:**

1. **Check index exists:**
   ```bash
   ls -la index/faiss/clockify/
   ```

2. **Reload index:**
   - Restart the server
   - Or manually call index manager

3. **Lower similarity threshold:**
   ```python
   # In vector search strategy
   results = index.search(query_vec, k * 2)  # Get more candidates
   results = [r for r in results if r.score > 0.3]  # Lower threshold
   ```

4. **Try BM25-only search:**
   ```bash
   curl -H "x-api-token: change-me" \
     'http://localhost:7000/search?q=timesheet&k=5&strategy=bm25'
   ```

---

### Issue: "Authentication Failed"

**Error:** "401 Unauthorized"

**Solution:**

```bash
# Check token
export API_TOKEN="your-token"

curl -H "x-api-token: $API_TOKEN" http://localhost:7000/health

# Default token
curl -H "x-api-token: change-me" http://localhost:7000/health
```

---

### Issue: "Database/Index Corruption"

**Symptom:** Segmentation fault or strange errors from FAISS

**Diagnosis:**

```bash
# Verify index files
python3 -c "import faiss; faiss.read_index('index/faiss/clockify/index')"
```

**Solutions:**

1. **Rebuild index:**
   ```bash
   rm -rf index/faiss/clockify/*
   python3 -m src.ingest  # Rebuild from data
   ```

2. **Restore from backup:**
   ```bash
   cp /backup/index/faiss/clockify/* index/faiss/clockify/
   ```

---

## Performance Optimization Checklist

- [ ] Check end-to-end latency: p95 < 1000ms
- [ ] Monitor cache hit rate: > 50%
- [ ] Verify memory usage: < 2GB
- [ ] Check bottleneck stages: Review with `/perf?detailed=true`
- [ ] Test query patterns: Vary k, namespace, query type
- [ ] Monitor error rates: < 0.1%
- [ ] Validate result quality: Sample searches are relevant
- [ ] Review feature adoption: MMR, time decay, decomposition usage

---

## Debug Mode

### Enable Verbose Logging

```bash
export LOG_LEVEL=DEBUG
python3 -m uvicorn src.server:app
```

### Check Specific Stages

```bash
# Watch vector search performance
watch -n 1 'curl -s http://localhost:7000/perf | \
  jq ".by_stage.vector_search | {mean: .mean_ms, p95: .p95_ms, count: .count}"'
```

### Profile a Single Query

```python
import cProfile
import pstats
from src.retrieval_engine import RetrievalEngine

profiler = cProfile.Profile()
profiler.enable()

engine = RetrievalEngine()
results = engine.search(embedding, "query", chunks, k=5)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Support Resources

- **Logs:** Check `logs/` directory
- **Metrics:** `/perf` endpoint (detailed view)
- **Health:** `/health` endpoint
- **Config:** `/config` endpoint
- **Docs:** `docs/` directory

---

## When All Else Fails

1. **Check logs:**
   ```bash
   tail -f logs/app.log | grep ERROR
   ```

2. **Restart server:**
   ```bash
   pkill -f "uvicorn src.server"
   python3 -m uvicorn src.server:app --reload
   ```

3. **Reset caches:**
   ```bash
   curl -X POST http://localhost:7000/cache/clear
   ```

4. **Verify connectivity:**
   ```bash
   # LLM
   curl http://10.127.0.192:11434/api/tags

   # Index
   ls -la index/faiss/*/
   ```

5. **Report issue with:**
   - Error message
   - Request: `curl` command that reproduces it
   - Metrics: Output of `/perf?detailed=true`
   - Logs: Recent log entries
   - Environment: Python version, OS, Docker/native
