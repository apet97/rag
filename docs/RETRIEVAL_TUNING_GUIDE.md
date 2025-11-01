# Retrieval Tuning Guide

## Overview

Retrieval quality determines whether your RAG system finds the right documents for each question. This guide explains how to tune the hybrid retrieval engine for optimal precision and recall.

## Current System Architecture

**Hybrid Retrieval:** Combines two approaches
1. **BM25 (Lexical):** Traditional keyword matching
2. **Vector (Semantic):** Embedding similarity

**Fusion Method:** Reciprocal Rank Fusion (RRF)
- Merges results from both rankers
- Balanced by `SEARCH_LEXICAL_WEIGHT` parameter

## Key Parameters

### 1. SEARCH_LEXICAL_WEIGHT (Most Important)

**What it does:** Controls the balance between keyword and semantic matching

```bash
SEARCH_LEXICAL_WEIGHT=0.35  # Current: 35% BM25, 65% vector
```

**When to adjust:**

| Value | Use Case | Behavior |
|-------|----------|----------|
| **0.0** | Pure semantic | Ignores exact keywords, finds conceptually similar docs |
| **0.2** | Semantic-heavy | Good for paraphrased questions, synonym handling |
| **0.35** | Balanced (current) | General purpose, favors semantic |
| **0.5** | True hybrid | Equal weight to keywords and semantics |
| **0.7** | Keyword-heavy | Good for technical terms, product names |
| **1.0** | Pure BM25 | Only exact keyword matching (fast, but brittle) |

**Example impact:**

Query: "how to edit client"
- **0.35:** Returns "Edit Client Profile", "Manage Clients", "Update Client Info"
- **0.70:** Returns only docs with exact phrase "edit client"

Query: "how to edit a client's email"
- **0.35:** Returns same docs as "edit client" (semantic similarity)
- **0.70:** Different results due to extra keywords "email"

**Tuning procedure:**

```bash
# 1. Test current setting
curl "http://localhost:7001/search?q=how+to+edit+client&k=10"

# 2. Adjust in .env
SEARCH_LEXICAL_WEIGHT=0.25  # Try lower (more semantic)

# 3. Restart and retest
pkill -f uvicorn && uvicorn src.server:app --host 0.0.0.0 --port 7001
curl "http://localhost:7001/search?q=how+to+edit+client&k=10"

# 4. Compare results
```

### 2. MAX_CONTEXT_CHUNKS

**What it does:** Maximum number of document chunks passed to LLM

```bash
MAX_CONTEXT_CHUNKS=8  # Default
```

**Trade-offs:**
- **Higher (12-16):** More context, better answers, slower, more tokens
- **Lower (4-6):** Faster, cheaper, but may miss relevant info

**Tuning:**
```bash
# Check if LLM is using all context
grep "Retrieved chunks:" logs/*.log

# If LLM consistently uses <50% of chunks, reduce:
MAX_CONTEXT_CHUNKS=6

# If answers lack detail, increase:
MAX_CONTEXT_CHUNKS=12
```

### 3. CONTEXT_CHAR_LIMIT

**What it does:** Maximum characters per chunk in LLM context

```bash
CONTEXT_CHAR_LIMIT=1200  # Default
```

**When to adjust:**
- **Increase (1500-2000):** Long-form articles, need full paragraphs
- **Decrease (800-1000):** Short snippets, reduce noise

### 4. RETRIEVAL_K (Advanced)

**What it does:** Number of candidates to retrieve before reranking

**Auto-tuned** based on query type:
- How-to: k=40
- Factual: k=20
- Other: k=15

## Measuring Retrieval Quality

### Metric 1: Recall@K

"Do the top K results contain the correct answer?"

```bash
# Manual test
curl "http://localhost:7001/search?q=how+to+create+project&k=5" | \
  jq '.results[] | {title, url}'

# Check: Is the target doc in top 5?
```

**Targets:**
- Recall@5: >90% (most queries find answer in top 5)
- Recall@10: >95%

### Metric 2: Mean Reciprocal Rank (MRR)

"On average, what rank is the correct answer?"

```bash
# Test queries
queries=(
  "how to create project"
  "edit client email"
  "delete workspace"
)

for q in "${queries[@]}"; do
  echo "Query: $q"
  curl -s "http://localhost:7001/search?q=$q&k=10" | \
    jq '.results[0:3] | .[] | .title'
  echo ""
done
```

**Calculate MRR:**
```
MRR = average(1 / rank_of_correct_answer)
```

**Target:** MRR > 0.7 (correct answer in top 2 on average)

### Metric 3: Search Latency

```bash
# Check logs
grep "Search.*-> .*results.*in.*ms" logs/*.log

# Example output:
# Search 'how to edit client' k=5 -> 5 results (unique URLs) in 146ms
```

**Targets:**
- Search latency: <200ms
- Chat latency (retrieval + LLM): <8s

## Common Tuning Scenarios

### Scenario 1: Similar Queries Return Different Results

**Problem:**
- "how to edit client" returns Doc A
- "how to edit a client's email" returns Doc B
- They should return the same docs

**Diagnosis:**
```bash
SEARCH_LEXICAL_WEIGHT=0.50  # Too high, sensitive to keywords
```

**Fix:**
```bash
SEARCH_LEXICAL_WEIGHT=0.35  # Lower, favor semantic similarity
```

**Why it works:** Semantic embeddings capture that "edit client" and "edit client email" are conceptually similar, despite different keywords.

### Scenario 2: Missing Technical Terms

**Problem:**
- Query: "API rate limit"
- Returns: Generic docs about "limits" and "APIs" separately
- Misses: Specific "API rate limit" documentation

**Diagnosis:**
```bash
SEARCH_LEXICAL_WEIGHT=0.35  # Too semantic, ignores exact phrase
```

**Fix:**
```bash
SEARCH_LEXICAL_WEIGHT=0.60  # Higher, require keyword match
```

**Why it works:** BM25 heavily rewards exact phrase matches like "API rate limit".

### Scenario 3: Answers Lack Detail

**Problem:**
- LLM says "I don't have enough information"
- But relevant docs exist in corpus

**Diagnosis:**
```bash
MAX_CONTEXT_CHUNKS=8  # Too few chunks
```

**Fix:**
```bash
MAX_CONTEXT_CHUNKS=12  # More context
CONTEXT_CHAR_LIMIT=1500  # Longer chunks
```

**Validate:**
```bash
# Check retrieved chunks
grep -A10 "Retrieved chunks:" logs/*.log
```

### Scenario 4: Slow Retrieval (>500ms)

**Problem:**
- Search takes 4+ seconds
- User experience is poor

**Diagnosis:**
```bash
# Check logs
grep "Slow stage:" logs/*.log
# Output: Slow stage: diversity_filter took 3948ms
```

**Fix:** (Already applied in latest code)
```python
# src/retrieval_engine.py
if len(results) > 20:
    return results[:self.config.k_final]  # Skip expensive MMR
```

**Alternative:**
```bash
# Disable diversity filtering entirely
# In src/retrieval_engine.py, set:
apply_diversity_penalty=False
```

## Advanced: Hybrid Sweep for Optimal Weight

Use the evaluation script to find the best lexical weight:

```bash
# Run hybrid sweep
python3 tools/hybrid_sweep.py \
  --query-file eval/test_queries.txt \
  --output eval/sweep_results.json

# Analyze results
python3 -c "
import json
with open('eval/sweep_results.json') as f:
    results = json.load(f)

for weight, metrics in results.items():
    print(f'Weight {weight}: MRR={metrics[\"mrr\"]:.3f}, Recall@5={metrics[\"recall5\"]:.3f}')
"
```

**Expected output:**
```
Weight 0.0: MRR=0.650, Recall@5=0.88
Weight 0.2: MRR=0.710, Recall@5=0.92
Weight 0.35: MRR=0.735, Recall@5=0.94  <- Current
Weight 0.5: MRR=0.720, Recall@5=0.93
Weight 0.7: MRR=0.680, Recall@5=0.90
Weight 1.0: MRR=0.620, Recall@5=0.85
```

**Interpretation:** Weight=0.35 maximizes both MRR and Recall@5.

## Production Monitoring

### Dashboard Metrics

Track in your monitoring system:

```python
# Metrics to collect
retrieval_latency_p50  # Median latency
retrieval_latency_p95  # 95th percentile
retrieval_recall_at_5  # % queries with answer in top 5
mrr_score              # Mean reciprocal rank
```

### Alert Thresholds

```yaml
alerts:
  - name: slow_retrieval
    condition: retrieval_latency_p95 > 500ms
    action: Check diversity filter, index size

  - name: low_recall
    condition: retrieval_recall_at_5 < 0.85
    action: Review SEARCH_LEXICAL_WEIGHT, check corpus coverage

  - name: answer_refusals
    condition: refusal_rate > 0.30
    action: Check MAX_CONTEXT_CHUNKS, ANSWERABILITY_THRESHOLD
```

## Quick Reference

### For Technical Documentation (APIs, Code)

```bash
SEARCH_LEXICAL_WEIGHT=0.55  # Favor keywords for technical terms
MAX_CONTEXT_CHUNKS=10
CONTEXT_CHAR_LIMIT=1500
```

### For Conversational Help Docs (How-tos)

```bash
SEARCH_LEXICAL_WEIGHT=0.30  # Favor semantic for paraphrasing
MAX_CONTEXT_CHUNKS=8
CONTEXT_CHAR_LIMIT=1200
```

### For Mixed Content (Current Setup)

```bash
SEARCH_LEXICAL_WEIGHT=0.35  # Balanced, slight semantic preference
MAX_CONTEXT_CHUNKS=8
CONTEXT_CHAR_LIMIT=1200
```

## See Also

- [Answerability Tuning](ANSWERABILITY_TUNING.md) - Fix "I don't know" responses
- [Add New Documents](ADD_NEW_DOCUMENTS.md) - Expand knowledge base
- [Parameter Reference](../README.md#configuration) - All config options
