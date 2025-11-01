# Adding New Documents to the Knowledge Base

## Overview

This guide explains how to add new documents to your RAG system's index. The process involves updating corpus configuration, running the ingestion pipeline, and validating the results.

## Quick Start (3 Steps)

```bash
# 1. Add URLs to ALLOWLIST
echo "https://help.clockify.me/en/articles/new-article" >> codex/ALLOWLIST.txt

# 2. Rebuild index
python3 tools/ingest_v2.py

# 3. Restart server
pkill -f uvicorn && uvicorn src.server:app --host 0.0.0.0 --port 7001
```

## Step-by-Step Guide

### Step 1: Prepare Your Corpus

#### Option A: Add URLs to Crawl

Edit `codex/ALLOWLIST.txt`:

```bash
# clockify/ALLOWLIST.txt - URL patterns to crawl
https://help.clockify.me/en/articles/123456-projects
https://help.clockify.me/en/articles/789012-clients
https://help.clockify.me/en/collections/workspace-*
```

**URL patterns supported:**
- Exact match: `https://help.clockify.me/en/articles/123456`
- Prefix match: `https://help.clockify.me/en/collections/workspace-`
- Wildcard: `https://help.clockify.me/en/articles/*`

**Check for exclusions:**
```bash
# Make sure URLs aren't blocked
grep "^https://help.clockify.me/en/articles/123456" codex/DENYLIST.txt
# (should return nothing)
```

#### Option B: Provide Pre-Scraped JSON

If you already have scraped content, create enriched JSON:

```json
{
  "https://help.clockify.me/en/articles/123456": {
    "url": "https://help.clockify.me/en/articles/123456",
    "title": "How to Create a Project",
    "h1": "Creating Projects in Clockify",
    "content": "Full article text here...",
    "content_type": "text/html",
    "status_code": 200,
    "scraped_at": "2025-11-01T12:00:00Z"
  }
}
```

Save as `codex/CRAWLED_LINKS_enriched.json` (will be merged with existing).

### Step 2: Run Ingestion

The ingestion pipeline:
1. Loads ALLOWLIST/DENYLIST policies
2. Scrapes/loads documents
3. Chunks text based on strategy (url_level, section, sliding_window)
4. Generates embeddings
5. Builds FAISS index

**Run ingestion:**

```bash
python3 tools/ingest_v2.py
```

**Expected output:**
```
🔧 RAG Ingestion Pipeline v2
================================================================================

📊 Configuration:
  Namespaces: clockify_url
  Embedding model: intfloat/multilingual-e5-base (768D)
  Chunk strategy: url_level
  Index root: index/faiss

📚 Loading corpus from codex/CRAWLED_LINKS_enriched.json
✓ Loaded 253 documents

🔨 Processing namespace: clockify_url
  Chunking with strategy: url_level
  ✓ Created 253 chunks

🧮 Generating embeddings (batch_size=32)...
[████████████████████████] 253/253 chunks (100%)
✓ Embeddings complete: 253 vectors (768D)

💾 Building FAISS index: IndexFlatIP (inner product)
✓ Index built: 253 vectors
✓ Saved: index/faiss/clockify_url/index.bin
✓ Saved: index/faiss/clockify_url/meta.json

✅ Ingestion complete: 253 documents indexed
```

**Troubleshooting ingestion errors:**

```bash
# If scraping fails
ERROR: HTTPError 403 for URL https://...
Fix: Check ALLOWLIST pattern, verify URL is accessible

# If embedding fails
ERROR: Embedding dimension mismatch
Fix: Check EMBEDDING_MODEL in .env matches previous index

# If out of memory
ERROR: MemoryError during faiss.IndexFlatIP
Fix: Process in smaller batches, or use less memory-intensive index
```

### Step 3: Validate the New Index

#### 3a. Check Document Count

```bash
# Before ingestion
ls -lh index/faiss/clockify_url/index.bin
# (note file size)

# After ingestion
ls -lh index/faiss/clockify_url/index.bin
# (should be larger if docs added)

# Check metadata
python3 -c "
import json
with open('index/faiss/clockify_url/meta.json') as f:
    meta = json.load(f)
print(f'Total vectors: {len(meta)}')
print(f'Sample: {meta[0]}')
"
```

#### 3b. Test Retrieval

Start server and test search:

```bash
# Start server
uvicorn src.server:app --host 0.0.0.0 --port 7001

# Test search for new content
curl "http://localhost:7001/search?q=your+new+topic&k=5" \
  -H "x-api-token: $API_TOKEN" | jq '.results[] | .title'
```

**Expected:** New documents appear in results

#### 3c. Verify Corpus Coverage

```bash
# Check INDEX_DIGEST for corpus hash
cat codex/INDEX_DIGEST.txt
```

Example:
```
corpus_hash=abc123def456
combined=clockify_url:abc123:768
```

## Incremental vs Full Rebuild

### Incremental Addition (Recommended)

**When to use:**
- Adding <20% new documents
- Existing docs unchanged
- Same embedding model

**How:**
1. Append new URLs to ALLOWLIST
2. Run `python3 tools/ingest_v2.py` (merges with existing)
3. Restart server

### Full Rebuild

**When to use:**
- Changing embedding model (e.g., switching from 384D to 768D)
- Changing chunk strategy (url_level → sliding_window)
- Major corpus overhaul (>50% content changed)

**How:**
```bash
# 1. Backup current index
cp -r index/faiss/clockify_url index/faiss/clockify_url.backup

# 2. Delete old index
rm -rf index/faiss/clockify_url/

# 3. Update config if needed
# Edit .env: EMBEDDING_MODEL, CHUNK_STRATEGY, etc.

# 4. Run ingestion
python3 tools/ingest_v2.py

# 5. Validate (Step 3 above)
```

## Advanced: Custom Chunking Strategies

### URL-Level Chunking (Default)

One chunk per document. Best for:
- Short articles (<2000 words)
- Each URL is self-contained
- Fast retrieval

```bash
CHUNK_STRATEGY=url_level
```

### Section Chunking

Split by headers (H2/H3). Best for:
- Long articles with clear sections
- Want to retrieve specific subsections
- Better precision

```bash
CHUNK_STRATEGY=section
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

### Sliding Window Chunking

Overlapping text windows. Best for:
- Very long documents
- No clear section structure
- Maximum recall

```bash
CHUNK_STRATEGY=sliding_window
CHUNK_SIZE=800
CHUNK_OVERLAP=200
```

## Managing Document Updates

### Update Existing Document

```bash
# 1. Content has changed - need to re-scrape
# Force re-scrape by touching ALLOWLIST
touch codex/ALLOWLIST.txt

# 2. Re-run ingestion (will re-fetch and update)
python3 tools/ingest_v2.py
```

### Remove Document

```bash
# 1. Add URL to DENYLIST
echo "https://help.clockify.me/en/articles/deprecated-article" >> codex/DENYLIST.txt

# 2. Full rebuild (incremental doesn't support removal)
rm -rf index/faiss/clockify_url/
python3 tools/ingest_v2.py
```

## Corpus Organization Best Practices

### Multiple Namespaces

For different knowledge bases:

```bash
NAMESPACES=clockify_help,clockify_api,clockify_blog
```

Each namespace gets its own index:
```
index/faiss/
├── clockify_help/
├── clockify_api/
└── clockify_blog/
```

### Namespace-Specific ALLOWLISTs

```bash
# clockify/ALLOWLIST.txt - Help articles
https://help.clockify.me/en/*

# api/ALLOWLIST.txt - API docs
https://api.clockify.me/docs/*
```

## Monitoring Corpus Health

### Check for Stale Documents

```bash
# Find docs older than 30 days
python3 -c "
import json
from datetime import datetime, timedelta
with open('codex/CRAWLED_LINKS_enriched.json') as f:
    corpus = json.load(f)

cutoff = datetime.now() - timedelta(days=30)
stale = [url for url, doc in corpus.items()
         if datetime.fromisoformat(doc['scraped_at'].replace('Z', '+00:00')) < cutoff]

print(f'Stale documents: {len(stale)}')
for url in stale[:10]:
    print(f'  {url}')
"
```

### Validate Coverage

```bash
# Compare ALLOWLIST vs actual indexed URLs
comm -23 <(grep "^https://" codex/ALLOWLIST.txt | sort) \
         <(jq -r 'keys[]' codex/CRAWLED_LINKS_enriched.json | sort) \
  > missing_urls.txt

cat missing_urls.txt
# (should be empty or contain only excluded URLs)
```

## See Also

- [Retrieval Tuning Guide](RETRIEVAL_TUNING_GUIDE.md) - Optimize search quality
- [Parameter Reference](../README.md#configuration) - All config options
- [RUNBOOK](../codex/RUNBOOK_v2.md) - Production operations
