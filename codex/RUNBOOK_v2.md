# RAG Operations Runbook v2

Production operations guide for the Clockify RAG system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Build & Deploy](#build--deploy)
4. [Health Checks](#health-checks)
5. [Troubleshooting](#troubleshooting)
6. [Quality Gates](#quality-gates)
7. [Rollback](#rollback)

## Quick Start

### Minimal Production Path

```bash
# Prerequisites: Python 3.10+, git, VPN access for LLM

# 1) Clone and setup
git clone https://github.com/apet97/rag && cd rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt

# 2) Configure
cp -n .env.example .env
# Edit .env if needed (defaults work for VPN setup)

# 3) Build index
make ingest_v2

# 4) Verify quality
make offline_eval

# 5) Start server
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

Access UI at http://localhost:7001

## Environment Setup

### Required Environment Variables

The system uses sensible defaults from `.env.example`. Key variables:

```bash
# API Server
ENV=dev                           # or 'prod'
API_PORT=7001
API_HOST=0.0.0.0
API_TOKEN=change-me               # Change in production!

# LLM (VPN-only, no API key)
LLM_BASE_URL=http://10.127.0.192:11434
LLM_MODEL=gpt-oss:20b
LLM_API_TYPE=ollama
LLM_TIMEOUT_SECONDS=30
MOCK_LLM=false                    # Set to 'true' for offline testing

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# Search
SEARCH_LEXICAL_WEIGHT=0.50        # 50% BM25, 50% vector
RETRIEVAL_K=20

# Namespaces
NAMESPACE=clockify
CHUNK_STRATEGY=url_level

# Policy
ALLOWLIST_PATH=codex/ALLOWLIST.txt
DENYLIST_PATH=codex/DENYLIST.txt
ENRICHED_LINKS_JSON=codex/CRAWLED_LINKS_enriched.json
```

### CORS Configuration

For production, set explicit allowed origins:

```bash
CORS_ALLOWED_ORIGINS=https://app.example.com,https://api.example.com
```

Default includes `localhost:7001`, `localhost:8080`, and `https://ai.coingdevelopment.com`.

## Build & Deploy

### 1. Ingestion (Build Index)

```bash
# Build FAISS index from enriched corpus
make ingest_v2

# Output: index/faiss/clockify_url/
# Stats: codex/INGEST_STATS_v2.md
```

Expected output:
- `index/faiss/clockify_url/index.bin` (FAISS index)
- `index/faiss/clockify_url/meta.json` (metadata)
- Stats showing chunks, coverage, allowlist enforcement

### 2. Quality Gates (Offline Eval)

```bash
# Run offline evaluation against test queries
make offline_eval

# Output: codex/OFFLINE_EVAL.md
```

Quality thresholds:
- **Hit@5 ≥ 0.85** (general topics)
- **Hit@5 ≥ 0.80** (thin topics)
- **Metadata coverage ≥ 90%** (title or h1 present)
- **No namespace leaks**

### 3. Runtime Smoke (Staging/Prod)

```bash
# Smoke test against deployed instance
make runtime_smoke BASE_URL=http://10.127.0.192:7000

# Checks:
# - /healthz endpoint
# - /search functionality
# - /chat with allowlist enforcement
```

### 4. Corpus Freeze (Release)

```bash
# Freeze corpus state for release
make corpus_freeze

# Output:
# - codex/CORPUS_FREEZE_YYYYMMDD.json
# - codex/INDEX_DIGEST.txt
# - codex/PROVENANCE.md
```

## Health Checks

### Endpoint: GET /healthz

```bash
curl http://localhost:7001/healthz
```

Response:
```json
{
  "ok": true,
  "namespace": "clockify",
  "index_present": true,
  "index_digest": "sha256:abc123...",
  "search_lexical_weight": 0.5,
  "chunk_strategy": "url_level",
  "embedding_dim": 384
}
```

### Endpoint: GET /readyz

Same as `/healthz` but used for k8s readiness probes.

### Deep Health Check

```bash
curl 'http://localhost:7001/health?deep=1'
```

Checks LLM connectivity and circuit breaker status.

### VPN LLM Health

```bash
# Manual check
curl -i http://10.127.0.192:11434/api/tags

# Or use the provider
python3 -c "
from src.providers.oss20b_client import OSS20BClient
client = OSS20BClient(mock=False)
print(client.health_check())
"
```

See [VPN_SMOKE.md](../VPN_SMOKE.md) for detailed smoke test procedures.

## Troubleshooting

### LLM Connection Issues

**Symptom**: `/health` shows `llm_ok: false`

**Cause**: VPN disconnected or LLM server unreachable

**Fix**:
1. Check VPN connection
2. Verify LLM endpoint: `curl http://10.127.0.192:11434/api/tags`
3. Check firewall rules
4. Enable mock mode for testing: `export MOCK_LLM=true`

### Index Not Found

**Symptom**: `/healthz` shows `index_present: false`

**Cause**: Index not built or wrong namespace

**Fix**:
```bash
# Rebuild index
make ingest_v2

# Check namespace matches env
echo $NAMESPACE
ls -la index/faiss/
```

### Embedding Dimension Mismatch

**Symptom**: `RuntimeError: Embedding dimension mismatch`

**Cause**: Index built with different embedding model

**Fix**:
```bash
# Verify config
echo "Model: $EMBEDDING_MODEL"
echo "Dim: $EMBEDDING_DIM"

# Rebuild index with correct model
make ingest_v2
```

### CORS Errors

**Symptom**: Browser shows CORS error

**Cause**: Origin not in allowed list

**Fix**:
```bash
# Add origin to .env
CORS_ALLOWED_ORIGINS=http://localhost:8080,https://app.example.com

# Or for dev (temporary):
export CORS_ALLOWED_ORIGINS=http://localhost:8080
```

### Non-Allowlisted Citations

**Symptom**: Search returns external URLs

**Cause**: Allowlist not enforced or index contains old data

**Fix**:
```bash
# Check allowlist
cat codex/ALLOWLIST.txt

# Rebuild index (enforces allowlist)
make ingest_v2

# Verify runtime guard
grep -A5 "def _is_allowed" src/server.py
```

### Tests Failing on CI

**Symptom**: CI tests fail due to missing LLM

**Cause**: VPN not available in CI

**Fix**:
```bash
# CI should set mock mode
export MOCK_LLM=true

# Tests should skip VPN-only tests
pytest -v -m "not vpn"
```

## Quality Gates

### Pre-Deploy Checklist

- [ ] `make ingest_v2` succeeds
- [ ] `make offline_eval` passes all gates
- [ ] `OFFLINE_EVAL.md` shows Hit@5 ≥ 0.85
- [ ] `/healthz` returns `ok: true`
- [ ] `/search?q=create+project` returns results
- [ ] `/chat` returns citations from allowlist only
- [ ] VPN smoke test passes (if applicable)

### Post-Deploy Verification

```bash
# Health
curl http://<host>:7001/healthz

# Search
curl -H 'x-api-token: <token>' \
  'http://<host>:7001/search?q=create+project&k=5'

# Chat (check citations)
curl -X POST http://<host>:7001/chat \
  -H 'Content-Type: application/json' \
  -H 'x-api-token: <token>' \
  -d '{"query": "How do I create a project?", "namespace": "clockify"}'
```

### Monitoring Metrics

Prometheus metrics at `/metrics`:
- `rag_requests_total` - Total requests by endpoint
- `rag_request_duration_seconds` - Latency histogram
- `rag_cache_hits_total` - Cache hit rate
- `rag_circuit_breaker_state` - Circuit breaker status

## Rollback

### Rollback Index

```bash
# List available corpus freezes
ls -lh codex/CORPUS_FREEZE_*.json

# Restore from freeze
cp codex/CORPUS_FREEZE_20251025.json codex/CRAWLED_LINKS_enriched.json

# Rebuild index
make ingest_v2
```

### Rollback Code

```bash
# Git rollback
git log --oneline -10
git checkout <previous-commit>

# Reinstall deps
pip install -r requirements.txt -c constraints.txt

# Rebuild
make ingest_v2
make offline_eval

# Restart server
pkill -f uvicorn
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

### Emergency: Revert to Last Green PR

```bash
# Find last green commit
gh run list --branch main --limit 10

# Checkout green commit
git checkout <green-sha>

# Rebuild
make ingest_v2

# Restart
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

## See Also

- [HANDOFF_NEXT_AI.md](HANDOFF_NEXT_AI.md) - System overview and handoff
- [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) - Deployment procedures
- [ROLLBACK_PLAN.md](ROLLBACK_PLAN.md) - Rollback procedures
- [VPN_SMOKE.md](../VPN_SMOKE.md) - VPN smoke tests
- [QUALITY_GATES.md](QUALITY_GATES.md) - Quality gate definitions
