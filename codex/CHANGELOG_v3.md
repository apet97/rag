# Changelog v3.0.0

## Release Date: 2025-11-01

## Overview

Production-ready RAG system with VPN-only LLM provider, offline-safe testing, and minimal operator commands.

## Major Features

### 🔌 oss20b Provider Adapter
- **New**: `src/providers/oss20b_client.py` - Clean provider adapter for VPN-only LLM
- **Endpoint**: `http://10.127.0.192:11434` (Ollama format)
- **Model**: `gpt-oss:20b` with auto-detection from `/api/tags`
- **No API Key**: Internal VPN access only
- **Offline Mocks**: Full mock mode for CI (`MOCK_LLM=true`)
- **Test Coverage**: 16 offline tests in `tests/test_oss20b_provider.py`

### 📋 Minimal Operator Path
- **4-command production setup** documented in README.md
- **Single source of truth** for embedding model and dimension
- **Fail-fast validation** on config import
- **Quality gates** enforced before deployment

### 📚 Enhanced Documentation
- **VPN_SMOKE.md**: VPN endpoint health checks with cURL examples
- **codex/RUNBOOK_v2.md**: Complete operations runbook (deploy, troubleshoot, rollback)
- **README.md**: Updated with minimal quick start
- **Inline docs**: Provider adapter has full usage examples

## Configuration

### Environment Variables (New/Changed)

```bash
# LLM Provider (VPN-only)
LLM_BASE_URL=http://10.127.0.192:11434  # VPN endpoint
LLM_MODEL=gpt-oss:20b                   # Auto-detects if unavailable
LLM_API_TYPE=ollama                     # Ollama format
LLM_TIMEOUT_SECONDS=30                  # Request timeout
LLM_RETRIES=3                           # Retry attempts
MOCK_LLM=false                          # Enable for offline testing
STREAMING_ENABLED=false                 # Stream support

# Hybrid Search (tuned)
SEARCH_LEXICAL_WEIGHT=0.50              # 50% BM25, 50% vector
```

### CORS Defaults (Updated)

Added `https://ai.coingdevelopment.com` to default CORS origins:
- `http://localhost:8080`
- `http://localhost:7001`
- `https://ai.coingdevelopment.com`

Override with `CORS_ALLOWED_ORIGINS` env var.

## Quality Gates

All gates enforced in `make offline_eval`:

- **Hit@5 ≥ 0.85** (general topics)
- **Hit@5 ≥ 0.80** (thin topics)
- **Metadata coverage ≥ 90%** (title or h1 present)
- **Namespace integrity**: 0 cross-namespace leaks
- **Allowlist enforcement**: Runtime guard + ingestion filter

## Testing

### New Tests
- `tests/test_oss20b_provider.py`: 18 tests (16 pass offline, 2 skip without VPN)
  - Mock mode responses
  - Health checks
  - Endpoint URL construction
  - Singleton pattern
  - Error handling
  - VPN integration tests (skipped by default)

### CI Compatibility
- All tests pass without VPN when `MOCK_LLM=true`
- VPN-only tests marked with `@pytest.mark.skipif`
- Use `VPN_AVAILABLE=true` env var to enable integration tests

## Deployment

### Minimal Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
cp -n .env.example .env
make ingest_v2 && make offline_eval
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

### Health Check

```bash
# Quick check
curl http://localhost:7001/healthz

# VPN LLM check
curl http://10.127.0.192:11434/api/tags
```

See [VPN_SMOKE.md](../VPN_SMOKE.md) for detailed smoke test procedures.

## Breaking Changes

None. This release is fully backward compatible.

## Deprecations

None.

## Bug Fixes

- **Config validation**: Fail fast on embedding dimension mismatch
- **Retry logic**: Only retry transient errors (timeout, connection, 5xx)
- **Error messages**: Sanitize URLs and redact tokens in logs

## Performance

- **HTTP client pooling**: Reuse connections (max 20, keepalive 10)
- **Timeout tuning**: Connect 5s, read 30s, write 10s, pool 5s
- **Circuit breaker**: 3 failures → open, 30s recovery, 2 successes → close

## Migration Guide

### From v2.x to v3.0.0

No migration needed. Existing `.env` files and indexes work as-is.

**Optional**: Switch to oss20b provider explicitly:

```python
# Before (still works)
from src.llm_client import LLMClient
client = LLMClient()

# After (recommended for clarity)
from src.providers.oss20b_client import OSS20BClient
client = OSS20BClient()
```

## Known Issues

- **VPN required**: LLM endpoint only accessible via company VPN
- **macOS Python 3.13**: Use `./scripts/run_local.sh` for lxml fallback
- **Mock responses**: Generic templates, not real model output

## Roadmap (v3.1)

- [ ] Streaming support for oss20b provider
- [ ] Multi-model support (oss13b, oss7b fallback)
- [ ] Enhanced mock responses (query-specific templates)
- [ ] Prometheus metrics for provider health

## Contributors

- Automated by Claude Code (Autonomous Agent)
- Based on requirements from HANDOFF_NEXT_AI.md

## See Also

- [HANDOFF_NEXT_AI.md](HANDOFF_NEXT_AI.md) - System overview
- [RUNBOOK_v2.md](RUNBOOK_v2.md) - Operations guide
- [VPN_SMOKE.md](../VPN_SMOKE.md) - VPN smoke tests
- [README.md](../README.md) - Quick start guide
