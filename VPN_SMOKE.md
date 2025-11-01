# VPN Smoke Test - Internal oss20b LLM

This document describes how to verify connectivity to the internal VPN-only LLM endpoint.

## Prerequisites

- Connected to company VPN
- LLM server running at `http://10.127.0.192:11434`
- Model `gpt-oss:20b` available on the server

## Quick Health Check (cURL)

```bash
# Set base URL
BASE="${LLM_BASE_URL:-http://10.127.0.192:11434}"

# 1. Check /api/tags endpoint (list available models)
echo "=== Checking /api/tags ==="
curl -sS -i "$BASE/api/tags"

# 2. Auto-detect model
echo -e "\n=== Auto-detecting model ==="
MODEL="$(curl -sS "$BASE/api/tags" | tr -d '\n' | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | grep -iE '^gpt-oss' | head -n1)"
[ -z "$MODEL" ] && MODEL="gpt-oss:20b"
echo "Using model: $MODEL"

# 3. Test /api/chat endpoint (ping test)
echo -e "\n=== Testing /api/chat ==="
curl -sS -i -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false}"
```

## Expected Output

### /api/tags
```json
{
  "models": [
    {
      "name": "gpt-oss:20b",
      "modified_at": "2025-10-15T12:34:56Z",
      "size": 12345678
    }
  ]
}
```

### /api/chat
```json
{
  "model": "gpt-oss:20b",
  "created_at": "2025-11-01T10:00:00Z",
  "message": {
    "role": "assistant",
    "content": "pong"
  },
  "done": true
}
```

## Python Smoke Test

```bash
# Using the oss20b provider
cd /path/to/rag
source .venv/bin/activate

python3 -c "
from src.providers.oss20b_client import OSS20BClient

client = OSS20BClient(mock=False)

# Health check
health = client.health_check()
print(f'Health: {health}')

if health['ok']:
    # Ping test
    response = client.chat([{'role': 'user', 'content': 'ping'}])
    print(f'Response: {response}')
else:
    print('ERROR: Health check failed - check VPN connection')
"
```

## Make Target Smoke Test

The repo includes a `runtime_smoke` make target that runs end-to-end tests against a deployed instance:

```bash
# Test local instance
make runtime_smoke BASE_URL=http://localhost:7001

# Test staging instance (requires VPN)
make runtime_smoke BASE_URL=http://10.127.0.192:7000
```

This runs:
1. `/healthz` check
2. `/search` endpoint test
3. `/chat` endpoint test with allowlist validation

## Troubleshooting

### Connection Refused
```
curl: (7) Failed to connect to 10.127.0.192 port 11434: Connection refused
```
**Solution:** Check VPN connection. The LLM endpoint is only accessible via company VPN.

### Timeout
```
curl: (28) Operation timed out after 30000 milliseconds
```
**Solution:**
- Verify VPN is connected and stable
- Check if LLM server is running
- Try increasing timeout with `--max-time 60`

### 404 Not Found
```
HTTP/1.1 404 Not Found
```
**Solution:**
- Verify endpoint paths: `/api/tags` and `/api/chat`
- Check LLM server configuration

### 403 Forbidden
```
HTTP/1.1 403 Forbidden
```
**Solution:**
- Verify VPN access permissions
- Check firewall rules
- Contact infra team if permissions issue

## CI/Offline Testing

For CI environments without VPN access, use mock mode:

```bash
# Enable mock mode
export MOCK_LLM=true

# Run tests (will use offline mocks)
pytest tests/test_oss20b_provider.py
```

All tests pass without VPN when `MOCK_LLM=true`.

## Integration with RAG Server

The RAG server automatically uses the oss20b provider when configured:

```bash
# .env configuration
LLM_BASE_URL=http://10.127.0.192:11434
LLM_MODEL=gpt-oss:20b
LLM_API_TYPE=ollama
MOCK_LLM=false  # Set to true for offline testing
```

Start the server:

```bash
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

The server will:
1. Auto-detect available models from `/api/tags`
2. Fall back to `gpt-oss:20b` if detection fails
3. Use mock responses if VPN is unavailable (when `MOCK_LLM=true`)

## See Also

- `.env.example` - Full configuration reference
- `codex/RUNBOOK_v2.md` - Operations runbook
- `src/providers/oss20b_client.py` - Provider implementation
- `tests/test_oss20b_provider.py` - Test suite
