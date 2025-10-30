#!/usr/bin/env bash
set -euo pipefail
BASE="http://10.127.0.192:11434"
{
  date
  echo "Checking tags…"
  curl -sS --connect-timeout 1 -m 3 "$BASE/api/tags" || echo "tags failed"
  echo
  echo "Test generate…"
  curl -sS --connect-timeout 1 -m 5 -X POST "$BASE/api/generate" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-20b","prompt":"hello","stream":false}' || echo "generate failed"
  echo
  echo "Test chat…"
  curl -sS --connect-timeout 1 -m 5 -X POST "$BASE/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"hello"}],"stream":false}' || echo "chat failed"
} > ~/Downloads/rag/codex/ENDPOINT_CHECK.log 2>&1
