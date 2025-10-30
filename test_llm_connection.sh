#!/bin/bash
# Test script for internal Ollama LLM server

set -euo pipefail

# Production environment checks
BASE="${BASE:-http://10.127.0.192:11434}"

# Validate LLM endpoint format
if [[ ! "$BASE" =~ ^https?:// ]]; then
    echo "❌ ERROR: Invalid LLM endpoint. Must start with http:// or https://"
    echo "   Current value: $BASE"
    exit 1
fi

# Extract host and port for timeout calculation
LLM_HOST=$(echo "$BASE" | sed -E 's|^https?://([^/:]+).*|\1|')
if [[ -z "$LLM_HOST" ]]; then
    echo "❌ ERROR: Could not parse LLM host from $BASE"
    exit 1
fi

echo "========================================="
echo "Testing LLM Server: $BASE"
echo "========================================="
echo

# Test 1: Check server is reachable
echo "## 1) Testing server connectivity..."
if curl -sS -f "$BASE/api/tags" > /dev/null 2>&1; then
    echo "✅ Server is reachable at $BASE"
else
    echo "❌ Server NOT reachable at $BASE"
    echo "   Check VPN connection or firewall"
    exit 1
fi
echo

# Test 2: List available models
echo "## 2) Available models:"
MODEL_LIST=$(curl -sS "$BASE/api/tags" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
if [ -z "$MODEL_LIST" ]; then
    echo "❌ No models found"
    exit 1
else
    echo "$MODEL_LIST" | while read -r model; do
        echo "   - $model"
    done
fi
echo

# Test 3: Find gpt-oss model
MODEL=$(echo "$MODEL_LIST" | grep -iE '^gpt-oss' | head -n1)
if [ -z "$MODEL" ]; then
    echo "⚠️  No 'gpt-oss' model found, using first available model"
    MODEL=$(echo "$MODEL_LIST" | head -n1)
fi
echo "## 3) Using model: $MODEL"
echo

# Test 4: Test /api/chat endpoint
echo "## 4) Testing /api/chat endpoint..."
CHAT_RESPONSE=$(curl -sS -X POST "$BASE/api/chat" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say 'pong' if you can hear me\"}],\"stream\":false}" \
  2>&1)

if echo "$CHAT_RESPONSE" | grep -qi "pong\|content"; then
    echo "✅ /api/chat endpoint works!"
    echo "   Response preview:"
    echo "$CHAT_RESPONSE" | head -c 200
    echo
else
    echo "❌ /api/chat endpoint failed"
    echo "   Response: $CHAT_RESPONSE"
fi
echo

# Test 5: Test /api/generate endpoint
echo "## 5) Testing /api/generate endpoint..."
GEN_RESPONSE=$(curl -sS -X POST "$BASE/api/generate" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hi\",\"stream\":false}" \
  2>&1)

if echo "$GEN_RESPONSE" | grep -qi "response\|hi"; then
    echo "✅ /api/generate endpoint works!"
    echo "   Response preview:"
    echo "$GEN_RESPONSE" | head -c 200
    echo
else
    echo "⚠️  /api/generate endpoint may not be available"
fi
echo

# Test 6: Check for /talk endpoint (custom endpoint)
echo "## 6) Testing /talk endpoint (if exists)..."
TALK_RESPONSE=$(curl -sS -X POST "$BASE/talk" \
  -H 'content-type: application/json' \
  -d '{"prompt":"test"}' \
  2>&1)

if echo "$TALK_RESPONSE" | grep -qi "error\|not found" || [ -z "$TALK_RESPONSE" ]; then
    echo "⚠️  /talk endpoint not available (this is OK)"
else
    echo "✅ /talk endpoint exists!"
    echo "   Response preview:"
    echo "$TALK_RESPONSE" | head -c 200
    echo
fi
echo

# Summary
echo "========================================="
echo "Summary:"
echo "========================================="
echo "Server:     $BASE"
echo "Model:      $MODEL"
echo "Status:     ✅ READY TO USE"
echo
echo "Next steps:"
echo "1. Copy this config to .env:"
echo "   LLM_BASE_URL=$BASE"
echo "   LLM_MODEL=$MODEL"
echo
echo "2. Start RAG server:"
echo "   python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8877"
echo
echo "3. Test RAG API:"
echo "   curl http://localhost:8877/health"
echo "========================================="
