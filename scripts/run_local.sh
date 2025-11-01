#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Creating venv (.venv) and installing deps..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt -c constraints.txt >/dev/null

echo "[2/4] Building index (v2, url_level). Falls back to stub embeddings if real model unavailable..."
export NAMESPACE=${NAMESPACE:-clockify}
export CHUNK_STRATEGY=${CHUNK_STRATEGY:-url_level}
python3 tools/ingest_v2.py

echo "[3/4] Using internal LLM at 10.127.0.192:11434 (no API key). Autodetecting gpt-oss model at runtime..."
export LLM_BASE_URL=${LLM_BASE_URL:-http://10.127.0.192:11434}

echo "[4/4] Starting API on http://localhost:7001 ..."
exec uvicorn src.server:app --host 0.0.0.0 --port 7001

