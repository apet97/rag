#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Creating venv (.venv) and installing deps..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt -c constraints.txt >/dev/null

echo "[2/5] Preparing logs directory (structured JSONL at logs/retrieval_metrics.log)..."
mkdir -p logs

echo "[3/5] Building/using index (v2, url_level)."
export NAMESPACE=${NAMESPACE:-clockify}
export CHUNK_STRATEGY=${CHUNK_STRATEGY:-url_level}

# Decide embedding backend automatically unless user overrides EMBEDDINGS_BACKEND
DEFAULT_MODEL=${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}
DEFAULT_DIM=${EMBEDDING_DIM:-384}

MODEL_CACHE_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HOME/.cache/huggingface/hub}"
MODEL_CACHE_PATH="$MODEL_CACHE_HOME/models--${DEFAULT_MODEL//\//--}"

if [[ -z "${EMBEDDINGS_BACKEND:-}" ]]; then
  if [[ -d "$MODEL_CACHE_PATH" ]]; then
    export EMBEDDINGS_BACKEND=real
    export EMBEDDING_MODEL="$DEFAULT_MODEL"
    export EMBEDDING_DIM="$DEFAULT_DIM"
    echo "→ Using REAL embeddings (cache found at $MODEL_CACHE_PATH)"
  else
    export EMBEDDINGS_BACKEND=stub
    echo "→ Using STUB embeddings (no local model cache at $MODEL_CACHE_PATH)"
  fi
else
  echo "→ EMBEDDINGS_BACKEND preset to '$EMBEDDINGS_BACKEND'"
fi

# If a prebuilt index exists, reuse it; otherwise build one now
INDEX_DIR="$ROOT_DIR/index/faiss/${NAMESPACE}_url"
META_JSON="$INDEX_DIR/meta.json"
if [[ -f "$META_JSON" ]]; then
  echo "Found prebuilt index at $INDEX_DIR — skipping ingest"
else
  echo "Building index into $INDEX_DIR ..."
  if ! python3 tools/ingest_v2.py; then
    echo "Ingest failed with current backend. Falling back to STUB embeddings and retrying..."
    export EMBEDDINGS_BACKEND=stub
    python3 tools/ingest_v2.py
  fi
fi

echo "[4/5] Using internal LLM at 10.127.0.192:11434 (no API key). Autodetecting gpt-oss model at runtime..."
export LLM_BASE_URL=${LLM_BASE_URL:-http://10.127.0.192:11434}

echo "[5/5] Starting API on http://localhost:7001 ..."
echo "Metrics JSONL: $ROOT_DIR/logs/retrieval_metrics.log"
exec uvicorn src.server:app --host 0.0.0.0 --port 7001
