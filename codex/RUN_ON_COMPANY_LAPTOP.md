# Company Laptop Quickstart (Offline)

These instructions assume no VPN and no internet access. You will run everything fully offline using the stub embedding backend and local indexes.

## 0) Prereqs
- Python 3.9+ available on the laptop
- Git clone of the repo on disk
- No internet/VPN required for the default flow below

```bash
# From your home directory (example)
cd ~/Downloads
git clone https://github.com/apet97/rag.git   # or use the SSH URL if you prefer
cd rag
```

## 1) Create virtualenv and install deps
```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## 2) Build local indexes (stub, offline)
This builds a small FAISS index from the enriched URLs without downloading any model files.

```bash
export EMBEDDINGS_BACKEND=stub
export NAMESPACE=clockify
export CHUNK_STRATEGY=url_level
python3 tools/ingest_v2.py   # writes index/faiss/clockify_url/
```

Optional: also build the h2/h3 ablation index (for long help pages)
```bash
export CHUNK_STRATEGY=h2_h3_blocks
python3 tools/ingest_v2.py   # writes index/faiss/clockify_h23/
```

## 3) Run the API (offline)
```bash
# Use the stub index (url-level). The default port in README examples is 7001
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

Open another terminal for quick health checks:
```bash
curl http://localhost:7001/healthz
curl http://localhost:7001/readyz
```

## 4) Try search and chat locally
The API enforces a token. For development the default is `change-me`.

```bash
# Search
curl -H 'x-api-token: change-me' 'http://localhost:7001/search?q=create%20project&namespace=clockify_url&k=5'

# Chat (offline mode fabricates an answer with [1] source for smoke)
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: change-me' \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I create a project?","k":2,"namespace":"clockify_url"}'
```

## 5) Run tests (offline)
```bash
# Health, allowlist refill, chunking, citations
pytest -q \
  tests/test_health_endpoints.py \
  tests/test_allowlist_refill.py \
  tests/test_chunking_clockify.py \
  tests/test_citation_validator.py

# End-to-end smoke against the stub index
pytest -q tests/test_endpoints_e2e.py
```

## 6) Offline hybrid tuning (optional)
Compare lexical/dense weight settings using the stub index (no network needed).

```bash
python3 codex/scripts/hybrid_sweep.py \
  --index-root index/faiss \
  --namespace clockify_url \
  --eval codex/RAG_EVAL_TASKS.jsonl \
  --out codex/HYBRID_TUNING.md \
  --query-backend stub

# See results:
sed -n '1,160p' codex/HYBRID_TUNING.md
```

If another weight improves Hit@5 by ≥ 0.01, update `SEARCH_LEXICAL_WEIGHT` in `.env` and
note the decision in `codex/HYBRID_TUNING.md`.

## 7) Real encoder (optional, requires one-time internet)
If you want true semantic metrics and better relevance:
1) On any internet-connected machine, create a venv and install the compatibility stack:
   - `transformers==4.30.2` `tokenizers==0.13.3` `huggingface_hub==0.14.1`
2) `EMBEDDINGS_BACKEND=real` and run `python tools/ingest_v2.py`; this pulls the model once
   and writes the FAISS+meta index to `index/faiss/clockify_url/`.
3) Copy the `index/faiss/clockify_url/` directory to your company laptop.
4) Run the hybrid sweep locally (offline) — it will use the on-disk vectors; no downloads.

## 8) Common envs
- `EMBEDDINGS_BACKEND=stub` (offline) or `real` (requires pre-downloaded model)
- `NAMESPACE=clockify` (ingest) and `namespace` request parameter set to `clockify_url`
- `API_TOKEN` default is `change-me` (dev only)
- `SEARCH_LEXICAL_WEIGHT` default remains as committed; adjust after tuning

## 9) Troubleshooting
- If `faiss-cpu` build fails on macOS arm, ensure you’re using the provided constraints and a clean venv.
- If you see network attempts, set `HF_HUB_OFFLINE=1` and confirm `EMBEDDINGS_BACKEND=stub`.
- The server logs to `logs/retrieval_metrics.log` with JSONL metrics; safe to inspect offline.

