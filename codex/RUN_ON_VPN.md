# Company Laptop Quickstart (VPN)

These steps assume you are on the company VPN and can reach the internal LLM at `http://10.127.0.192:11434`. You do not need public internet for serving; model downloads during ingest may require access to model hubs. If model downloads are restricted, see the pre‑download note at the end.

## 0) Clone and set up Python
```bash
cd ~/Downloads
git clone https://github.com/apet97/rag.git
cd rag
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## 1) Configure environment
Either copy the example env or export directly.

```bash
cp .env.example .env
# Edit .env and ensure:
# LLM_BASE_URL=http://10.127.0.192:11434
# LLM_MODEL=gpt-oss:20b
# LLM_API_TYPE=ollama
# LLM_USE_HARMONY=auto   # or true
# API_TOKEN=<choose-a-secure-token>
# SEARCH_LEXICAL_WEIGHT=0.50
```

## 2) Build index (real embeddings)
Choose a model and matching dimension. Two recommended options:
- `sentence-transformers/all-MiniLM-L6-v2` (dim 384) — compact and fast.
- `intfloat/multilingual-e5-base` (dim 768) — multilingual.

```bash
# Option A: MiniLM (384)
export EMBEDDINGS_BACKEND=real
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_DIM=384
export NAMESPACE=clockify
export CHUNK_STRATEGY=url_level
python3 tools/ingest_v2.py   # writes index/faiss/clockify_url/

# (Optional) H2/H3 ablation index for long pages
export CHUNK_STRATEGY=h2_h3_blocks
python3 tools/ingest_v2.py   # writes index/faiss/clockify_h23/
```

## 3) Run the API
```bash
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

Health and smoke:
```bash
curl http://localhost:7001/healthz
curl http://localhost:7001/readyz
curl -H 'x-api-token: <your-token>' \
  'http://localhost:7001/search?q=create%20project&namespace=clockify_url&k=5'
```

## 4) Optional — Hybrid tuning on VPN
Once the index is built, run a sweep with real query embeddings to find the best lexical weight.

```bash
python3 codex/scripts/hybrid_sweep.py \
  --index-root index/faiss \
  --namespace clockify_url \
  --eval codex/RAG_EVAL_TASKS.jsonl \
  --out codex/HYBRID_TUNING.md \
  --query-backend real

sed -n '1,160p' codex/HYBRID_TUNING.md
```
If Hit@5 improves by ≥ 0.01 for a different weight, update `SEARCH_LEXICAL_WEIGHT` in `.env` and add a note to `codex/HYBRID_TUNING.md`.

## 5) One‑command bootstrap (optional)
If you prefer a scripted flow on VPN:
```bash
./scripts/bootstrap.sh
make ingest_v2
make serve
```

## Pre‑download note (if model download is blocked)
- Build the index once on a machine with internet access (using the commands above).
- Copy the resulting `index/faiss/clockify_url/` directory to your VPN laptop.
- You can now run the API and tuning locally without downloading models.

