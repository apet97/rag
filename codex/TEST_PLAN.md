# Test Plan

Scope: Validate endpoint reachability, config sanity, index/embedding dimension consistency, basic KB coverage, and retrieval smokes without requiring the API server to run.

Prereqs
- Python 3.9+
- `faiss-cpu` and `sentence-transformers` installed if you want real embeddings; otherwise set `EMBEDDINGS_BACKEND=stub`.
- Index present under `index/faiss/<namespace>` with `index.bin` or `index.faiss` and `meta.json`.

1) LLM Endpoint Smoke
- Command: `bash codex/scripts/endpoint_smoke.sh`
- Expected: `ENDPOINT_CHECK.log` contains results of `/api/tags`, `/api/generate`, `/api/chat` probes; if unreachable, shows `Timeout was reached` and continues.
- Pass: No unhandled errors; log file written.

2) Config Audit
- Command: Open `codex/CONFIG_AUDIT.md`
- Verify variables match your environment and that `.env` aligns with these defaults.
- Pass: No unknown or missing critical variables.

3) KB Coverage Summary
- Command: `python3 codex/scripts/count_pages.py`
- Expected: `codex/KB_COVERAGE.md` with counts by extension, PDF page totals, largest files, suspect files.
- Pass: File written; largest files list populated.

4) Embedding Dimension Consistency
- Command: `python3 codex/scripts/dims_check.py`
- Expected: Prints per-namespace dims and any mismatches vs configured dim. Exits 1 on mixed dims, 2 if meta contains embeddings.
- Pass: All namespaces share one dim; no meta.json contains `embedding`.

5) Retrieval + Chat Smoke (optional; requires Ollama reachable and local index)
- Env: `export LLM_BASE_URL=http://10.127.0.192:11434`
- Command: `python3 codex/scripts/rag_smoke.py`
- Expected: For 5–8 queries, prints top-k titles/ids with overlap scores and latencies. Writes `codex/rag_smoke.out`.
- Pass: Each query yields at least 1–3 results; overlap scores reasonable (>0.05 for relevant docs).

6) Server Response Serialization (if API server is running)
- Start server: `uvicorn src.server:app --host 0.0.0.0 --port 7000`
- Queries: `curl -s 'http://localhost:7000/search?q=timesheet&k=5&namespace=clockify' -H 'x-api-token: change-me' | jq .`
- Check: Ensure no `embedding` appears in results; verify `latency_ms` and `metadata` present.
- Pass: No raw embeddings in JSON payload.

7) Graceful Degrade on Non-reconstructible Index
- Use/prepare an index where `index.reconstruct` is unsupported.
- Start server: `uvicorn src.server:app --host 0.0.0.0 --port 7000`
- Expected: Startup completes; logs warn that reconstruct not supported; hybrid path falls back to vector-only without crashing.
- Pass: `/health` returns ok; `/search` returns results.

