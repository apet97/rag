# Diagnostics

Endpoint status: The Ollama-compatible endpoint at http://10.127.0.192:11434 is unreachable from this host. See ENDPOINT_CHECK.log. Proceeded with static analysis only.

Issues found and suggested fixes (file references use repo-relative paths):

1) Severity: High — Embedding dimension drift between modules
- Evidence: src/embeddings_async.py:21-29 originally parsed `EMBEDDING_DIM` from env separately from `src/embeddings.py`.
- Impact: Mixed 384/768 dimension can cause shape errors during similarity/dot products and mismatched index loads.
- Fix: Import `EMBEDDING_DIM` from `src.embeddings` as single source of truth. Implemented.
- Test: Run `codex/scripts/dims_check.py` and verify all namespaces share the same dim and match configured dim.

2) Severity: High — Hard failure when FAISS reconstruct not supported
- Evidence: src/index_manager.py:130-160 previously raised `RuntimeError` on reconstruct failure.
- Impact: Startup crash on IVF/HNSW indexes without direct map support; service unavailable instead of degrading.
- Fix: Degrade gracefully—log warning and skip embedding caching; allow vector-only search and hybrid path to fall back. Implemented.
- Test: Start with an index that doesn’t support reconstruct; verify `/health` returns ok and vector search works.

3) Severity: Medium — Inconsistent Ollama embedding endpoints between scripts
- Evidence: src/embeddings_async.py uses `/api/embeddings` with `{model, prompt}`; src/ingest_from_jsonl.py:37-58 posts to `/api/embed` with `{model, input}`.
- Impact: Ingest script may fail against Ollama depending on version; confusion when configuring.
- Fix: Standardize to `/api/embeddings` and consistent payload. Not patched (script is optional), documented here.
- Test: If using `ingest_from_jsonl.py`, align endpoint and payload.

4) Severity: Medium — ingest_from_jsonl meta format incompatible with IndexManager
- Evidence: src/ingest_from_jsonl.py:146-166 writes `meta.json` as a raw list of records, whereas IndexManager expects an object with `rows` or `chunks`, `dim`, and `normalized`.
- Impact: IndexManager will fail to load namespaces produced by this ingest script.
- Fix: Prefer `src/embed.py` for building indexes or update ingest_from_jsonl.py to emit the same meta schema as `embed.py`.
- Test: Rebuild index with `embed.py`; ensure `meta.json` contains `dim` and `rows`.

5) Severity: Low — Namespace env default can cause missing namespace errors if set
- Evidence: install.sh:274 sets `NAMESPACES=clockify,langchain`; server startup validates both (src/server.py:121-180).
- Impact: If `langchain` index isn’t present, startup fails. Server already auto-derives namespaces when env is unset.
- Fix: Recommendation: leave `NAMESPACES` unset in most environments so auto-discovery is used. No code change.
- Test: Unset `NAMESPACES` and verify `/health` shows discovered namespaces.

6) Severity: Low — Ensure embeddings never leak in responses
- Evidence: src/server.py:336-345 filters embedding in `search_ns`; src/server.py:391-406 filters in hybrid; src/server.py:576-587 filters in merge.
- Impact: If forgotten, raw vectors bloat payloads and break JSON.
- Fix: Verified and left protective `_remove_embedding_from_result` helper in use.
- Test: Exercise `/search` and `/chat` and confirm no `embedding` keys appear.

7) Severity: Low — Normalization consistency
- Evidence: src/embeddings.py L2-normalizes outputs; src/embeddings_async.py normalizes per request; embed.py normalizes index vectors before add; IndexManager’s `is_normalized` reads meta.json.
- Impact: Cosine/IP scoring consistency depends on normalization.
- Fix: Verified. Add `dims_check.py` and rely on meta `normalized` flag for observability.
- Test: `/health` should report index_normalized true; dims_check verifies dims.

Reproduction notes
- Endpoint probe: run `codex/scripts/endpoint_smoke.sh`.
- Dimension check: `python3 codex/scripts/dims_check.py`.
- KB coverage: `python3 codex/scripts/count_pages.py`.


8) Severity: Medium — ingest endpoint and meta schema (fixed)
- Evidence: src/ingest_from_jsonl.py used `/api/embed` + `embeddings` and emitted raw list as meta.
- Fix: Switched to `/api/embeddings` with `{model,prompt}` and wrote meta.json compatible with IndexManager/embed.py (`rows`, `chunks`, `dim`, `normalized`).
- Test: Re-run ingest; confirm `meta.json` has `dim` and `rows`; `dims_check.py` passes.

9) Severity: Low — Missing namespaces crash on startup (fixed)
- Evidence: src/server.py startup loop raised RuntimeError on missing indexes/metadata.
- Fix: Startup now warns and skips invalid namespaces; proceeds if at least one valid namespace exists.
- Test: Set `NAMESPACES=clockify,langchain` with only `clockify` present; server should start and load `clockify`.
