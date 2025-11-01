# Project Handoff — Clockify RAG (v3)

This document gives the next AI/tool everything needed to understand, run, and improve the Clockify RAG stack. It covers architecture, configs, artifacts, CI, quality gates, and common workflows.

- Repo: https://github.com/apet97/rag
- Default branch: main
- Current release: v3.0.0
- Release URL: see codex/RELEASE_URL.txt
- Namespace: `clockify`

## 0) Status Snapshot (Nov 1, 2025)
- Remote main (protected): up to 3195cee
- Latest changes prepared in PR: https://github.com/apet97/rag/pull/21
  - CORS defaults include https://ai.coingdevelopment.com and localhost:7001 (src/server.py:130)
  - Dev‑mode token ergonomics (accept any non‑empty token when API_TOKEN=change‑me) (src/server.py:748)
  - Prefer Clockify namespaces by default when auto‑deriving (src/server.py:88)
  - UI auto‑targets full host:7001 when UI runs on a different port (public/js/api.js:1, public/js/main-qwen.js:1)
  - Bootstrap fallback strips lxml/readability‑lxml/trafilatura on macOS/Py3.13 (scripts/run_local.sh:1)
  - Added policy files: codex/ALLOWLIST.txt, codex/DENYLIST.txt, codex/canonical_params_ignore.txt
- CI status on PR #21:
  - Static Analysis (strict): success
  - Static Analysis (strict flake8): success
  - RAG CI Pipeline: success
  - Large files check: success
  - RAG Corpus CI: success (latest run)
- Blocker to merge: 1 approving review required by branch protection

Merge steps (gh):
- gh pr review 21 --approve
- gh pr merge 21 --squash --delete-branch

## 1) What This Is
A production‑ready Retrieval‑Augmented Generation (RAG) system for Clockify help content. It provides strict allowlist enforcement, enriched corpus ingestion (v2), hybrid retrieval (BM25+vector), citations, and quality gates.

## 2) Directory Map (key paths)
- `src/` – server, retrieval, chunking, embeddings, metrics, configs
- `tools/` – ingestion and chunkers (v2, ablations)
- `codex/` – artifacts: corpus, evals, gates, runbooks, patches
- `index/` – generated FAISS indexes (ignored; keep `.gitkeep`)
- `tests/` – unit/integration tests (offline‑friendly)
- `.github/workflows/` – CI (RAG Corpus CI + strict static analyses)

## 3) Core Concepts
- Allowlist/Denylist enforced at ingestion and runtime (Clockify only)
- Enriched corpus (titles, canonical, checksums) from `codex/CRAWLED_LINKS_enriched.json`
- Hybrid retrieval: lexical@w + dense@(1-w), default `SEARCH_LEXICAL_WEIGHT=0.50`
- Namespaced indexes under `index/faiss/clockify/`
- Citations return URL and title; guard drops non‑allowlisted URLs and refills

## 4) Configuration
Environment variables are centralized and validated (see `.env.example`, `src/config.py`). Important:
- `ALLOWLIST_PATH=codex/ALLOWLIST.txt`
- `DENYLIST_PATH=codex/DENYLIST.txt`
- `ENRICHED_LINKS_JSON=codex/CRAWLED_LINKS_enriched.json`
- `NAMESPACE=clockify`
- `EMBEDDING_MODEL` and `EMBEDDING_DIM` (single source of truth)
- Chunking: `CHUNK_STRATEGY`, `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`
- Hybrid weight: `SEARCH_LEXICAL_WEIGHT=0.50`

## 5) Build + Run (offline‑first)
- Create venv, install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt -c constraints.txt`
- Ingest v2 corpus and build index:
  - `make ingest_v2`
- Offline eval + quality gates:
  - `make offline_eval`
- Serve API:
  - `uvicorn src.server:app --host 0.0.0.0 --port 7001`
  - Built‑in UI is served at `/` (http://localhost:7001)
- Smoke (when staging is reachable):
  - `make runtime_smoke BASE_URL=http://10.127.0.192:7000`

### New Tests Added (v3 incremental)
- Health endpoint contract tests: `tests/test_health_endpoints.py`
- Allowlist refill behavior (helper-backed): `tests/test_allowlist_refill.py`
- H2/H3 chunking behavior and list/code preservation: `tests/test_chunking_clockify.py`
- Citation validator unit tests: `tests/test_citation_validator.py`

Run a subset locally:
- `pytest -q tests/test_health_endpoints.py tests/test_allowlist_refill.py tests/test_chunking_clockify.py`
- `pytest -q tests/test_citation_validator.py`

## 6) Endpoints (FastAPI)
- `GET /healthz` and `GET /readyz` – report ok, namespace, index_present, index_digest, lexical_weight, chunk_strategy
- `GET /search?q=...&k=5` – hybrid search over title|h1|path fields
- `POST /chat` – full RAG with citations; allowlist guard enforced

Notes on auth and CORS:
- Dev mode: when `ENV!=prod` and `API_TOKEN=change-me`, any non‑empty `x-api-token` is accepted.
- CORS defaults include: `http://localhost:8080`, `http://localhost:7001`, and `https://ai.coingdevelopment.com` (see src/server.py:130). Override with `CORS_ALLOWED_ORIGINS` as needed.

## 7) Artifacts (codex/)
- Corpus: `CRAWLED_LINKS_enriched.json`, `ALLOWLIST.txt`, `DENYLIST.txt`
- Ingestion: `INGEST_STATS_v2.md`, `INGEST_FAILED_v2.txt`
- Evaluation: `OFFLINE_EVAL.md`, `RAG_RUNTIME_EVAL.md` (optional)
- Quality: `QUALITY_GATES.md`, `ALERTS.md`, `DASHBOARD.json`
- Provenance/Freeze: `CORPUS_FREEZE_YYYYMMDD.json`, `INDEX_DIGEST.txt`, `PROVENANCE.md`
- Ops docs: `RUNBOOK_v2.md`, `DEPLOYMENT_PLAN.md`, `ROLLBACK_PLAN.md`

## 8) CI + Branch Protection
- Required checks: “RAG Corpus CI”, “Large files check”
- RAG Corpus CI matrix 3.10/3.11; runs black/isort, flake8 (src‑scoped), mypy subset, and tests/offline eval
- PRs must be merged (no direct push to main); admins enforced

PR operations (gh):
- List runs for branch: `gh run list --limit 10 | rg <branch>`
- Rerun failing run: `gh run rerun <run-id>`
- Watch checks: `gh run watch --exit-status --interval 10`

## 9) Ingestion v2 + Chunking
- `tools/ingest_v2.py` reads `ENRICHED_LINKS_JSON`, applies allow/deny, chunks content
- Chunkers in `tools/chunkers.py`: `url_level` (default) and `h2_h3_blocks` (ablation)
- Output indexes in `index/faiss/clockify_*` and stats in `codex/INGEST_STATS_v2.md`

## 10) Retrieval & Ranking
- Title=3, h1=2, path=1 weights for lexical scoring
- Hybrid fusion combines BM25 and dense scores; weight configurable via env
- Reranker optional; fallbacks if unavailable
- Citations include title+url and are strictly allowlisted

## 11) Quality Gates
- Metadata coverage after ingest (title or h1) ≥ 90%
- Offline Hit@5 ≥ 0.85 (thin topics ≥ 0.80)
- Namespace integrity: 0 cross‑namespace leaks
- Runtime smoke: allowlist‑only citations

## 12) Make Targets
- `make ingest_v2` – build index from enriched corpus
- `make offline_eval` – compute retrieval metrics
- `make runtime_smoke` – run live smoke against staging/prod
- `make hybrid_sweep` – tune `SEARCH_LEXICAL_WEIGHT`
- `make corpus_freeze` – write freeze + index digests

## 13) Common Pitfalls
- Missing deps (e.g., `prometheus-client`) → `pip install -r requirements.txt`
- Large indexes accidentally committed → keep `index/` ignored (see `.gitignore`)
- Non‑allowlisted citations → verify allowlist regex and runtime guard
- EMBEDDING_DIM mismatch → fails fast at import in `src/config.py`
- macOS Python 3.13 + lxml builds → use `./scripts/run_local.sh` (fallback strips lxml/readability/trafilatura). For real embeddings without internet, pre‑seed the HuggingFace cache or run `./scripts/run_local_real.sh` on a connected host once.

## 14) How To Extend
- Add new integrations/docs: update allowlist rules and ingest; run `make ingest_v2`
- Tune hybrid search: run `codex/scripts/hybrid_sweep.py` → update default weight
- Switch chunking: set `CHUNK_STRATEGY=h2_h3_blocks` and rebuild

## 15) Handoff Checklist
- [ ] `.env` configured (or use `.env.example` defaults)
- [ ] Index present under `index/faiss/clockify_*`
- [ ] `OFFLINE_EVAL.md` shows Hit@5 within gates
- [ ] Release artifacts updated (`CORPUS_FREEZE_*`, `INDEX_DIGEST.txt`)
- [ ] Branch protection intact; CI green on PRs
  - For current PR #21, approve + merge when all greens present.

## 16) Links
- Release v3.0.0: see codex/RELEASE_URL.txt
- CI runs: https://github.com/apet97/rag/actions
- Docs: README.md, RUNBOOK_v2.md, DEPLOYMENT_PLAN.md, ROLLBACK_PLAN.md
