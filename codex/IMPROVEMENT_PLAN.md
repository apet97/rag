# Codebase Review and Improvement Plan (v3)

This document summarizes an end‑to‑end review of the repository and proposes a prioritized, low‑risk improvement plan. It focuses on correctness, maintainability, retrieval quality, runtime stability, and CI hygiene.

## Executive Summary
- Architecture is solid: strict allowlist corpus policy, enriched metadata ingestion, hybrid retrieval, citations, and ops/runbooks are in place.
- CI and branch protection are configured; main is stable. We added small mypy fixes to reduce static analysis noise without behavior changes.
- Next steps prioritize stricter typing and lint coverage in `src/`, retrieval quality guardrails, and test expansion for runtime behaviors.

## Strengths
- Clear ingestion v2 with allow/deny rules and metadata enrichment.
- Hybrid retrieval (BM25 + dense) with tunable weights; chunking ablation strategies.
- Runtime allowlist guard and citation shaping (title + URL).
- Ops maturity: runbooks, gates, offline eval, runtime smoke, freeze/digest artifacts.

## Gaps and Risks
- Type safety: several modules still permissive (mypy set to non‑strict). Risk of silent regressions.
- Lint coverage: flake8 restricted to src; tests/eval/scripts still have issues (intended exclusion for now).
- Tests: runtime guard tests exist but could be expanded (allowlist drop/refill, health endpoints contract, index namespace integrity).
- Performance tuning: Hybrid weight default chosen; recommend periodic sweeps and reporting.

## Recommended Improvements (Progress tracked)

### 1) Type Safety (Short Term)
- DONE: Addressed easy wins and made `mypy src` clean (non‑strict global config):
  - `embeddings_async.py` (Callable), `server.py` (test helper), `query_decomposition.py` (no‑any‑return), `tuning_config.py` (float cast), `llm_client.py` (circuit breaker returns)
- NEXT: Expand to strict subsets: `retrieval_engine.py`, `search_improvements.py`, `scoring.py`, and selected server endpoints with per‑module strict config.

### 2) Lint and Imports (Short Term)
- Clean unused imports/variables across `src/` to allow enabling basic F401/F841 for src in CI without noise.
- Keep `tests/`, `eval/`, and `scripts/` excluded until staged cleanup to avoid blocking.

### 3) Tests (Short→Medium)
- DONE: Unit tests for health endpoints, allowlist drop/refill (helper), and H2/H3 chunking behavior.
- NEXT: Add end‑to‑end negative test for non‑allowlisted citations in chat (mock retrieval), and validate citation shape explicitly at chat layer.

### 4) Retrieval Quality (Medium)
- Periodic `hybrid_sweep.py` runs and update default `SEARCH_LEXICAL_WEIGHT` based on `HYBRID_TUNING.md` best Hit@5 then MRR.
- Topic‑targeted evals to raise thin‑topic coverage (SSO/SAML/SCIM, webhooks, rate limits).

### 5) Performance + Monitoring (Medium)
- Confirm JSONL metrics include: `retrieval_latency_ms`, `num_candidates`, `lexical_weight`, `chunk_strategy`; snapshot weekly into `codex/PERF_BASELINE.md`.
- Introduce basic percentile alerts (p50/p95 thresholds already captured in `ALERTS.md`).

### 6) Security + Policy (Medium)
- Keep allowlist enforced at retrieval; log dropped citations with reason.
- Consider bandit and pip‑audit as enforced CI steps with allowlisted ignores for false positives.

### 7) Docs & DX (Short)
- Handoff is added: `codex/HANDOFF_NEXT_AI.md`.
- Link improvement plan from README or docs index; keep it updated alongside CI changes.

## 30/60/90‑Day Checklist

- 30 days
  - Expand strict mypy to `retrieval_engine`, `search_improvements`, `scoring`, server health/config.
  - Add allowlist guard and citation shape tests.
  - Run `hybrid_sweep.py`; update default weight if improvement ≥+0.01 Hit@5.

- 60 days
  - Clean unused imports/vars across `src`; enable F401/F841 rules for src in CI.
  - Add long‑page chunking tests (h2/h3 boundaries).
  - Refresh `OFFLINE_EVAL.md` and gates.

- 90 days
  - Evaluate `h2_h3_blocks` as default for long pages if it improves Hit@5 without latency regressions.
  - Harden runtimes: tighten timeouts, confirm circuit‑breaker trip/restore behavior in live smoke.

## Current Changes (merged)
- Small mypy fixes (no behavior change):
  - `src/tuning_config.py`: precise return typing and float cast for decay.
  - `src/chunkers/clockify.py`: cast BeautifulSoup `get_text` to `str`.
  - `src/retrieval_engine.py`: typed pytz import to silence import‑untyped noise.
  - `src/llm_client.py`: return types safe with cast; explicit `str(resp.text)`.
  - `src/query_decomposition.py`: lazy import `LLMClient` to avoid type assignment errors.
- Tests added:
  - `tests/test_health_endpoints.py`, `tests/test_allowlist_refill.py`, `tests/test_chunking_clockify.py`

## Links
- Repo: https://github.com/apet97/rag
- CI: https://github.com/apet97/rag/actions
- Handoff: `codex/HANDOFF_NEXT_AI.md`
- This plan: `codex/IMPROVEMENT_PLAN.md`
