**Summary**
- Purpose-fit Clockify RAG with strong retrieval stack (hybrid vector+BM25 via RRF, MMR diversity, time-decay), solid grounding (inline citations + citation validator), answerability checks, semantic caching, query decomposition, and operational hardening (timeouts, retries, circuit breaker, metrics).
- Overall readiness: high for a help-center AI agent. A few surgical improvements will lift precision, determinism, and maintainability.

**Retrieval & Indexing**
- Hybrid retrieval with RRF fusion: `src/retrieval_engine.py:372` `_fuse_results(...)` uses Reciprocal Rank Fusion (RRF_K_CONSTANT) to merge vector and BM25 without tuning.
- MMR diversity to reduce redundancy: `src/retrieval_engine.py:423` `_apply_diversity_penalty(...)` balances relevance and diversity.
- BM25 with per-namespace LRU cache and locking: `src/retrieval_engine.py:186` (cache) and thread-safe `get_scores()` usage.
- Vector embeddings reconstructed and cached at startup: `src/index_manager.py:128` `index.reconstruct(i)` and embedded into in-memory chunk dicts (`ensure_loaded`: `src/index_manager.py:60`).
- Deterministic, E5-correct embeddings (prefixes + L2 norm): `src/embeddings.py:108` `embed_query`, `src/embeddings.py:26` `EMBEDDING_MODEL`.

**Chunking & Domain Signals**
- HTML-aware chunking for Clockify help: `src/chunkers/clockify.py:60` parses h2/h3, preserves lists/tables, attaches breadcrumbs/anchors/updated_at.
- Time-decay uses updated_at to boost freshness.

**Query Understanding**
- Query decomposition with heuristics + fast LLM fallback: `src/query_decomposition.py:342` and `_llm_decompose_fallback`: `src/query_decomposition.py:269`.
- Synonym/glossary expansion: `src/query_expand.py` and structured weighted variants.
- Adaptive retrieval breadth by query type: `src/search_improvements.py` (detect type, adaptive k, field boosts).

**Grounding, Citations, Answerability**
- Strict system prompts for Clockify: `src/prompt.py:13` `SYSTEM_PROMPT_CLOCKIFY` and server prompt mirrors this.
- Inline citations and numbered Sources; validator enforces format: `src/citation_validator.py:114` `validate_citations`.
- Answerability (hallucination guard) with Jaccard overlap and refusal fallback: `src/llm_client.py:116` `compute_answerability_score`, used in chat: `src/server.py:1307`.

**LLM Client (oss20b)**
- Default model is `gpt-oss:20b` (overridable): `src/llm_client.py:221`; Local client defaults to `oss20b`: `src/llm/local_client.py:20`.
- Hardened httpx client: timeouts, limits, retries on transient errors, circuit breaker.
- Streaming supported behind `STREAMING_ENABLED` flag.

**Performance & Ops**
- Performance tracker with stage metrics and /perf endpoint: `src/performance_tracker.py` and `src/server.py` (/perf, /metrics, /health, /ready).
- Semantic answer cache keyed by embedding fingerprint + doc ids: `src/semantic_cache.py`.
- FastAPI app with explicit CORS and token, namespace auto-derivation from index on disk: `src/server.py`.

**Tests & Eval**
- Retrieval and chat tests, latency budgets, LLM health, decomposition coverage: `tests/`.
- Evals with goldset and diagnostics: `eval/README_EVAL.md` and scripts.

**What’s Strong**
- Hybrid retrieval done right (RRF + MMR) and BM25 caching improves quality/latency.
- Domain-aware chunking (breadcrumbs/anchors) boosts answer navigation and citation UX.
- Solid grounding stack: inline citations + validator + answerability threshold + “Not in docs” guard.
- Deterministic embeddings (E5 prefixes + L2 norm) and deterministic generation defaults (temperature=0.0).
- Operational hardening: circuit breaker, retries, timeouts, metrics, semantic cache.

**Gaps / Risks (Impact → Fix)**
- MMR similarity uses token overlap, not vectors (moderate → upgrade): compute candidate-candidate similarity with embeddings already cached on chunks inside `RetrievalResult` pipeline; fallback to token overlap only when embeddings absent.
- Prompt duplication (low → refactor): server builds prompts manually while `RAGPrompt` exists. Unify on one builder to avoid drift.
- Model naming inconsistency (low → config): defaults use `gpt-oss:20b`, elsewhere `oss20b`. Standardize via env `LLM_MODEL` and accept aliases.
- Reranker optionality is good; if enabled, ensure it runs only on top-N post-RRF (it already does) and capture latency in tracker.
- Embedding backend download risk offline (moderate in new envs): code supports `EMBEDDINGS_BACKEND=stub`. Document/stage a local cache or package embeddings during build.
- Answerability Jaccard may under/over-fire for paraphrases (low → tune): add light stemming/stopwording, and boost overlap on quoted parameter names common in docs.
- BM25 tokenization is simple `.lower().split()` (low → improve): add basic alnum regex tokenization and stopwords; optional stemming.

**Fit for “Clockify Help” Agent**
- Retrieval focuses on help content with breadcrumbs, anchors, sections. Chat builds concise answers with inline citations and a “Not in docs” fallback. With oss20b as the generator and the current pipeline, the agent is production-capable for help-center flows.

**Concrete Recommendations**
- P0 (Quality):
  - Switch MMR similarity to vector cosine using cached chunk embeddings. Keep token-overlap as fallback.
  - Enforce single prompt builder (use `RAGPrompt.build_messages`) in `src/server.py` chat/search to guarantee consistent instructions/citations.
- P1 (Recall/Precision):
  - Expand `data/domain/glossary.json` with API synonyms (e.g., webhook/event, token/API key, SSO/SAML/SSO) and product taxonomy; drive more `expand_structured` variants.
  - BM25 tokenization: replace `.split()` with a regex tokenizer and stopword list; optionally porter stemming.
  - Add entity-aware field boosts from `src/search_improvements.py` into hybrid fusion inputs (title/section boosts already available).
- P1 (Ops):
  - Normalize model naming in config: document `LLM_MODEL=oss20b` as standard, keep `gpt-oss:20b` as default alias; surface active model in `/config` (already shown).
  - Preload FAISS with `make_direct_map` at build time to ensure `reconstruct()` works on all index types; document this in index build scripts.
- P2 (UX):
  - Source section breadcrumbs are present; include anchors in Sources for deep links (server already does for chat, keep parity in search response where useful).
  - Consider minor “answerability coaching” in failures: return short “Not in docs” plus 1–2 related queries from `query_optimizer.suggest_refinements`.

**Config Tips (oss20b)**
- Set `LLM_API_TYPE=ollama`, `LLM_BASE_URL=http://127.0.0.1:11434`, `LLM_MODEL=oss20b` (or keep `gpt-oss:20b` if your endpoint expects that tag).
- For offline/dev: `MOCK_LLM=true`, `EMBEDDINGS_BACKEND=stub` to run tests without downloads.

**Representative Code References**
- RRF fusion: `src/retrieval_engine.py:372`
- MMR diversity: `src/retrieval_engine.py:423`
- Hybrid strategy: `src/retrieval_engine.py:283`
- FAISS vector reconstruction cache: `src/index_manager.py:128`
- E5 embeddings with prefixes: `src/embeddings.py:108`, `src/embeddings.py:26`
- Chat pipeline + answerability: `src/server.py:1133`, `src/llm_client.py:116`
- Citations validator: `src/citation_validator.py:114`
- Query decomposition: `src/query_decomposition.py:342` (LLM fallback: `src/query_decomposition.py:269`)
- Clockify chunking: `src/chunkers/clockify.py:60`

**Verdict**
- Strong foundation, production-ready for a Clockify help agent with oss20b. Implement the P0/P1 items to further reduce redundancy, tighten grounding, and simplify maintenance.

