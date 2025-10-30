# Changelog

## v3.0.0 – Production cutover
- Unified Clockify-only corpus with v2 enriched metadata
- Default chunking: url_level (h2/h3 experimental)
- Hybrid retrieval tuned: SEARCH_LEXICAL_WEIGHT=0.50
- Allowlist guard at runtime; no raw embeddings in JSON
- Health endpoints: /healthz, /readyz
- Monitoring: retrieval_metrics.log → DASHBOARD.json; alerts (ALERTS.md)
- CI: lint, type-check, tests, offline eval, corpus freeze

See codex/CHANGELOG_RAG_v2_to_v3.md for migration details.
