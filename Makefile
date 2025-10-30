PYTHON ?= python3

ingest:
	@echo "Running Clockify Help ingestion (process -> chunk -> embed)..."
	$(PYTHON) -m src.ingest

# v2 ingestion (enriched JSON with allow/deny)
ingest_v2:
	@echo "Building v2 index from enriched corpus (CHUNK_STRATEGY=$${CHUNK_STRATEGY:-url_level})..."
	CHUNK_STRATEGY=$${CHUNK_STRATEGY:-url_level} $(PYTHON) tools/ingest_v2.py

offline_eval:
	@echo "Running offline lexical eval..."
	$(PYTHON) codex/scripts/offline_eval.py

runtime_smoke:
	@echo "Running runtime smoke (BASE_URL=$${BASE_URL:-http://localhost:7000})..."
	BASE_URL=$${BASE_URL:-http://localhost:7000} $(PYTHON) codex/scripts/rag_smoke.py

hybrid_sweep:
	@echo "Running hybrid weighting sweep..."
	$(PYTHON) codex/scripts/hybrid_sweep.py

corpus_freeze:
	@echo "Freezing corpus and index digests..."
	$(PYTHON) codex/scripts/corpus_freeze.py

deploy_staging:
	@echo "Deploying to staging..."
	$(PYTHON) codex/scripts/deploy.py staging

deploy_prod:
	@echo "Deploying to production..."
	$(PYTHON) codex/scripts/deploy.py prod

rollback:
	@echo "Rolling back to previous index/config..."
	$(PYTHON) codex/scripts/rollback.py

serve:
	uvicorn src.server:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-7001}

ui:
	@echo "Starting demo UI on http://localhost:8080..."
	@echo "Press Ctrl+C to stop"
	@cd ui && $(PYTHON) -m http.server 8080

test-llm:
	$(PYTHON) scripts/test_llm_connection.py

test-rag:
	$(PYTHON) scripts/test_rag_pipeline.py

retriever-test:
	@echo "Running offline retrieval evaluation..."
	$(PYTHON) eval/run_eval.py --k 5 --context-k 4 --json

eval:
	@echo "Running RAG evaluation harness..."
	$(PYTHON) eval/run_eval.py

eval-full:
	@echo "Running pytest-based evaluation suite..."
	SKIP_API_EVAL=false API_HOST=localhost API_PORT=7000 $(PYTHON) -m pytest tests/test_clockify_rag_eval.py -v

eval-health:
	@echo "Checking API health..."
	@curl -s http://localhost:7000/health | $(PYTHON) -m json.tool

eval-glossary:
	@echo "Running glossary and hybrid retrieval evaluation..."
	$(PYTHON) scripts/eval_rag.py

eval-axioms:
	@echo "Running comprehensive RAG Standard v1 evaluation (AXIOM 1-9)..."
	$(PYTHON) eval/run_eval.py --base-url http://localhost:7000

coverage-audit:
	@echo "Running coverage audit..."
	$(PYTHON) scripts/coverage_audit.py --namespace clockify --summary

coverage-audit-json:
	@echo "Generating coverage audit JSON..."
	$(PYTHON) scripts/coverage_audit.py --namespace clockify --output coverage_audit_latest.json --summary

.PHONY: ingest serve ui test-llm test-rag retriever-test eval eval-full eval-health eval-glossary eval-axioms coverage-audit coverage-audit-json
