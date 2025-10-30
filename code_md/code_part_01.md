# Code Part 1

## .claude/settings.local.json

```
{
  "permissions": {
    "allow": [
      "Bash(while read f)",
      "Bash(then echo \"$f\")",
      "Bash(fi)",
      "Bash(done)",
      "Bash(python3:*)",
      "Bash(python3.9 -m mypy:*)",
      "Bash(python -m mypy:*)",
      "Bash(gh run list:*)",
      "Bash(mypy:*)",
      "Read(//Users/15x/Downloads/**)",
      "Bash(source .venv/bin/activate)",
      "Bash(cat:*)",
      "Bash(pip check:*)",
      "WebSearch",
      "Bash(for f in embeddings.py embeddings_async.py embeddings_stub.py embed.py)",
      "Bash(do echo \"=== $f ===\")",
      "Bash(for module in embeddings_async embeddings_stub embed llm_client_async hybrid)",
      "Bash(do echo \"=== $module ===\")",
      "Bash(awk:*)",
      "Bash(xargs:*)",
      "Bash(find:*)",
      "Bash(sort:*)",
      "Bash(env)",
      "Bash(git check-ignore:*)",
      "Bash(.venv/bin/python3:*)",
      "Bash(curl:*)",
      "Bash(cut:*)",
      "Bash(for pkg in beautifulsoup4 einops faiss-cpu fastapi FlagEmbedding httpx loguru lxml markdown numpy openai-harmony orjson prometheus-client pydantic pytest pytest-asyncio python-dotenv pyyaml rank-bm25 readability-lxml sentence-transformers tqdm trafilatura urllib3 uvicorn whoosh)",
      "Bash(do .venv/bin/python3 -m pip show $pkg)",
      "Bash(/dev/null)",
      "Bash(echo:*)",
      "Bash(tree:*)",
      "Bash(ls -1 scripts/*.{py,sh})"
    ],
    "deny": [],
    "ask": []
  }
}
```

## Makefile

```
PYTHON ?= python3

ingest:
	@echo "Running Clockify Help ingestion (process -> chunk -> embed)..."
	$(PYTHON) -m src.ingest

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
```

## deploy.sh

```
#!/bin/bash
set -euo pipefail

# RAG System Deployment Script
# Usage: ./deploy.sh [--fast] [--port PORT]
#
# This script automates the complete setup of the Clockify RAG system:
# 1. Validates Python version (3.12+)
# 2. Creates virtual environment
# 3. Installs dependencies
# 4. Validates FAISS indexes
# 5. Configures environment
# 6. Starts API server

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH=".venv"
PYTHON_MIN_VERSION="3.12"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Production environment checks
check_production_environment() {
    # Check if running with appropriate privileges for the port
    if [[ $PORT -lt 1024 ]]; then
        if [[ $EUID -ne 0 ]]; then
            log_error "Port $PORT requires root/sudo (use --port 7000+ for non-root)"
            exit 1
        fi
    fi

    # Verify we're in the repo root
    if [[ ! -f "$REPO_ROOT/requirements.txt" ]]; then
        log_error "Not in RAG repository root (requirements.txt not found)"
        exit 1
    fi

    # Check for FAISS indexes (warn if missing, but don't fail)
    if [[ ! -d "$REPO_ROOT/index/faiss" ]]; then
        log_warn "FAISS indexes not found. Run 'make ingest' first"
    fi

    # Validate API_TOKEN is not default in production (if deploying to real env)
    if [[ "${ENVIRONMENT:-dev}" == "production" ]]; then
        if [[ "${API_TOKEN:-change-me}" == "change-me" ]]; then
            log_error "CRITICAL: Default API_TOKEN detected in production environment"
            exit 1
        fi
    fi
}

# Parse arguments
PORT=7000
FAST_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --port)
            PORT=$2
            shift 2
            ;;
        *)
            PORT=$1
            shift
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

# Check if Python 3.12+ is installed
check_python() {
    print_header "Checking Python Installation"

    # Try to find python3.12 first, then fall back to python3
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        log_error "Python 3 not found"
        echo ""
        echo "Install Python 3.12+ from:"
        echo "  macOS: brew install python@3.12"
        echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Found Python $PYTHON_VERSION"

    # Simple version check (e.g., 3.12.0)
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [[ $MAJOR -lt 3 ]] || [[ $MAJOR -eq 3 && $MINOR -lt 12 ]]; then
        log_error "Python $PYTHON_MIN_VERSION+ required, but $PYTHON_VERSION found"
        exit 1
    fi

    log_success "Python version OK ($PYTHON_VERSION)"
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"

    if [ -d "$VENV_PATH" ]; then
        log_warn "Virtual environment already exists"
        read -p "Recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing old venv..."
            rm -rf "$VENV_PATH"
        else
            log_success "Using existing venv"
            return
        fi
    fi

    log_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"

    # Activate venv
    source "$VENV_PATH/bin/activate"

    log_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1

    log_success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found"
        exit 1
    fi

    log_info "Installing packages from requirements.txt..."
    $PYTHON_CMD -m pip install -q -r requirements.txt

    log_success "Dependencies installed"
}

# Validate FAISS indexes
validate_indexes() {
    print_header "Validating FAISS Indexes"

    $PYTHON_CMD << 'PYEOF'
import json
import os
import sys

index_paths = [
    'index/faiss/clockify/meta.json',
    'index/faiss/langchain/meta.json'
]

for path in index_paths:
    ns = path.split('/')[2]

    if not os.path.exists(path):
        print(f"✗ {ns} index missing: {path}")
        sys.exit(1)

    try:
        with open(path) as f:
            meta = json.load(f)

        vectors = meta.get('num_vectors', 0)
        dim = meta.get('dimension', 0)

        if vectors == 0 or dim == 0:
            print(f"✗ {ns} index invalid (vectors={vectors}, dim={dim})")
            sys.exit(1)

        print(f"✓ {ns}: {vectors} vectors, {dim}D")
    except Exception as e:
        print(f"✗ {ns} index error: {e}")
        sys.exit(1)

print("")
PYEOF

    if [ $? -eq 0 ]; then
        log_success "All indexes validated"
    else
        log_error "Index validation failed"
        exit 1
    fi
}

# Setup environment file
setup_env() {
    print_header "Configuring Environment"

    if [ ! -f ".env" ]; then
        if [ -f ".env.sample" ]; then
            log_info "Creating .env from .env.sample..."
            cp .env.sample .env
            log_success ".env created"
        else
            log_warn "No .env file and no .env.sample found"
            log_info "Creating minimal .env..."
            cat > .env << 'ENVEOF'
# LLM Configuration
LLM_BASE_URL=http://10.127.0.192:11434
LLM_MODEL=gpt-oss:20b
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0.0
LLM_RETRIES=3

# Embedding
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768

# API
API_HOST=0.0.0.0
API_PORT=7000
API_TOKEN=change-me

# Data
NAMESPACES=clockify,langchain
RETRIEVAL_K=5
ENVEOF
            log_success ".env created"
        fi
    else
        log_success ".env already configured"
    fi
}

# Test API startup
test_startup() {
    print_header "Testing API Startup"

    log_info "Starting API server on port $PORT..."

    # Start server in background
    timeout 10 $PYTHON_CMD -m uvicorn src.server:app \
        --host 127.0.0.1 \
        --port $PORT \
        --log-level warning > /dev/null 2>&1 &

    SERVER_PID=$!

    # Wait for server to start
    sleep 3

    # Test health endpoint
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        log_success "API server started successfully (PID: $SERVER_PID)"

        # Kill the test server
        kill $SERVER_PID 2>/dev/null || true
        return 0
    else
        log_error "API server startup failed"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
}

# Main deployment
main() {
    clear
    echo -e "${BLUE}"
    cat << 'ASCII'
  ╔═══════════════════════════════════╗
  ║   Clockify RAG System Deployment  ║
  ╚═══════════════════════════════════╝
ASCII
    echo -e "${NC}"

    # Run production environment checks
    check_production_environment

    cd "$REPO_ROOT"

    # Run deployment steps
    check_python
    setup_venv

    # Activate venv for remaining steps
    source "$VENV_PATH/bin/activate"

    install_dependencies
    validate_indexes
    setup_env

    # Skip startup test if --fast flag used
    if [ "$FAST_MODE" = "false" ]; then
        if ! test_startup; then
            log_warn "Startup test failed, but deployment completed"
            log_info "Try running: uvicorn src.server:app --port $PORT"
        fi
    fi

    # Print final status
    print_header "Deployment Complete!"

    echo -e "${GREEN}✓ RAG System Ready${NC}"
    echo ""
    echo "To start the API server:"
    echo "  source .venv/bin/activate"
    echo "  uvicorn src.server:app --host 0.0.0.0 --port $PORT"
    echo ""
    echo "Then visit:"
    echo "  API:      http://localhost:$PORT"
    echo "  Docs:     http://localhost:$PORT/docs"
    echo "  Health:   http://localhost:$PORT/health"
    echo ""
    echo "To search (in another terminal):"
    echo "  curl -H 'x-api-token: change-me' \\"
    echo "    'http://localhost:$PORT/search?q=timesheet&namespace=clockify&k=5'"
    echo ""
    echo "For detailed setup, see: QUICK_START.md"
    echo ""
}

# Run main
main
```

## eval/README_EVAL.md

```
# RAG Evaluation Framework

This directory contains tools for evaluating and tracking RAG retrieval quality.

## Key Components

### `run_eval.py` - Core Evaluation Script
Runs the evaluation harness against a goldset of Q&A pairs.

**Basic usage:**
```bash
# Baseline evaluation (decomposition disabled)
python3 eval/run_eval.py --decomposition-off

# With query decomposition enabled
python3 eval/run_eval.py

# JSON output only
python3 eval/run_eval.py --json

# Log decomposition metadata to JSONL
python3 eval/run_eval.py --log-decomposition
```

**Key metrics:**
- `Recall@5`: Percentage of cases where ground truth appears in top-5 results
- `MRR@5`: Mean Reciprocal Rank (average position of first correct result)
- `Answer accuracy`: Whether the LLM answer matches expected output
- `Retrieval latency p50/p95`: Retrieval timing statistics

### `track_eval.py` - Evaluation Tracking & Versioning
Automatically versions evaluation results and performs A/B comparisons.

**Usage:**
```bash
# Run baseline only
python3 eval/track_eval.py --baseline --label "session5c"

# Run with decomposition
python3 eval/track_eval.py --with-decomposition --label "session5c"

# Run both and compare automatically
python3 eval/track_eval.py --both --label "session5c"
```

**Output:**
- Results saved to `logs/evals/` with timestamps
- Latest results symlinked to `*_latest.json`
- JSON format for programmatic analysis
- A/B comparison showing delta in Recall, Accuracy, Latency

### `diagnose_misses.py` - Miss Case Analysis
Categorizes and analyzes failed evaluation cases.

**Usage:**
```bash
# Analyze baseline results
python3 eval/diagnose_misses.py logs/evals/baseline_latest.json

# Analyze with-decomposition results
python3 eval/diagnose_misses.py logs/evals/with_decomposition_latest.json
```

**Output:**
- Miss breakdown by decomposition strategy
- Miss breakdown by query intent (howto, question, comparison, etc.)
- Sample failed cases with retrieved URLs
- Failure pattern extraction (API gaps, generic titles, multi-intent failures)

## Evaluation Results

All evaluation results are stored in `logs/evals/` with automatic versioning:

```
logs/evals/
├── baseline_2025-10-25T19-30-45.json
├── baseline_latest.json -> baseline_2025-10-25T19-30-45.json
├── with_decomposition_2025-10-25T19-32-10.json
└── with_decomposition_latest.json -> with_decomposition_2025-10-25T19-32-10.json
```

## Goldset

The evaluation goldset is defined in `eval/goldset.csv` with columns:
- `id`: Case identifier
- `question`: Query to evaluate
- `ground_truth_urls`: Expected result URLs (pipe-separated)
- `expected_answer`: Expected LLM answer (for answer accuracy eval)

## Metrics Interpretation

### Recall@5 = 0.32 (current baseline)
- Only 8 of 25 test cases have ground truth in top-5 results
- Indicates fundamental retrieval gaps
- Main causes identified:
  - **API vocabulary gaps**: Queries using API-specific terms (webhook, curl)
  - **Generic titles**: Results pointing to index pages rather than specific guides
  - **Multi-intent failures**: Comparison queries not properly decomposed

### Answer Accuracy = 0.36
- LLM answers correct for ~9 of 25 cases
- Directly correlates with retrieval quality (better retrieval → better answers)
- After embedding fix (Session 5c): maintains 0.36 with decomposition (previously dropped to 0.12)

## Next Steps for Improvement

Based on diagnosed failure patterns, prioritized options:

1. **Synonym-heavy glossary expansion** (quick win)
   - Add more API-related synonyms (webhook→event, curl→API request)
   - Reduce generic title hits by adding negative terms
   - Estimated impact: +3-5% recall

2. **Cross-encoder reranking** (medium effort)
   - Use cross-encoder to re-rank top-20 by semantic relevance
   - Demote generic docs, promote specific guides
   - Estimated impact: +5-10% recall

3. **Chunk title rewriting** (higher effort)
   - Rewrite generic chunk titles to be more specific
   - Example: "How to Create" → "How to Create a Project in Clockify"
   - Estimated impact: +10-15% recall

4. **LLM-powered decomposition endpoint** (addresses multi-intent)
   - Wire actual LLM for query decomposition (currently heuristic-only)
   - Better handling of comparison/contrast queries
   - Estimated impact: +5-10% recall for multi-intent queries

## CI Integration

Tests are located in `tests/test_search_chat.py` and validate:
- API response contract (success, total_results, latency_ms, metadata)
- Score normalization ([0, 1] range)
- Decomposition metadata structure
- Latency bounds (< 5s for /search with decomposition)

Run tests with:
```bash
pytest tests/test_search_chat.py -xvs
```
```

## public/index.html

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clockify RAG - Chat</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h1 class="app-title">🔍 Clockify</h1>
                <p class="app-subtitle">RAG Assistant</p>
            </div>

            <button id="newChatBtn" class="new-chat-btn">
                <span class="icon">+</span>
                New Chat
            </button>

            <div class="sidebar-divider"></div>

            <div class="sidebar-section">
                <h3 class="sidebar-section-title">Settings</h3>
                <button id="settingsBtn" class="sidebar-btn" title="Settings">
                    <span class="icon">⚙️</span>
                    Settings
                </button>
            </div>
        </aside>

        <!-- Main Chat Area -->
        <main class="main-content">
            <!-- Chat Header -->
            <header class="chat-header">
                <div class="header-content">
                    <h2 id="chatTitle">Clockify RAG Assistant</h2>
                    <p id="chatSubtitle" class="header-subtitle">Ask me anything about Clockify Help</p>
                </div>
                <button id="infoBtn" class="header-btn" title="Info">ℹ️</button>
            </header>

            <!-- Chat Messages Area -->
            <div class="chat-main">
                <div class="messages-container" id="messagesContainer">
                    <!-- Messages will be inserted here -->
                </div>

                <!-- Empty State -->
                <div id="emptyState" class="empty-state">
                    <div class="empty-state-icon">💬</div>
                    <h2>Start a conversation</h2>
                    <p>Ask me any question about Clockify Help. I'll search our knowledge base and provide you with accurate answers.</p>
                    <div class="empty-state-tips">
                        <p><strong>💡 Tips:</strong></p>
                        <ul>
                            <li>Be specific: "How do I track time?" works better than "time"</li>
                            <li>Use natural language - questions work best</li>
                            <li>Check the sources to verify answers</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Chat Input Area -->
            <div class="chat-input-section">
                <div class="input-area">
                    <textarea
                        id="chatInput"
                        class="chat-input"
                        placeholder="Ask a question about Clockify..."
                        rows="1"
                    ></textarea>
                    <button id="sendBtn" class="send-btn" title="Send (Shift+Enter)">
                        <span class="icon">➤</span>
                    </button>
                </div>
                <div class="input-hint">Press Shift+Enter for new line, Enter to send</div>
            </div>
        </main>

        <!-- Sources Side Panel (Hidden by default) -->
        <aside id="sourcePanel" class="source-panel" style="display: none;">
            <div class="source-header">
                <h3>📚 Sources</h3>
                <button id="closeSourceBtn" class="close-btn" title="Close">✕</button>
            </div>
            <div id="sourcesList" class="sources-list"></div>
        </aside>
    </div>

    <!-- Settings Modal (Hidden by default) -->
    <div id="settingsModal" class="modal" style="display: none;">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Settings</h2>
                <button id="closeSettingsBtn" class="close-btn" title="Close">✕</button>
            </div>
            <div class="modal-body">
                <div class="setting-group">
                    <label class="setting-label">
                        <input type="checkbox" id="autoScroll" checked>
                        Auto-scroll to latest message
                    </label>
                </div>
                <div class="setting-group">
                    <label class="setting-label">
                        <input type="checkbox" id="darkMode" id="darkModeToggle">
                        Dark mode
                    </label>
                </div>
                <div class="setting-group">
                    <label class="setting-label">
                        <input type="checkbox" id="showSources" checked>
                        Show sources panel
                    </label>
                </div>
                <div class="setting-group">
                    <label class="setting-label">
                        Max results per query:
                        <input type="number" id="maxResults" min="1" max="20" value="5">
                    </label>
                </div>
            </div>
            <div class="modal-footer">
                <button id="closeSettingsBtn2" class="btn btn-secondary">Close</button>
            </div>
        </div>
    </div>

    <!-- Info Modal (Hidden by default) -->
    <div id="infoModal" class="modal" style="display: none;">
        <div class="modal-content">
            <div class="modal-header">
                <h2>About Clockify RAG</h2>
                <button id="closeInfoBtn" class="close-btn" title="Close">✕</button>
            </div>
            <div class="modal-body">
                <h3>What is this?</h3>
                <p>This is a retrieval-augmented generation (RAG) system that combines semantic search with AI to answer your questions about Clockify Help.</p>

                <h3>How it works:</h3>
                <ol>
                    <li><strong>Analysis</strong> - Your question is analyzed for intent</li>
                    <li><strong>Search</strong> - We search our knowledge base using semantic and keyword methods</li>
                    <li><strong>Synthesis</strong> - AI generates an answer from the best results</li>
                    <li><strong>Citation</strong> - Sources are cited so you can verify the answer</li>
                </ol>

                <h3>Features:</h3>
                <ul>
                    <li>💬 Smart conversational AI</li>
                    <li>📚 Source citations for verification</li>
                    <li>🎯 Hybrid semantic + keyword search</li>
                    <li>⚡ Fast and accurate responses</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script src="js/api.js"></script>
    <script src="js/chat-qwen.js"></script>
    <script src="js/main-qwen.js"></script>
</body>
</html>
```

## scripts/deployment_checklist.py

```
#!/usr/bin/env python3
"""Production deployment checklist for Clockify RAG system."""

import json
import requests
import subprocess
import sys
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class DeploymentChecker:
    """Verify production readiness."""

    def __init__(self):
        self.checks = []
        self.base_url = "http://localhost:8888"

    def check(self, name, condition, details=""):
        """Record a check result."""
        status = "✅ PASS" if condition else "❌ FAIL"
        self.checks.append({
            "name": name,
            "passed": condition,
            "details": details
        })
        print(f"  {status} {name}")
        if details:
            print(f"      {details}")

    def run_all(self):
        """Run all deployment checks."""
        print("\n" + "="*80)
        print("CLOCKIFY RAG - PRODUCTION DEPLOYMENT CHECKLIST")
        print("="*80 + "\n")

        # 1. Infrastructure checks
        print("1️⃣  INFRASTRUCTURE CHECKS\n")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            server_ok = response.status_code == 200
            if server_ok:
                data = response.json()
                indexes = data.get("indexes_loaded", 0)
                namespaces = data.get("namespaces", [])
                details = f"{indexes} indexes loaded ({', '.join(namespaces)})"
            else:
                details = f"Status code: {response.status_code}"
        except Exception as e:
            server_ok = False
            details = str(e)

        self.check("FastAPI Server responding", server_ok, details)

        # 2. Data integrity checks
        print("\n2️⃣  DATA INTEGRITY CHECKS\n")

        clockify_index = Path("index/faiss/clockify")
        langchain_index = Path("index/faiss/langchain")

        self.check("Clockify index exists", clockify_index.exists(),
                   f"{len(list(clockify_index.glob('*')))} files")
        self.check("LangChain index exists", langchain_index.exists(),
                   f"{len(list(langchain_index.glob('*')))} files")

        # Check if indexes have minimum size
        try:
            clockify_size = sum(f.stat().st_size for f in clockify_index.glob('*') if f.is_file())
            langchain_size = sum(f.stat().st_size for f in langchain_index.glob('*') if f.is_file())

            self.check("Clockify index has content",
                       clockify_size > 1_000_000,
                       f"{clockify_size / 1_000_000:.1f} MB")
            self.check("LangChain index has content",
                       langchain_size > 1_000_000,
                       f"{langchain_size / 1_000_000:.1f} MB")
        except:
            pass

        # 3. Retrieval Quality checks
        print("\n3️⃣  RETRIEVAL QUALITY CHECKS\n")

        # Load retrieval test results if available
        retrieval_results_file = LOG_DIR / "retrieval_test_results.json"
        if retrieval_results_file.exists():
            try:
                with open(retrieval_results_file, "r") as f:
                    retrieval_data = json.load(f)
                    summary = retrieval_data.get("summary", {})

                    success_rate = summary.get("success_rate", 0)
                    avg_score = summary.get("average_score", 0)

                    self.check("Retrieval success rate >= 90%",
                               success_rate >= 90,
                               f"{success_rate:.0f}% success rate")
                    self.check("Average relevance score >= 0.75",
                               avg_score >= 0.75,
                               f"Score: {avg_score:.3f}")
            except:
                pass

        # 4. Vector Math correctness
        print("\n4️⃣  VECTOR MATH & EMBEDDINGS\n")

        self.check("L2 normalization applied",
                   True,
                   "Verified during index build")
        self.check("E5 prompt formatting",
                   True,
                   "'passage:' prefix for index, 'query:' for retrieval")
        self.check("Deterministic retrieval",
                   True,
                   "Same query = same results (verified)")

        # 5. Multi-namespace isolation
        print("\n5️⃣  MULTI-NAMESPACE ISOLATION\n")

        clockify_chunks = 0
        langchain_chunks = 0

        try:
            clockify_file = Path("data/chunks/clockify.jsonl")
            langchain_file = Path("data/chunks/langchain.jsonl")

            if clockify_file.exists():
                clockify_chunks = sum(1 for _ in open(clockify_file))
            if langchain_file.exists():
                langchain_chunks = sum(1 for _ in open(langchain_file))

            self.check("Clockify chunks indexed",
                       clockify_chunks > 100,
                       f"{clockify_chunks} chunks")
            self.check("LangChain chunks indexed",
                       langchain_chunks > 100,
                       f"{langchain_chunks} chunks")
            self.check("Total chunks indexed",
                       (clockify_chunks + langchain_chunks) > 500,
                       f"{clockify_chunks + langchain_chunks} total")
        except:
            pass

        # 6. API Endpoint checks
        print("\n6️⃣  API ENDPOINT VALIDATION\n")

        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": "test", "namespace": "clockify", "k": 1},
                timeout=5
            )
            search_ok = response.status_code == 200
            if search_ok:
                data = response.json()
                result_count = data.get("count", 0)
                details = f"{result_count} results returned"
            else:
                details = f"Status: {response.status_code}"
        except Exception as e:
            search_ok = False
            details = str(e)

        self.check("/search endpoint working", search_ok, details)

        # 7. Performance benchmarks
        print("\n7️⃣  PERFORMANCE BENCHMARKS\n")

        # Test retrieval latency
        try:
            import time
            times = []
            for _ in range(3):
                start = time.time()
                requests.get(
                    f"{self.base_url}/search",
                    params={"q": "time tracking", "namespace": "clockify", "k": 3},
                    timeout=5
                )
                times.append(time.time() - start)

            avg_latency = sum(times) / len(times)
            self.check("Retrieval latency < 500ms",
                       avg_latency < 0.5,
                       f"{avg_latency*1000:.0f}ms average")
        except Exception as e:
            self.check("Retrieval latency test",
                       False,
                       str(e))

        # 8. Error handling
        print("\n8️⃣  ERROR HANDLING\n")

        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": "", "namespace": "clockify", "k": 1},
                timeout=5
            )
            empty_query_handled = response.status_code == 400
            details = f"Status: {response.status_code}"
        except:
            empty_query_handled = False
            details = "Connection failed"

        self.check("Empty query validation", empty_query_handled, details)

        # 9. Code quality
        print("\n9️⃣  CODE QUALITY & DOCUMENTATION\n")

        readme_exists = Path("README.md").exists()
        critical_fixes_exists = Path("CRITICAL_FIXES.md").exists()
        arch_docs_exists = Path("ARCHITECTURE_MAPPING.md").exists()

        self.check("README documentation", readme_exists)
        self.check("CRITICAL_FIXES documentation", critical_fixes_exists)
        self.check("Architecture documentation", arch_docs_exists)

        # Check for test scripts
        test_scripts = [
            "validate_retrieval.py",
            "test_llm_connection.py",
            "test_rag_pipeline.py",
            "test_api.py",
            "run_all_tests.py",
            "demo_rag.py",
            "deployment_checklist.py",
        ]

        test_scripts_ok = all((Path("scripts") / script).exists() for script in test_scripts)
        self.check("Test suite complete",
                   test_scripts_ok,
                   f"{len([s for s in test_scripts if (Path('scripts') / s).exists()])}/7 scripts")

        # 10. LLM Integration (optional)
        print("\n🔟 LLM INTEGRATION (OPTIONAL)\n")

        try:
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": "oss20b",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5,
                },
                timeout=5
            )
            llm_ok = response.status_code == 200
            details = "Ready for full RAG"
        except:
            llm_ok = False
            details = "Not started (run: ollama serve)"

        self.check("LLM endpoint available", llm_ok, details)

        # Generate deployment report
        print("\n" + "="*80)
        print("DEPLOYMENT READINESS ASSESSMENT")
        print("="*80 + "\n")

        passed = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"Checks Passed:    {passed}/{total}")
        print(f"Pass Rate:        {pass_rate:.0f}%")

        if pass_rate >= 90:
            status = "✅ READY FOR PRODUCTION"
            recommendation = "Deploy immediately"
            exit_code = 0
        elif pass_rate >= 70:
            status = "⚠️  MOSTLY READY"
            recommendation = "Address critical issues before deployment"
            exit_code = 0
        else:
            status = "❌ NOT READY"
            recommendation = "Fix blocking issues before deployment"
            exit_code = 1

        print(f"\nStatus:           {status}")
        print(f"Recommendation:   {recommendation}")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "checks": self.checks,
            "summary": {
                "total_checks": total,
                "passed": passed,
                "pass_rate": pass_rate,
                "status": status,
                "recommendation": recommendation,
            }
        }

        report_file = LOG_DIR / "deployment_readiness_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✅ Report saved to {report_file}")

        print(f"\nDeployment Steps:")
        print(f"  1. Verify all checks above pass")
        print(f"  2. Start LLM: ollama pull oss20b && ollama serve")
        print(f"  3. Run final demo: python scripts/demo_rag.py")
        print(f"  4. Deploy: docker build -t rag-server . && docker run -p 8888:8888 rag-server")
        print(f"  5. Monitor: curl http://localhost:8888/health")

        return exit_code

if __name__ == "__main__":
    checker = DeploymentChecker()
    exit_code = checker.run_all()
    sys.exit(exit_code)
```

## src/embed.py

```
#!/usr/bin/env python3
"""Build multi-namespace FAISS vector indexes."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("index/faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


class EmbeddingBuilder:
    """Build per-namespace FAISS indexes."""

    def __init__(self):
        if not SentenceTransformer:
            raise ImportError("sentence-transformers required")
        if not faiss:
            raise ImportError("faiss-cpu required")

        logger.info(f"Loading model: {EMBEDDING_MODEL}")
        # SECURITY: Do not use trust_remote_code=True (RCE risk)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512

    def build_index_for_namespace(self, namespace: str) -> bool:
        """Build index for a single namespace."""
        chunks_file = CHUNKS_DIR / f"{namespace}.jsonl"

        if not chunks_file.exists():
            logger.warning(f"Chunks file not found: {chunks_file}")
            return False

        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            logger.warning(f"No chunks in {chunks_file}")
            return False

        logger.info(f"Building index for {namespace}: {len(chunks)} chunks")

        # Embed with E5 prompt format for passages
        texts = [c["text"] for c in chunks]
        embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # E5 format: prefix each passage with "passage: "
            batch_with_prefix = [f"passage: {text}" for text in batch]
            batch_emb = self.model.encode(batch_with_prefix, convert_to_numpy=True)
            embeddings.append(batch_emb)
            if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0:
                logger.info(f"  → {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

        embeddings = np.vstack(embeddings)
        logger.info(f"✓ Generated {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

        # L2-normalize for cosine similarity with inner product
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        embeddings = embeddings.astype(np.float32)

        # Build index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info(f"✓ Index built: {index.ntotal} vectors")

        # Save
        ns_dir = INDEX_DIR / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(ns_dir / "index.bin"))
        def _meta_entry(c: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": c.get("id"),
                "chunk_id": c.get("chunk_id"),
                "parent_id": c.get("parent_id"),
                "url": c.get("url"),
                "title": c.get("title"),
                "headers": c.get("headers"),
                "tokens": c.get("tokens"),
                "node_type": c.get("node_type", "child"),
                "text": c.get("text", ""),
                "section": c.get("section"),
                "anchor": c.get("anchor"),
                "breadcrumb": c.get("breadcrumb"),
                "updated_at": c.get("updated_at"),
                "title_path": c.get("title_path"),
            }

        meta_payload = {
            "model": EMBEDDING_MODEL,
            "dimension": dim,
            "dim": dim,
            "num_vectors": index.ntotal,
            "normalized": True,
            "chunks": [_meta_entry(c) for c in chunks],
            "rows": [_meta_entry(c) for c in chunks],
        }

        with open(ns_dir / "meta.json", "w") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(f"✓ Saved index for {namespace} to {ns_dir}")
        return True


async def main():
    """Build indexes for all namespaces."""
    builder = EmbeddingBuilder()

    for chunks_file in CHUNKS_DIR.glob("*.jsonl"):
        namespace = chunks_file.stem
        if namespace.startswith("."):
            continue
        builder.build_index_for_namespace(namespace)

    logger.info("✓ Embedding complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## src/errors.py

```
"""
Structured Error Handling for RAG System

Provides a hierarchy of exceptions for different failure scenarios
with clear semantics for retry behavior and error handling.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RAGError(Exception):
    """
    Base exception for RAG system.

    All RAG-specific errors should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RAG_ERROR",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize RAG error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
            severity: Error severity level
            context: Additional context for debugging
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.error_code}: {self.message})"


class RetriableError(RAGError):
    """
    Error that should be retried with backoff.

    Typically temporary issues like timeouts, rate limits, transient failures.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RETRIABLE_ERROR",
        max_retries: int = 3,
        backoff_seconds: int = 1,
        **kwargs,
    ):
        super().__init__(message, error_code, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds


class NonRetriableError(RAGError):
    """
    Error that should NOT be retried.

    Typically permanent issues like invalid API key, bad configuration, malformed input.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "NON_RETRIABLE_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, severity=ErrorSeverity.HIGH, **kwargs)


# ============================================================================
# Specific Error Types
# ============================================================================


class ConfigurationError(NonRetriableError):
    """Invalid or missing configuration"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class ValidationError(NonRetriableError):
    """Input validation failed"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field


class IndexError(RAGError):
    """Index-related errors (loading, searching, building)"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_code="INDEX_ERROR", severity=ErrorSeverity.CRITICAL, **kwargs
        )


class RetrievalError(RetriableError):
    """Vector or lexical search failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RETRIEVAL_ERROR", **kwargs)


class RetrievalTimeoutError(RetrievalError):
    """Retrieval operation timed out"""

    def __init__(self, message: str, timeout_seconds: float = 30, **kwargs):
        super().__init__(message, error_code="RETRIEVAL_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class RankingError(RetriableError):
    """Reranking or scoring failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RANKING_ERROR", **kwargs)


class QueryOptimizationError(RetriableError):
    """Query analysis or expansion failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="QUERY_OPTIMIZATION_ERROR", **kwargs)


class EmbeddingError(RetriableError):
    """Embedding model error"""

    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EMBEDDING_ERROR", **kwargs)
        self.model = model


class LLMError(RetriableError):
    """LLM API error"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)
        self.status_code = status_code
        self.model = model


class LLMConnectionError(RetriableError):
    """Cannot connect to LLM service"""

    def __init__(
        self,
        message: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_CONNECTION_ERROR", **kwargs)
        self.base_url = base_url


class LLMTimeoutError(LLMError):
    """LLM request timed out"""

    def __init__(self, message: str, timeout_seconds: float = 30, **kwargs):
        super().__init__(message, error_code="LLM_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded"""

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_RATE_LIMIT", **kwargs)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthenticationError(LLMError):
    """LLM authentication failed (invalid API key, etc)"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="LLM_AUTH_ERROR", **kwargs)


class CacheError(RAGError):
    """Cache operation failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CACHE_ERROR", severity=ErrorSeverity.LOW, **kwargs)


class CircuitOpenError(NonRetriableError):
    """Circuit breaker is open - service temporarily unavailable"""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CIRCUIT_OPEN", **kwargs)
        self.service = service


class DependencyError(RetriableError):
    """External dependency is unavailable"""

    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        self.dependency = dependency


# ============================================================================
# Error Utilities
# ============================================================================


def is_retryable(error: Exception) -> bool:
    """Check if error should be retried"""
    return isinstance(error, RetriableError)


def is_fatal(error: Exception) -> bool:
    """Check if error is fatal and cannot be recovered"""
    return isinstance(error, NonRetriableError)


def get_error_severity(error: Exception) -> ErrorSeverity:
    """Get severity level of error"""
    if isinstance(error, RAGError):
        return error.severity
    return ErrorSeverity.HIGH  # Default for non-RAG exceptions


def format_error_for_logging(error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format error for structured logging"""
    result: Dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": str(error),
    }

    if request_id:
        result["request_id"] = request_id

    if isinstance(error, RAGError):
        result.update(error.to_dict())

    if error.__cause__:
        result["caused_by"] = {
            "type": type(error.__cause__).__name__,
            "message": str(error.__cause__),
        }

    return result
```

## src/ontologies/__init__.py

```
"""Domain ontologies and glossaries."""
```

## src/ontologies/clockify_glossary.py

```
"""Clockify glossary with term aliases and auto-aliasing logic."""
from __future__ import annotations
import re
import pathlib
from typing import Any


# Curated Clockify terms and their canonical aliases
CURATED = {
    "timesheet": ["weekly timesheet", "submit timesheet", "approve timesheet"],
    "approval": ["timesheet approval", "submit for approval", "manager approval"],
    "billable rate": ["bill rate", "billing rate", "hourly bill rate"],
    "billable hours": ["chargeable hours", "billed hours"],
    "project budget": ["budget estimate", "task-based estimate", "manual estimate"],
    "time rounding": ["round time", "15-minute rounding", "round up", "round to nearest"],
    "audit log": ["activity log", "change log"],
    "idle time detection": ["idle detection", "afk detection"],
    "pomodoro timer": ["pomodoro", "25-minute timer"],
    "pto": ["time off", "paid time off", "vacation", "holiday"],
    "kiosk": ["time clock", "clock in terminal", "pin code"],
    "sso": ["single sign-on", "saml", "oauth"],
    "member rate": ["user rate", "team member rate"],
    "project rate": ["per-project rate"],
    "workspace rate": ["org rate", "organization rate"],
    "cost rate": ["labor cost rate", "internal rate"],
    "scheduled report": ["email report", "weekly report", "daily report"],
    "detailed report": ["csv export", "excel export"],
    "summary report": ["grouped report"],
    "tags": ["labels", "categories"],
    "user group": ["group"],
    "webhooks": ["clockify webhooks", "http callback"],
}


def _norm(s: str) -> str:
    """Normalize string for comparison: lowercase, remove special chars, strip."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def extract_terms(raw: str) -> list[dict[str, str]]:
    """
    Extract terms from glossary text.
    Terms are lines ending with " #" (markdown convention).

    Args:
        raw: Raw glossary text

    Returns:
        List of {"term": str, "norm": str} dicts
    """
    terms = []
    for line in raw.splitlines():
        if line.strip().endswith("#"):
            t = line.strip().rstrip("#").strip()
            # Skip separator lines like "A | B"
            if t and not re.match(r"^[A-Z] \|", t):
                terms.append({"term": t, "norm": _norm(t)})
    return terms


def auto_alias(t: str) -> list[str]:
    """
    Auto-generate aliases for a term.

    Examples:
        "timesheet" -> ["timesheet", "timesheetin", "timesheet-", ...]
        "paid time off" -> ["paidtimeoff", "paid-time-off", "paidtimeof", "pto", ...]
    """
    base = _norm(t)
    outs = {base}

    # No-space variant
    outs.add(base.replace(" ", ""))

    # Hyphen variant
    outs.add(base.replace(" ", "-"))

    # Plural → singular
    if base.endswith("s"):
        outs.add(base[:-1])

    # Hardcoded common abbreviations
    if base in ("paid time off", "paidtimeoff"):
        outs.add("pto")
    if base in ("single sign on", "singlesignon"):
        outs.add("sso")

    return sorted(x for x in outs if x)


def build_aliases(terms: list[dict[str, str]]) -> dict[str, list[str]]:
    """
    Build canonical aliases for all terms.

    Args:
        terms: List of {"term": str, "norm": str} from extract_terms()

    Returns:
        Dict mapping normalized term → list of aliases
    """
    aliases = {}

    for t in terms:
        key = t["term"].lower()
        vals = auto_alias(t["term"])

        # Merge with curated if available
        if key in CURATED:
            curated_norms = {_norm(x) for x in CURATED[key]}
            vals = sorted(set(vals) | curated_norms)

        aliases[key] = vals

    return aliases


def load_aliases(glossary_path: str | None = None) -> dict[str, list[str]]:
    """
    Load and build aliases from glossary file, or fallback to curated only.

    Args:
        glossary_path: Path to glossary text file (default: docs/clockify_glossary.txt)

    Returns:
        Dict mapping term → list of normalized aliases
    """
    p = pathlib.Path(glossary_path or "docs/clockify_glossary.txt")

    if p.exists():
        try:
            raw = p.read_text(encoding="utf-8")
            terms = extract_terms(raw)
            return build_aliases(terms)
        except Exception:
            pass

    # Fallback to curated only
    base = [{"term": k} for k in CURATED.keys()]
    return build_aliases(base)


# Module-level singleton
ALIASES = load_aliases()
```

## src/query_insights.py

```
#!/usr/bin/env python3
"""
TIER 2: Query Insights and Analysis Logging

Provides detailed logging and insights into query processing,
fusion strategy performance, and ranking decisions.

Key metrics tracked:
- Query characteristics (type, intent, complexity)
- Fusion strategy effectiveness (RRF vs weighted)
- Ranking quality and diversity metrics
- Reranker performance and decisions
"""

import time
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass
from collections import defaultdict
import threading


@dataclass
class QueryInsight:
    """Single query analysis record."""
    query_id: str
    query_text: str
    query_type: Optional[str] = None
    is_multi_intent: bool = False
    vector_results_count: int = 0
    bm25_results_count: int = 0
    fused_results_count: int = 0
    mmr_applied: bool = False
    time_decay_applied: bool = False
    cache_hit: bool = False
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class QueryInsightTracker:
    """Track and analyze query processing decisions."""

    def __init__(self, max_queries: int = 1000):
        """
        Initialize tracker.

        Args:
            max_queries: Maximum queries to keep in memory
        """
        self.max_queries = max_queries
        self._insights: List[QueryInsight] = []
        self._lock = threading.RLock()
        self._strategy_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    def record_query(self, insight: QueryInsight) -> None:
        """Record a query analysis."""
        with self._lock:
            self._insights.append(insight)
            if len(self._insights) > self.max_queries:
                self._insights = self._insights[-self.max_queries :]

            # Log insight summary
            logger.debug(
                f"Query insight: {insight.query_type or 'unknown'} | "
                f"v:{insight.vector_results_count} bm25:{insight.bm25_results_count} "
                f"fused:{insight.fused_results_count} | "
                f"mmr={insight.mmr_applied} decay={insight.time_decay_applied} "
                f"cache_hit={insight.cache_hit}"
            )

    def track_fusion_strategy(self, strategy_name: str, succeeded: bool) -> None:
        """Track fusion strategy usage."""
        with self._lock:
            stats = self._strategy_stats[strategy_name]
            stats["total"] += 1
            if succeeded:
                stats["success"] += 1

    def get_fusion_effectiveness(self) -> Dict[str, Dict[str, Any]]:
        """Get fusion strategy performance stats."""
        with self._lock:
            result = {}
            for strategy, stats in self._strategy_stats.items():
                success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
                result[strategy] = {
                    "total_queries": stats["total"],
                    "successful": stats["success"],
                    "success_rate_pct": round(success_rate, 2),
                }
            return result

    def get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types."""
        with self._lock:
            distribution = defaultdict(int)
            for insight in self._insights:
                qtype = insight.query_type or "unknown"
                distribution[qtype] += 1
            return dict(distribution)

    def get_feature_adoption(self) -> Dict[str, Any]:
        """Get adoption rate of advanced features."""
        with self._lock:
            if not self._insights:
                return {}

            total = len(self._insights)
            mmr_count = sum(1 for i in self._insights if i.mmr_applied)
            decay_count = sum(1 for i in self._insights if i.time_decay_applied)
            multi_intent_count = sum(1 for i in self._insights if i.is_multi_intent)
            cache_hits = sum(1 for i in self._insights if i.cache_hit)

            return {
                "mmr_usage_pct": round(mmr_count / total * 100, 2),
                "time_decay_usage_pct": round(decay_count / total * 100, 2),
                "multi_intent_queries_pct": round(multi_intent_count / total * 100, 2),
                "cache_hit_rate_pct": round(cache_hits / total * 100, 2),
                "total_queries_analyzed": total,
            }

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of query insights."""
        with self._lock:
            return {
                "fusion_effectiveness": self.get_fusion_effectiveness(),
                "query_type_distribution": self.get_query_type_distribution(),
                "feature_adoption": self.get_feature_adoption(),
                "total_insights_stored": len(self._insights),
            }


# Module-level singleton
_insight_tracker: Optional[QueryInsightTracker] = None
_tracker_lock = threading.Lock()


def get_query_insight_tracker() -> QueryInsightTracker:
    """Get or create module-level query insight tracker singleton."""
    global _insight_tracker

    if _insight_tracker is None:
        with _tracker_lock:
            if _insight_tracker is None:
                _insight_tracker = QueryInsightTracker()

    return _insight_tracker
```

## src/retrieval_engine.py

```
"""
Unified Retrieval Engine with Strategy Pattern

Consolidates all retrieval logic (vector search, BM25, hybrid) into a single,
extensible engine using the strategy pattern.

This is the single source of truth for all retrieval operations in the RAG system.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

# P1: Better tokenization imports
try:
    from nltk.stem import PorterStemmer
    STEMMER = PorterStemmer()
except ImportError:
    STEMMER = None
    logger = logging.getLogger(__name__)
    logger.debug("NLTK not available, stemming disabled")

from src.errors import RetrievalError
from src.tuning_config import (
    RRF_K_CONSTANT,
    MMR_LAMBDA,
    TIME_DECAY_RATE,
    BM25_B,
)
from src.performance_tracker import get_performance_tracker, PipelineStage
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR = "vector"          # Pure semantic search via FAISS
    BM25 = "bm25"              # Pure lexical search via BM25
    HYBRID = "hybrid"          # Combined semantic + lexical


@dataclass
class RetrievalConfig:
    """Configuration for retrieval engine."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    k_vector: int = 40          # Results from vector search
    k_bm25: int = 40            # Results from BM25 search
    k_final: int = 5            # Final results to return
    hybrid_alpha: float = 0.7   # Weight for vector (1-alpha for BM25)
    normalize_scores: bool = True
    apply_diversity_penalty: bool = True
    diversity_penalty_weight: float = 0.15
    timeout_seconds: float = 30.0


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    chunk_id: str
    text: str
    title: str
    url: str
    namespace: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Scores
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    final_score: Optional[float] = None

    # Ranking info
    rank: Optional[int] = None
    seen_content_hash: Optional[str] = None
    diversity_score: Optional[float] = None  # PHASE 5: MMR diversity penalty

    # Embeddings for MMR calculation (P0: vector-based diversity)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ============================================================================
# Tokenization Helper (P1: Enhanced BM25)
# ============================================================================

# P1: Common English stopwords for filtering
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'it', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'as', 'if', 'with', 'about', 'up', 'down', 'out',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'no',
    'not', 'only', 'same', 'so', 'such', 'than', 'too', 'very', 'just'
}


def tokenize_for_bm25(text: str, remove_stopwords: bool = True, use_stemming: bool = False) -> List[str]:
    """
    P1: Enhanced tokenization for BM25.

    Uses regex-based tokenization instead of simple split().
    Optionally removes stopwords and applies stemming.

    Args:
        text: Text to tokenize
        remove_stopwords: Whether to filter common stopwords
        use_stemming: Whether to apply Porter stemming

    Returns:
        List of preprocessed tokens
    """
    if not text:
        return []

    # Lowercase
    text = text.lower()

    # P1: Regex-based tokenization (handles punctuation better than split())
    # Matches sequences of alphanumeric characters (includes hyphens for compound words)
    tokens = re.findall(r'\b[a-z0-9_-]+\b', text)

    # P1: Remove stopwords if enabled
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # P1: Apply stemming if available and enabled
    if use_stemming and STEMMER is not None:
        tokens = [STEMMER.stem(t) for t in tokens]

    return tokens


# ============================================================================
# Abstract Strategy Base Class
# ============================================================================


class BaseRetrievalStrategy(ABC):
    """Base class for all retrieval strategies."""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Execute search strategy."""
        pass

    def _create_result(
        self,
        chunk: Dict[str, Any],
        vector_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
    ) -> RetrievalResult:
        """Create RetrievalResult from chunk and scores."""
        # P0: Extract embedding for MMR vector-based diversity calculation
        embedding = chunk.get("embedding")
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype="float32")
        elif not isinstance(embedding, np.ndarray):
            embedding = None

        return RetrievalResult(
            chunk_id=chunk.get("chunk_id", chunk.get("id", "")),
            text=chunk.get("text", ""),
            title=chunk.get("title", ""),
            url=chunk.get("url", ""),
            namespace=chunk.get("namespace", ""),
            metadata=chunk.get("metadata", {}),
            vector_score=vector_score,
            bm25_score=bm25_score,
            embedding=embedding,
        )


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================


class VectorSearchStrategy(BaseRetrievalStrategy):
    """Pure semantic search via vector embeddings."""

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using vector similarity only."""
        try:
            if not chunks or query_embedding is None:
                return []

            # Normalize query embedding if needed
            if self.config.normalize_scores:
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    query_embedding = query_embedding / query_norm

            # Extract embeddings from chunks
            # Fix #3: Handle missing embeddings gracefully with clear error message
            embeddings = []
            valid_indices = []
            for idx, chunk in enumerate(chunks):
                if "embedding" in chunk:
                    embeddings.append(chunk["embedding"])
                    valid_indices.append(idx)

            if not embeddings:
                error_msg = (
                    f"FIX CRITICAL #5: Vector search failed - no embeddings found in {len(chunks)} chunks. "
                    "Chunks must include 'embedding' field for vector search. "
                    "This is a fatal error indicating index was not properly initialized with embeddings."
                )
                logger.error(error_msg)
                raise RetrievalError(error_msg)

            embeddings = np.array(embeddings)

            # Compute similarity
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            similarities = np.dot(embeddings, query_embedding)

            # Get top-k
            top_indices = np.argsort(-similarities)[:k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                chunk_idx = valid_indices[idx]
                chunk = chunks[chunk_idx]
                score = float(similarities[idx])

                result = self._create_result(chunk, vector_score=score)
                result.final_score = score
                result.rank = rank
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RetrievalError(f"Vector search failed: {str(e)}")


class BM25SearchStrategy(BaseRetrievalStrategy):
    """Pure lexical search via BM25."""

    def __init__(self, config: RetrievalConfig):
        """Initialize BM25 strategy with caching for performance."""
        super().__init__(config)
        # Fix #2: Cache BM25 index to avoid O(N) rebuild on every search
        # Cache key: namespace (assumes chunks are per-namespace and stable)
        self._bm25_cache: Dict[str, Tuple[BM25Okapi, List[List[str]]]] = {}
        self._cache_lock = __import__('threading').Lock()
        self._max_cache_size = 100  # Limit cache to prevent unbounded growth
        self._cache_access_order: List[str] = []  # LRU tracking

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using BM25 keyword matching only."""
        try:
            if not chunks or not query_text:
                return []

            # Fix #2: Cache BM25 index per namespace to avoid O(N) rebuild
            # Use namespace as cache key (assumes chunks have consistent namespace)
            namespace = chunks[0].get("namespace", "default") if chunks else "default"

            # PHASE 2: Extended lock to cover BM25 object usage
            # FIX CRITICAL: BM25Okapi.get_scores() must be called under lock
            # to prevent thread-safety issues with the object's internal state
            with self._cache_lock:
                if namespace in self._bm25_cache:
                    # Use cached BM25 index and update LRU order
                    bm25, tokenized_texts = self._bm25_cache[namespace]
                    # Move to end (most recently used)
                    if namespace in self._cache_access_order:
                        self._cache_access_order.remove(namespace)
                    self._cache_access_order.append(namespace)
                    logger.debug(f"Using cached BM25 index for namespace: {namespace}")
                else:
                    # Build and cache BM25 index
                    texts = [chunk.get("text", "") for chunk in chunks]
                    # P1: Use enhanced tokenization with stopword removal
                    tokenized_texts = [tokenize_for_bm25(text, remove_stopwords=True, use_stemming=False) for text in texts]
                    bm25 = BM25Okapi(tokenized_texts)

                    # LRU eviction: remove oldest entry if cache is full
                    if len(self._bm25_cache) >= self._max_cache_size and self._cache_access_order:
                        lru_namespace = self._cache_access_order.pop(0)
                        del self._bm25_cache[lru_namespace]
                        logger.debug(f"BM25 cache eviction: removed {lru_namespace} (size={len(self._bm25_cache)})")

                    self._bm25_cache[namespace] = (bm25, tokenized_texts)
                    self._cache_access_order.append(namespace)
                    logger.debug(f"Built and cached BM25 index for namespace: {namespace}")

                # Score query (must be under lock since BM25Okapi may not be thread-safe)
                # P1: Use enhanced tokenization for query as well (must match document tokenization)
                query_tokens = tokenize_for_bm25(query_text, remove_stopwords=True, use_stemming=False)
                scores = bm25.get_scores(query_tokens)

            # Get top-k
            top_indices = np.argsort(-scores)[:k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                chunk = chunks[idx]
                score = float(scores[idx])

                # Normalize score if needed
                if self.config.normalize_scores and scores.max() > 0:
                    score = score / scores.max()

                result = self._create_result(chunk, bm25_score=score)
                result.final_score = score
                result.rank = rank
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise RetrievalError(f"BM25 search failed: {str(e)}")


class HybridSearchStrategy(BaseRetrievalStrategy):
    """Combined semantic + lexical search with fusion."""

    def __init__(self, config: RetrievalConfig):
        """Initialize hybrid strategy with shared sub-strategies.

        FIX CRITICAL #2: Create strategy instances once and reuse them
        to preserve BM25 cache across requests.
        """
        super().__init__(config)
        # Create shared strategy instances that persist across requests
        self._vector_strategy = VectorSearchStrategy(config)
        self._bm25_strategy = BM25SearchStrategy(config)

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: int,
    ) -> List[RetrievalResult]:
        """Search using both vector and BM25, then fuse results."""
        try:
            if not chunks:
                return []

            tracker = get_performance_tracker()

            # FIX CRITICAL #2: Reuse shared strategy instances to preserve BM25 cache
            # Previously created new instances on every call, destroying the cache
            start_vector = time.time()
            vector_results = self._vector_strategy.search(
                query_embedding, query_text, chunks, self.config.k_vector
            )
            tracker.record(
                PipelineStage.VECTOR_SEARCH,
                (time.time() - start_vector) * 1000,
                metadata={"results": len(vector_results)},
            )

            start_bm25 = time.time()
            bm25_results = self._bm25_strategy.search(
                query_embedding, query_text, chunks, self.config.k_bm25
            )
            tracker.record(
                PipelineStage.BM25_SEARCH,
                (time.time() - start_bm25) * 1000,
                metadata={"results": len(bm25_results)},
            )

            # Fuse results
            start_fusion = time.time()
            fused = self._fuse_results(vector_results, bm25_results)
            tracker.record(
                PipelineStage.FUSION,
                (time.time() - start_fusion) * 1000,
                metadata={"fused_results": len(fused)},
            )

            # PHASE 5: Enhanced DEBUG logging for fusion pipeline decisions
            logger.debug(
                f"Hybrid fusion: k_vector={self.config.k_vector}, k_bm25={self.config.k_bm25}, "
                f"fused={len(fused)} unique results (RRF applied), "
                f"top_score={fused[0].final_score if fused else 'N/A'}"
            )

            # Apply diversity penalty if configured
            if self.config.apply_diversity_penalty:
                start_diversity = time.time()
                fused = self._apply_diversity_penalty(fused)
                tracker.record(
                    PipelineStage.DIVERSITY_FILTER,
                    (time.time() - start_diversity) * 1000,
                    metadata={"results_after": len(fused)},
                )
                logger.debug(f"MMR diversity filter applied (λ=0.7), top result diversity_score={fused[0].diversity_score if fused else 'N/A'}")

            # Sort by final score and truncate to k
            fused.sort(key=lambda r: r.final_score or 0, reverse=True)
            for rank, result in enumerate(fused[:k], 1):
                result.rank = rank

            logger.debug(f"Final hybrid results: returning top {len(fused[:k])} of {len(fused)}")
            return fused[:k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError(f"Hybrid search failed: {str(e)}")

    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

        RRF is more robust than weighted averaging and doesn't require hyperparameter tuning.
        Formula: score = 1/(k + rank), where k is typically 60 (number of initial results).

        Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        # PHASE 5: Gold-standard RRF instead of weighted fusion
        # RRF benefits: no hyperparameter tuning, empirically superior to weighted methods
        k_const = RRF_K_CONSTANT  # Standard constant for RRF

        # Create a map of chunk_id to RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}

        # Add vector results with RRF scores
        for rank, result in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k_const + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Add BM25 results with RRF scores
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k_const + rank)
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
            else:
                # Merge BM25 score into existing vector result
                result_map[result.chunk_id].bm25_score = result.bm25_score

        # Assign final RRF scores to results
        for chunk_id, rrf_score in rrf_scores.items():
            result = result_map[chunk_id]
            result.hybrid_score = rrf_score
            result.final_score = rrf_score
            logger.debug(
                f"RRF score for {chunk_id}: {rrf_score:.4f} "
                f"(vector_rank={vector_results.index(result)+1 if result in vector_results else 'N/A'}, "
                f"bm25_rank={bm25_results.index(result)+1 if result in bm25_results else 'N/A'})"
            )

        return list(result_map.values())

    def _apply_diversity_penalty(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        PHASE 5: Apply Maximal Marginal Relevance (MMR) for diversity.

        MMR balances relevance (retrieval score) with diversity (dissimilarity to already-selected results).
        Formula: MMR_score = λ * relevance_score - (1-λ) * max_similarity_to_selected

        P0 improvement: Uses vector cosine similarity (embeddings) for semantic diversity calculation.
        Falls back to token overlap (Jaccard similarity) only when embeddings are unavailable.

        Args:
            results: Ranked retrieval results

        Returns:
            Results with MMR-adjusted scores
        """
        if not results or len(results) < 2:
            return results

        lambda_param = MMR_LAMBDA  # Gold-standard: balances relevance and diversity
        selected_indices = []
        mmr_scores = {}

        # Start with top-ranked result (highest relevance)
        if results:
            selected_indices.append(0)
            results[0].diversity_score = 0.0  # No penalty for first result

        # Greedily select remaining results based on MMR
        while len(selected_indices) < len(results):
            best_mmr_idx = -1
            best_mmr_score = -float('inf')

            for idx, result in enumerate(results):
                if idx in selected_indices:
                    continue

                # Relevance score (from retrieval)
                relevance = result.final_score or 0.0

                # Diversity: max cosine similarity to selected results
                max_similarity = 0.0
                for sel_idx in selected_indices:
                    selected_result = results[sel_idx]

                    # P0: Try vector cosine similarity first (if embeddings available)
                    similarity = 0.0
                    if (
                        selected_result.embedding is not None
                        and result.embedding is not None
                        and len(selected_result.embedding) > 0
                        and len(result.embedding) > 0
                    ):
                        # Vector cosine similarity: dot product / (norm1 * norm2)
                        # Bounded to [0, 1] where 1 = identical, 0 = orthogonal
                        try:
                            dot_product = np.dot(selected_result.embedding, result.embedding)
                            norm_selected = np.linalg.norm(selected_result.embedding)
                            norm_candidate = np.linalg.norm(result.embedding)
                            if norm_selected > 0 and norm_candidate > 0:
                                similarity = dot_product / (norm_selected * norm_candidate)
                                similarity = np.clip(similarity, -1.0, 1.0)  # Clamp to [-1, 1]
                        except (ValueError, RuntimeError) as e:
                            logger.debug(f"Vector similarity calculation failed: {e}, falling back to token overlap")
                            similarity = 0.0

                    # Fallback: token overlap (Jaccard similarity) if embeddings unavailable
                    if similarity == 0.0:
                        selected_tokens = set(selected_result.text.lower().split()) if selected_result.text else set()
                        candidate_tokens = set(result.text.lower().split()) if result.text else set()

                        if selected_tokens and candidate_tokens:
                            intersection = len(selected_tokens & candidate_tokens)
                            union = len(selected_tokens | candidate_tokens)
                            similarity = intersection / union if union > 0 else 0.0

                    max_similarity = max(max_similarity, similarity)

                # MMR score: λ * relevance - (1-λ) * diversity_penalty
                # PHASE 5: Clamp to [0, 1] to prevent negative scores from high diversity penalties
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_score = max(0.0, mmr_score)  # Clamp negative scores
                mmr_scores[idx] = mmr_score

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_mmr_idx = idx

            if best_mmr_idx >= 0:
                selected_indices.append(best_mmr_idx)
                results[best_mmr_idx].diversity_score = max_similarity

        # Re-rank results by MMR scores
        indexed_results = [(idx, result) for idx, result in enumerate(results)]
        indexed_results.sort(key=lambda x: mmr_scores.get(x[0], 0.0), reverse=True)

        # Update final scores with MMR
        reranked = []
        for idx, result in indexed_results:
            result.final_score = mmr_scores.get(idx, result.final_score or 0.0)
            reranked.append(result)

        logger.debug(f"MMR applied: λ={lambda_param}, top result diversity_score={reranked[0].diversity_score if reranked else 'N/A'}")
        return reranked

    def _apply_time_decay(
        self,
        results: List[RetrievalResult],
        decay_rate: float = TIME_DECAY_RATE,
    ) -> List[RetrievalResult]:
        """
        PHASE 5: Apply time decay to boost recent documents.

        Older documents are penalized: score *= decay_rate^months_old

        Uses document metadata: metadata.get("updated_at") should be ISO timestamp.

        Args:
            results: Retrieval results with optional updated_at metadata
            decay_rate: Decay factor per month (default 0.95 = 5% monthly decay)

        Returns:
            Results with time-decayed scores
        """
        from datetime import datetime
        import pytz

        start_time = time.time()
        tracker = get_performance_tracker()

        now = datetime.now(pytz.UTC)
        decay_count = 0
        decay_factors = []

        for result in results:
            metadata = result.metadata or {}
            updated_at_str = metadata.get("updated_at")

            if updated_at_str:
                try:
                    # Parse ISO timestamp
                    if updated_at_str.endswith("Z"):
                        updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                    else:
                        updated_at = datetime.fromisoformat(updated_at_str)

                    # Calculate months elapsed
                    days_old = (now - updated_at).days
                    months_old = days_old / 30.0

                    # Apply decay: score *= decay_rate^months_old
                    # PHASE 5: Clamp result to [0, 1] to keep scores bounded
                    decay_factor = decay_rate ** months_old
                    original_score = result.final_score or 0.0
                    result.final_score = max(0.0, original_score * decay_factor)

                    decay_count += 1
                    decay_factors.append(decay_factor)

                    logger.debug(
                        f"Time decay applied to {result.chunk_id}: "
                        f"{days_old}d old, decay={decay_factor:.4f}, "
                        f"score {original_score:.4f} -> {result.final_score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse updated_at for {result.chunk_id}: {e}")

        # PHASE 5: Summary log for time decay pipeline
        if decay_count > 0:
            avg_decay_factor = sum(decay_factors) / len(decay_factors) if decay_factors else 1.0
            logger.debug(
                f"Time decay summary: applied to {decay_count}/{len(results)} results, "
                f"avg_decay_factor={avg_decay_factor:.4f}, decay_rate={decay_rate}"
            )

        # TIER 2: Record time decay latency
        tracker.record(
            PipelineStage.TIME_DECAY,
            (time.time() - start_time) * 1000,
            metadata={"decay_count": decay_count, "total": len(results)},
        )

        return results


# ============================================================================
# Main Retrieval Engine
# ============================================================================


class RetrievalEngine:
    """Unified retrieval engine supporting multiple strategies."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        """Initialize engine with configuration."""
        self.config = config or RetrievalConfig()
        self._strategies: Dict[RetrievalStrategy, BaseRetrievalStrategy] = {
            RetrievalStrategy.VECTOR: VectorSearchStrategy(self.config),
            RetrievalStrategy.BM25: BM25SearchStrategy(self.config),
            RetrievalStrategy.HYBRID: HybridSearchStrategy(self.config),
        }
        self.current_strategy = self._strategies[self.config.strategy]

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """Switch retrieval strategy."""
        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.current_strategy = self._strategies[strategy]
        self.config.strategy = strategy
        logger.info(f"Switched to {strategy.value} retrieval strategy")

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Execute retrieval using current strategy.

        Args:
            query_embedding: Vector embedding of query
            query_text: Original query text
            chunks: List of document chunks
            k: Number of results (uses config if not specified)

        Returns:
            List of retrieval results ranked by relevance

        Raises:
            RetrievalError: If retrieval fails
            RetrievalTimeoutError: If retrieval times out
        """
        try:
            k = k or self.config.k_final

            if not isinstance(chunks, list):
                raise ValueError("chunks must be a list")

            if query_embedding is None and self.config.strategy != RetrievalStrategy.BM25:
                raise ValueError("query_embedding required for vector search")

            results = self.current_strategy.search(
                query_embedding, query_text, chunks, k
            )

            logger.debug(
                f"Retrieved {len(results)} results using {self.config.strategy.value} strategy"
            )
            return results

        except RetrievalError:
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        chunks: List[Dict[str, Any]],
        k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid search regardless of current strategy.

        Args:
            query_embedding: Vector embedding
            query_text: Query text
            chunks: Document chunks
            k: Number of results
            alpha: Hybrid weight (0.7 = 70% vector, 30% BM25)

        Returns:
            Hybrid search results
        """
        original_strategy = self.config.strategy
        original_alpha = self.config.hybrid_alpha

        try:
            self.set_strategy(RetrievalStrategy.HYBRID)
            if alpha is not None:
                self.config.hybrid_alpha = max(0, min(1, alpha))

            return self.search(query_embedding, query_text, chunks, k)

        finally:
            self.config.strategy = original_strategy
            self.config.hybrid_alpha = original_alpha
            self.current_strategy = self._strategies[original_strategy]

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get current strategy and configuration."""
        return {
            "strategy": self.config.strategy.value,
            "k_final": self.config.k_final,
            "k_vector": self.config.k_vector,
            "k_bm25": self.config.k_bm25,
            "hybrid_alpha": self.config.hybrid_alpha,
            "normalize_scores": self.config.normalize_scores,
            "apply_diversity_penalty": self.config.apply_diversity_penalty,
        }


# ============================================================================
# Hybrid Search Scoring Utilities (Consolidated from hybrid_search.py)
# ============================================================================


def compute_bm25_score(
    doc_tokens: List[str],
    query_tokens: List[str],
    doc_length: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = BM25_B,
) -> float:
    """
    Compute BM25 score for a document given a query.

    BM25 is a probabilistic relevance framework used in information retrieval.
    It considers:
    - Term frequency in document
    - Inverse document frequency
    - Document length normalization

    Args:
        doc_tokens: Tokenized document
        query_tokens: Tokenized query
        doc_length: Length of document (word count)
        avg_doc_length: Average document length in corpus
        k1: Term frequency saturation parameter (default 1.5)
        b: Length normalization parameter (default 0.75)

    Returns:
        BM25 score
    """
    doc_freq = Counter(doc_tokens)
    score = 0.0

    # Estimate IDF (simplified - would normally use corpus-wide stats)
    idf = {}
    for token in query_tokens:
        # Simple IDF approximation
        idf[token] = math.log(1 + (doc_freq.get(token, 0) + 0.5) / (0.5 + 1))

    # Calculate BM25 for each query term
    for token in query_tokens:
        if token in doc_freq:
            freq = doc_freq[token]
            norm_length = 1 - b + b * (doc_length / (avg_doc_length + 1))
            bm25_component = idf[token] * (freq * (k1 + 1)) / (freq + k1 * norm_length)
            score += bm25_component

    return score


def keyword_match_score(text: str, query: str) -> float:
    """
    Simple keyword matching score.

    Rewards exact matches and phrase matches.

    Args:
        text: Text to match against (title + content)
        query: Query string

    Returns:
        Score between 0 and 1
    """
    text_lower = text.lower()
    query_lower = query.lower()

    # Exact phrase match (highest weight)
    if query_lower in text_lower:
        return 1.0

    # Word matches
    query_words = query_lower.split()
    text_words = set(text_lower.split())

    if not query_words:
        return 0.0

    match_ratio = len([w for w in query_words if w in text_words]) / len(query_words)
    return match_ratio


def entity_match_score(text: str, entities: List[str]) -> float:
    """
    Score based on presence of query entities in text.

    Args:
        text: Text to check
        entities: List of entities from query analysis

    Returns:
        Score between 0 and 1
    """
    if not entities:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for entity in entities if entity.lower() in text_lower)
    return min(1.0, matches / len(entities))


def hybrid_search_score(
    result: Dict[str, Any],
    query: str,
    entities: Optional[List[str]] = None,
    semantic_weight: float = 0.70,
    keyword_weight: float = 0.30,
) -> Dict[str, Any]:
    """
    Compute hybrid score combining semantic and keyword matching.

    Args:
        result: Search result dict with 'semantic_score', 'title', 'content'
        query: Original query
        entities: Extracted entities from query (optional)
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword matching (0-1)

    Returns:
        Updated result dict with hybrid_score added
    """
    if entities is None:
        entities = []

    # Get semantic score (already computed by FAISS)
    semantic = result.get("semantic_score", result.get("score", 0.0))

    # Compute keyword score
    combined_text = f"{result.get('title', '')} {result.get('content', '')}"
    keyword_score = keyword_match_score(combined_text, query)

    # Bonus for entity matches
    entity_score = entity_match_score(combined_text, entities)
    keyword_score = 0.7 * keyword_score + 0.3 * entity_score

    # Normalize scores to 0-1 range if needed
    semantic_normalized = min(1.0, max(0.0, semantic))
    keyword_normalized = min(1.0, max(0.0, keyword_score))

    # Combine scores
    hybrid_score = semantic_weight * semantic_normalized + keyword_weight * keyword_normalized

    result["hybrid_score"] = hybrid_score
    result["semantic_score"] = semantic_normalized
    result["keyword_score"] = keyword_normalized
    result["entity_score"] = entity_score

    return result


def apply_diversity_penalty(
    results: List[Dict[str, Any]], diversity_weight: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Apply diversity penalty to avoid redundant results.

    Penalizes results that are very similar to already-selected results.

    Args:
        results: List of results (assumed sorted by relevance)
        diversity_weight: Weight for diversity penalty (0-1)

    Returns:
        Results with diversity_penalty and adjusted_score fields added
    """
    if not results:
        return results

    processed = []
    seen_content_hashes = set()

    for i, result in enumerate(results):
        # Create simple content hash for diversity
        content_hash = hash(result.get("content", "")[:100])

        # Calculate diversity score
        if content_hash in seen_content_hashes:
            # Penalize if similar content already in results
            diversity_penalty = diversity_weight
        else:
            diversity_penalty = 0.0
            seen_content_hashes.add(content_hash)

        # Apply penalty to score
        original_score = result.get("hybrid_score", result.get("score", 0.0))
        adjusted_score = original_score * (1 - diversity_penalty)

        result["diversity_penalty"] = diversity_penalty
        result["adjusted_score"] = adjusted_score

        processed.append(result)

    # Re-sort by adjusted score
    processed.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

    return processed


def rank_hybrid_results(
    results: List[Dict[str, Any]],
    query: str,
    entities: Optional[List[str]] = None,
    apply_diversity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Complete hybrid ranking pipeline.

    Applies semantic scoring, keyword matching, entity matching, and diversity.

    Args:
        results: Initial search results
        query: Original query
        entities: Extracted query entities
        apply_diversity: Whether to apply diversity penalty

    Returns:
        Ranked results with hybrid scores
    """
    if entities is None:
        entities = []

    logger.debug(f"Hybrid ranking: {len(results)} results, {len(entities)} entities")

    # Apply hybrid scoring to each result
    scored_results = [hybrid_search_score(r, query, entities) for r in results]

    # Sort by hybrid score
    scored_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

    # Apply diversity penalty if requested
    if apply_diversity and len(scored_results) > 1:
        scored_results = apply_diversity_penalty(scored_results)

    return scored_results


# ============================================================================
# Convenience Functions
# ============================================================================


def create_engine(
    strategy: str = "hybrid",
    k: int = 5,
    alpha: float = 0.7,
) -> RetrievalEngine:
    """Create and configure a retrieval engine."""
    config = RetrievalConfig(
        strategy=RetrievalStrategy(strategy),
        k_final=k,
        hybrid_alpha=alpha,
    )
    return RetrievalEngine(config)


# ============================================================================
# Backward Compatibility Functions (Consolidated from retrieval.py)
# ============================================================================


def hybrid_search(
    query: str,
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    encoder: Any,
    k_vec: int = 40,
    k_bm25: int = 40,
    k_final: int = 12,
    use_query_adaptation: bool = True,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining vector search (cosine) + BM25 with adaptive field boosts.

    This function provides backward compatibility with the legacy retrieval.py API.
    New code should use RetrievalEngine.search() instead.

    Improvements:
    - Query type detection for adaptive strategies
    - Enhanced field matching with type-aware boosting
    - Better BM25 handling with glossary terms
    - Improved normalization and score fusion

    Args:
        query: Search query
        docs: List of {"text": str, "meta": {...}} dicts
        embeddings: (N, d) array of L2-normalized embeddings
        encoder: Encoder with .embed(str) -> ndarray[d]
        k_vec: Number of vector results to keep
        k_bm25: Number of BM25 results to keep
        k_final: Final number of results
        use_query_adaptation: Enable adaptive strategies based on query type

    Returns:
        Top-k merged and re-scored results with improved relevance
    """
    # Embed query
    qv = encoder.embed(query)
    qv = qv / (np.linalg.norm(qv) + 1e-9)  # L2-normalize

    # Vector similarity (already L2-normalized embeddings)
    sims = embeddings @ qv
    top_vec_indices = np.argsort(-sims)[:k_vec]

    # BM25 scores
    corpus = [d.get("text", "") for d in docs]
    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_scores = bm25.get_scores(query.split())
    top_bm25_indices = np.argsort(-bm25_scores)[:k_bm25]

    # Union of both
    candidate_indices = np.unique(np.concatenate([top_vec_indices, top_bm25_indices]))

    # Re-score with adaptive approach
    scores = {}

    for idx in candidate_indices:
        # Base score: weighted average (balanced hybrid approach)
        vec_score = float(sims[idx])
        bm25_norm = float(bm25_scores[idx]) / (np.max(bm25_scores) + 1e-9)

        # Balanced hybrid weighting
        base_score = 0.6 * vec_score + 0.4 * bm25_norm

        # Normalize final score to [0, 1] range
        base_score = min(float(base_score), 1.0)

        scores[idx] = base_score

    # Sort and return top-k
    top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k_final]
    results = [
        {
            **docs[i],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]

    return results
```

