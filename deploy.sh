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
