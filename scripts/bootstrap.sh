#!/bin/bash
#
# bootstrap.sh - One-command setup for Clockify RAG on corporate VPN
#
# Usage:
#   ./scripts/bootstrap.sh
#
# Prerequisites:
#   - On corporate VPN (required to access http://10.127.0.192:11434)
#   - Python 3.11 or 3.12 installed
#   - Git repository already cloned
#
# This script:
#   1. Creates/activates Python virtual environment
#   2. Installs dependencies
#   3. Verifies LLM connectivity on VPN
#   4. Prints next steps
#

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Configuration
VPN_LLM_URL="http://10.127.0.192:11434"
VENV_DIR=".venv"
PYTHON_MIN_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Production environment checks
check_bootstrap_environment() {
    # Verify we're in the repo root
    if [[ ! -f "$REPO_ROOT/requirements.txt" ]]; then
        error "Not in repository root (requirements.txt not found)"
        return 1
    fi

    # Warn about VPN connectivity for production
    if ! curl -s --connect-timeout 2 "$VPN_LLM_URL/api/tags" > /dev/null 2>&1; then
        warn "Corporate VPN LLM not accessible. Will require manual LLM configuration."
        warn "Set LLM_BASE_URL environment variable if using different endpoint."
    fi

    success "Bootstrap environment validation complete"
    return 0
}

# Helper functions
info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

success() {
    printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
}

warn() {
    printf "${YELLOW}[WARN]${NC} %s\n" "$1"
}

error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1"
}

# Main script
main() {
    info "Clockify RAG Bootstrap - Corporate VPN Setup"
    echo ""

    # Check bootstrap environment
    check_bootstrap_environment || exit 1
    echo ""

    # Step 1: Check Python version
    info "Checking Python version..."
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.11 or later."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    info "Found Python $PYTHON_VERSION"

    # Step 2: Create virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
        success "Virtual environment created"
    else
        warn "Virtual environment already exists at $VENV_DIR"
    fi

    # Step 3: Activate virtual environment
    info "Activating virtual environment..."
    . "$VENV_DIR/bin/activate"
    success "Virtual environment activated"

    # Step 4: Upgrade pip
    info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    success "Pip upgraded"

    # Step 5: Install dependencies
    if [ -f "requirements.txt" ]; then
        info "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt > /dev/null 2>&1
        success "Dependencies installed"
    else
        error "requirements.txt not found in repository root"
        exit 1
    fi

    # Step 6: Verify .env exists or create from sample
    if [ ! -f ".env" ]; then
        if [ -f ".env.sample" ]; then
            info "Creating .env from .env.sample..."
            cp .env.sample .env
            success ".env created with VPN defaults (LLM at $VPN_LLM_URL)"
        else
            warn ".env and .env.sample not found"
            info "Using environment defaults (VPN LLM will be used automatically)"
        fi
    else
        success ".env already exists"
    fi

    # Step 7: Check VPN connectivity to LLM
    info "Checking VPN connectivity to LLM at $VPN_LLM_URL..."
    if command -v curl &> /dev/null; then
        if curl -s --connect-timeout 3 "$VPN_LLM_URL/api/tags" > /dev/null 2>&1; then
            success "✓ VPN connectivity verified - LLM is accessible"
        else
            warn "✗ Cannot reach LLM at $VPN_LLM_URL"
            warn "   Make sure you are on the corporate VPN"
            warn "   Or override LLM_BASE_URL environment variable"
        fi
    else
        warn "curl not found - skipping connectivity check"
    fi

    # Step 8: Print next steps
    echo ""
    echo "${GREEN}========================================${NC}"
    echo "${GREEN}Bootstrap Complete!${NC}"
    echo "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Load data (first time only):"
    echo "     make ingest"
    echo ""
    echo "  2. Start the RAG server:"
    echo "     make serve"
    echo ""
    echo "  3. In another terminal, test the API:"
    echo "     curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \\"
    echo "       'http://localhost:7001/search?q=how%20to%20track%20time&k=5'"
    echo ""
    echo "  4. Or use the chat endpoint:"
    echo "     curl -X POST http://localhost:7001/chat \\"
    echo "       -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -d '{\"question\": \"How do I create a project?\", \"k\": 5}'"
    echo ""
    echo "Documentation:"
    echo "  - Quickstart: See README.md (Company AI Setup section)"
    echo "  - Operations: See docs/RUNBOOK.md"
    echo "  - Pre-deployment: See docs/PROD_CHECKLIST.md"
    echo ""
    echo "Environment:"
    echo "  - LLM URL: ${VPN_LLM_URL} (set via LLM_BASE_URL env var)"
    echo "  - LLM Model: gpt-oss:20b (set via LLM_MODEL env var)"
    echo "  - API Token: change-me (for dev; override API_TOKEN in prod)"
    echo ""
    echo "To reactivate the environment in a new shell:"
    echo "  source .venv/bin/activate"
    echo ""
}

# Run main function
main
