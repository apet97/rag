#!/bin/bash
set -euo pipefail

# ============================================================================
# CLOCKIFY RAG SYSTEM - FOOLPROOF INSTALLATION SCRIPT
# ============================================================================
# This script validates and sets up the RAG system from a fresh clone.
# Handles all dependency issues, path problems, and validation checks.
#
# Usage: ./install.sh
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.11"
PYTHON_PREFERRED_VERSION="3.12"
VENV_PATH=".venv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# PRODUCTION ENVIRONMENT CHECKS
# ============================================================================

check_installation_environment() {
    # Verify repository structure
    if [[ ! -f "$REPO_ROOT/requirements.txt" ]] || [[ ! -d "$REPO_ROOT/src" ]]; then
        log_error "Invalid repository structure. Expected requirements.txt and src/ directory"
        return 1
    fi

    # Check available disk space (require 5GB minimum)
    AVAILABLE_DISK=$( (df "$REPO_ROOT" 2>/dev/null || df -h) | awk 'NR==2 {print int($4)}' )
    if [[ $AVAILABLE_DISK -lt 5242880 ]]; then  # 5GB in KB
        log_warn "Less than 5GB disk space available. Installation may fail."
    fi

    # Verify write permissions
    if [[ ! -w "$REPO_ROOT" ]]; then
        log_error "No write permissions in repository directory"
        return 1
    fi

    log_success "Environment validation passed"
    return 0
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    echo "═════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═════════════════════════════════════════════════════════════"
    echo ""
}

# ============================================================================
# STEP 1: VALIDATE PYTHON VERSION
# ============================================================================

check_python() {
    print_header "Step 1: Validating Python Installation"

    # Try python3.12 first (preferred), then python3
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        log_error "Python 3 not found. Please install Python 3.12+:"
        echo "  macOS: brew install python@3.12"
        echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Found $PYTHON_CMD with version $PYTHON_VERSION"

    # Version check (3.11+)
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [[ $MAJOR -lt 3 ]] || [[ $MAJOR -eq 3 && $MINOR -lt 11 ]]; then
        log_error "Python $PYTHON_MIN_VERSION+ required, but $PYTHON_VERSION found"
        exit 1
    fi

    if [[ $MAJOR -eq 3 && $MINOR -ge 14 ]]; then
        log_warn "Python 3.14+ detected. Some packages may have compatibility issues."
        log_warn "Recommended: Use Python 3.12 for best stability"
    fi

    log_success "Python version OK: $PYTHON_VERSION"
}

# ============================================================================
# STEP 2: CREATE VIRTUAL ENVIRONMENT
# ============================================================================

setup_venv() {
    print_header "Step 2: Setting Up Virtual Environment"

    if [ -d "$VENV_PATH" ]; then
        log_warn "Virtual environment already exists at .venv"
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

    log_info "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_PATH"

    log_info "Upgrading pip, setuptools, wheel..."
    # Activate venv for upgrades
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1

    log_success "Virtual environment created and activated"
}

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

install_dependencies() {
    print_header "Step 3: Installing Dependencies"

    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found in $REPO_ROOT"
        exit 1
    fi

    source "$VENV_PATH/bin/activate"

    log_info "Installing packages from requirements.txt..."
    log_info "(This may take 2-5 minutes on first install)"

    if ! pip install -r requirements.txt; then
        log_error "Failed to install dependencies. Common issues:"
        echo "  • Network connectivity: Check your internet connection"
        echo "  • Binary packages: Some packages need compilation (lxml, numpy)"
        echo "    On macOS: brew install libxml2 libxslt"
        echo "    On Ubuntu: sudo apt install libxml2-dev libxslt1-dev"
        echo "  • Python version: Ensure Python 3.12 with: python3 --version"
        exit 1
    fi

    log_success "Dependencies installed successfully"
}

# ============================================================================
# STEP 4: VALIDATE PROJECT STRUCTURE
# ============================================================================

validate_structure() {
    print_header "Step 4: Validating Project Structure"

    local missing_items=()

    # Check critical directories exist (or create them)
    for dir in data/raw data/clean data/chunks data/domain index/faiss/clockify index/faiss/langchain logs; do
        if [ ! -d "$dir" ]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    log_success "All required directories exist"

    # Check critical files
    if [ ! -f "src/server.py" ]; then
        missing_items+=("src/server.py")
    fi
    if [ ! -f ".env" ]; then
        log_warn ".env not found. Will create from defaults."
    fi

    if [ ${#missing_items[@]} -gt 0 ]; then
        log_error "Missing critical files:"
        for item in "${missing_items[@]}"; do
            echo "  • $item"
        done
        exit 1
    fi

    log_success "Project structure validated"
}

# ============================================================================
# STEP 5: VALIDATE IMPORTS
# ============================================================================

validate_imports() {
    print_header "Step 5: Validating Module Imports"

    source "$VENV_PATH/bin/activate"

    log_info "Testing critical imports..."

    if ! python3 -c "from src.rerank import rerank; print('✓ rerank')" 2>/dev/null; then
        log_error "Failed to import src.rerank"
        exit 1
    fi

    if ! python3 -c "from src.paths import PROJECT_ROOT; print('✓ paths')" 2>/dev/null; then
        log_error "Failed to import src.paths"
        exit 1
    fi

    if ! python3 -c "import faiss; print('✓ faiss')" 2>/dev/null; then
        log_error "Failed to import faiss. FAISS installation may be broken."
        exit 1
    fi

    log_success "All critical imports work"
}

# ============================================================================
# STEP 6: SETUP CONFIGURATION
# ============================================================================

setup_env_file() {
    print_header "Step 6: Configuring Environment"

    if [ ! -f ".env" ]; then
        log_info "Creating .env from defaults..."
        cat > .env << 'EOF'
# LLM Configuration (Ollama)
LLM_BASE_URL=http://10.127.0.192:11434
LLM_MODEL=nomic-embed-text:latest
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0.0
LLM_RETRIES=3

# Embedding
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_DIM=768

# API Configuration
API_HOST=0.0.0.0
API_PORT=7000
API_TOKEN=change-me

# Data
NAMESPACES=clockify,langchain
RETRIEVAL_K=5

# Optional
MOCK_LLM=false
ENV=dev
EOF
        log_success ".env file created"
    else
        log_success ".env already configured"
    fi
}

# ============================================================================
# STEP 7: CREATE SETUP SUCCESS SUMMARY
# ============================================================================

print_success_summary() {
    print_header "Installation Complete! ✓"

    echo -e "${GREEN}Your RAG system is ready to use!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate venv (if not already): source .venv/bin/activate"
    echo "  2. Start the server: uvicorn src.server:app --port 7000"
    echo "  3. Visit http://localhost:7000/docs for API documentation"
    echo ""
    echo "For development (auto-reload on code changes):"
    echo "  uvicorn src.server:app --reload --port 7000"
    echo ""
    echo "Configuration:"
    echo "  • API Token (default): change-me (set API_TOKEN in .env)"
    echo "  • LLM URL: ${LLM_BASE_URL:-http://10.127.0.192:11434}"
    echo "  • Embedding Model: nomic-embed-text:latest (via Ollama)"
    echo ""
    echo "Documentation:"
    echo "  • QUICK_START.md - Get started in 5 minutes"
    echo "  • ANALYSIS_AND_STATUS.md - Full codebase analysis"
    echo "  • IMPROVEMENTS_ROADMAP.md - Enhancement plans"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    clear
    echo -e "${BLUE}"
    cat << 'BANNER'
╔═══════════════════════════════════════════════════════════╗
║   Clockify RAG System - Installation                      ║
║   Validating & Setting Up Environment                     ║
╚═══════════════════════════════════════════════════════════╝
BANNER
    echo -e "${NC}"

    cd "$REPO_ROOT" || exit 1

    # Run environment validation first
    check_installation_environment || exit 1

    # Run all installation steps
    check_python
    setup_venv
    install_dependencies
    validate_structure
    validate_imports
    setup_env_file

    print_success_summary
}

# Run main installation
main "$@"
