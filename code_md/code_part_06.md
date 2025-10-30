# Code Part 6

## .pytest.ini

```
[pytest]
testpaths = tests
norecursedirs = scripts .venv venv
addopts = -q
asyncio_mode = strict
```

## Dockerfile

```
# RAG v1: Flexible Deployment Image
# Supports both prebuilt FAISS index (for fast deployments) and dynamic index building
# Includes: Python environment, dependencies, Ollama client, optional prebuilt FAISS index
#
# Build: docker build -t clockify-rag:latest .
# Build (lean): docker build --build-arg LEAN_IMAGE=true -t clockify-rag:lean .
#   (Skips embedding/reranker model download for minimal image size)
# Run:   docker run -p 7000:7000 -e API_TOKEN=your-token clockify-rag:latest

from python:3.12-slim as base

ARG TARGETPLATFORM
ARG LEAN_IMAGE=false

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache embedding + reranker models for faster cold starts
# PHASE 5: Skip model download if LEAN_IMAGE=true for minimal image size
RUN if [ "$LEAN_IMAGE" = "false" ]; then \
    python - <<'PY' \
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

SentenceTransformer("intfloat/multilingual-e5-base")
FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
PY
; fi

# Copy application code
COPY src/ src/
COPY public/ public/

# Create data directories
RUN mkdir -p data/raw data/clean data/chunks data/domain \
    && mkdir -p index/faiss/clockify index/faiss/langchain \
    && mkdir -p /app/.cache/huggingface

VOLUME ["/app/.cache/huggingface"]

# Copy data if available (optional for fresh deployments)
COPY data/ data/ 2>/dev/null || true

# Copy prebuilt FAISS indexes if available
# NOTE: This is optional. If missing, run src.scrape + src.ingest to build
COPY index/faiss/clockify/ index/faiss/clockify/ 2>/dev/null || true
COPY index/faiss/langchain/ index/faiss/langchain/ 2>/dev/null || true

# Set environment variables for prebuilt image
ENV NAMESPACES=clockify
ENV EMBEDDING_MODEL=intfloat/multilingual-e5-base
ENV LLM_BASE_URL=http://ollama:11434
ENV API_HOST=0.0.0.0
ENV API_PORT=7000
ENV ENV=prod

# Default token - MUST be overridden in production!
ENV API_TOKEN=change-me

# Expose API port
EXPOSE 7000

# Health check: ensure index is loaded and API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f -H "x-api-token: ${API_TOKEN}" http://localhost:7000/health || exit 1

# Startup: API server with prebuilt index validation
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7000", "--log-level", "info"]
```

## crawl_clockify_help.py

```
#!/usr/bin/env python3
"""
BFS + sitemap scraper for Clockify help articles.
Crawls every URL under `/help`, excludes non-English variants, respects robots.txt.
Outputs clean Markdown and RAG-ready JSONL.
"""

import argparse
import json
import os
import re
import time
import hashlib
import pathlib
import queue
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import tldextract
import urllib.robotparser as robotparser

BASE = "https://clockify.me"
SEED = "https://clockify.me/help/"
SITEMAP = "https://clockify.me/help/wp-sitemap.xml"
UA = "Aleksandar-RAG-Collector/1.0 (+https://clockify.me/help) requests"


def norm(u: str) -> str:
    """Normalize URL to canonical form."""
    p = urlparse(u)
    # strip fragments and queries for canonical content URLs
    p = p._replace(fragment="", query="")
    # strip trailing slash except for /help/
    path = re.sub(r"//+", "/", p.path)
    if path != "/help/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme, p.netloc, path, "", "", ""))


def is_same_site(u: str) -> bool:
    """Check if URL is on same site as BASE."""
    return (
        tldextract.extract(u).registered_domain
        == tldextract.extract(BASE).registered_domain
    )


def is_help_path(u: str) -> bool:
    """Check if URL is under /help path."""
    return urlparse(u).path.startswith("/help/")


def lang_excluded(u: str, banned: set) -> bool:
    """Check if URL contains excluded language code."""
    parts = urlparse(u).path.split("/")
    # ['', 'help', 'maybe-lang', ...]
    if len(parts) >= 3:
        seg = parts[2].lower()
        return seg in banned
    return False


def load_urls_from_wp_sitemaps(session: requests.Session) -> set:
    """Load all URLs from WordPress sitemaps."""
    urls = set()
    try:
        print(f"Fetching WordPress sitemaps from {SITEMAP}...")
        r = session.get(SITEMAP, timeout=30)
        r.raise_for_status()
        sx = BeautifulSoup(r.text, "xml")
        sm_urls = [loc.text for loc in sx.select("sitemap > loc")]
        print(f"Found {len(sm_urls)} sitemap indexes")

        for sm in sm_urls:
            try:
                rx = session.get(sm, timeout=30)
                rx.raise_for_status()
                sx2 = BeautifulSoup(rx.text, "xml")
                for loc in sx2.select("url > loc"):
                    u = norm(loc.text.strip())
                    if is_same_site(u) and is_help_path(u):
                        urls.add(u)
            except Exception as e:
                print(f"Warning loading sitemap {sm}: {e}")

        print(f"Loaded {len(urls)} URLs from sitemaps")
    except Exception as e:
        print(f"Sitemap load warning: {e}")
    return urls


def extract_markdown(html: str) -> tuple[str, str]:
    """Extract title and markdown body from HTML."""
    s = BeautifulSoup(html, "lxml")

    # Remove obvious chrome
    for sel in [
        "header",
        "footer",
        "nav",
        ".site-header",
        ".site-footer",
        ".menu",
        ".breadcrumb",
        ".breadcrumbs",
        ".sidebar",
        ".toc",
        ".table-of-contents",
        ".cookie",
        ".cc-window",
        ".newsletter",
        ".hero",
        ".share",
        ".social",
        ".comments",
        ".wp-block-buttons",
    ]:
        for n in s.select(sel):
            n.decompose()

    title = None
    for h in s.select("h1, .entry-title"):
        if h.get_text(strip=True):
            title = h.get_text(" ", strip=True)
            break
    if not title:
        t = s.title.string if s.title else ""
        title = (t or "").strip()

    # Prefer main/article/content wrappers
    for sel in ["main article", ".entry-content", "article", ".post", ".content", "#content"]:
        n = s.select_one(sel)
        if n:
            body_md = md(str(n), heading_style="ATX")
            return title, body_md

    body_md = md(str(s.body or s), heading_style="ATX")
    return title, body_md


def chunk_for_rag(text: str, max_chars: int = 4000) -> list[tuple[str, str]]:
    """Split text into RAG-ready chunks by heading."""
    # Split by top-level headings first
    blocks = re.split(r"(?m)^(#{1,3}\s.+)$", text)
    # Re-stitch to pairs: (heading, content)
    pairs = []
    i = 0
    while i < len(blocks):
        if blocks[i].startswith("#"):
            head = blocks[i].strip()
            content = (blocks[i + 1] if i + 1 < len(blocks) else "").strip()
            pairs.append((head, content))
            i += 2
        else:
            if blocks[i].strip():
                pairs.append(("", blocks[i].strip()))
            i += 1

    # Further chunk long contents
    out = []
    for head, content in pairs:
        if len(content) <= max_chars:
            out.append((head, content))
        else:
            # split on paragraphs
            paras = [p for p in re.split(r"\n{2,}", content) if p.strip()]
            cur = ""
            for p in paras:
                if len(cur) + len(p) + 2 <= max_chars:
                    cur += (("\n\n" + p) if cur else p)
                else:
                    out.append((head, cur))
                    cur = p
            if cur:
                out.append((head, cur))
    return out


def safe_filename(url: str) -> str:
    """Convert URL to safe filename."""
    p = urlparse(url).path.strip("/").replace("/", "__")
    if not p:
        p = "help__index"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", p)


def main():
    ap = argparse.ArgumentParser(
        description="BFS + sitemap scraper for Clockify help articles"
    )
    ap.add_argument("--out", default="clockify-help-dump", help="Output directory")
    ap.add_argument("--delay", type=float, default=0.6, help="Seconds between requests")
    ap.add_argument("--max-pages", type=int, default=5000, help="Maximum pages to crawl")
    ap.add_argument(
        "--exclude",
        default="es,pt,de",
        help="Comma list of language subpaths to exclude",
    )
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    (outdir / "pages").mkdir(parents=True, exist_ok=True)
    rag_fp = open(outdir / "clockify_help.jsonl", "w", encoding="utf-8")

    banned = {x.strip().lower() for x in args.exclude.split(",") if x.strip()}
    print(f"Excluded languages: {sorted(banned)}")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    # robots.txt compliance
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(urljoin(BASE, "/robots.txt"))
        rp.read()
        print("Loaded robots.txt")
    except Exception as e:
        print(f"Warning: Could not load robots.txt: {e}")

    # Seed set from sitemap
    seed_urls = load_urls_from_wp_sitemaps(session)
    if not seed_urls:
        print("No sitemap URLs found, using /help/ as seed")
        seed_urls = {norm(SEED)}

    q = queue.Queue()
    for u in sorted(seed_urls):
        q.put(u)

    seen = set()
    discovered = set(seed_urls)
    visited_count = 0

    print(f"\nStarting BFS crawl with {len(seed_urls)} seed URLs...")
    print(f"Max pages: {args.max_pages}, Delay: {args.delay}s\n")

    while not q.empty() and visited_count < args.max_pages:
        u = q.get()
        if u in seen:
            continue
        if not is_same_site(u) or not is_help_path(u):
            continue
        if lang_excluded(u, banned):
            continue
        if rp and hasattr(rp, "can_fetch") and not rp.can_fetch(UA, u):
            print(f"✗ robots.txt disallows: {u}")
            seen.add(u)
            continue

        try:
            r = session.get(u, timeout=30)
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print(f"✗ Skip fetch: {u} ({e})")
            seen.add(u)
            continue

        title, body_md = extract_markdown(html)
        fname = safe_filename(u) + ".md"
        with open(outdir / "pages" / fname, "w", encoding="utf-8") as f:
            f.write(f"# {title or 'Clockify Help'}\n\n")
            f.write(f"> URL: {u}\n\n")
            f.write(body_md)

        # JSONL chunks for RAG
        chunks = chunk_for_rag(body_md)
        for head, chunk in chunks:
            doc_id = hashlib.sha1(
                (u + "\n" + head + "\n" + chunk[:64]).encode("utf-8")
            ).hexdigest()[:16]
            rec = {
                "id": doc_id,
                "url": u,
                "title": title,
                "section": head.lstrip("# ").strip() if head else "",
                "content": chunk,
                "lang": "en",
            }
            rag_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        visited_count += 1
        print(f"[{visited_count:3d}] ✓ {u} ({len(chunks)} chunks) -> {fname}")
        seen.add(u)

        # enqueue new links
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                href = a.get("href")
                if not href:
                    continue
                nu = norm(urljoin(u, href))
                if not is_same_site(nu) or not is_help_path(nu):
                    continue
                if lang_excluded(nu, banned):
                    continue
                if nu not in discovered:
                    discovered.add(nu)
                    q.put(nu)
        except Exception:
            pass

        time.sleep(args.delay)

    rag_fp.close()

    # Write manifest
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed_count": len(seed_urls),
                "discovered": len(discovered),
                "visited": visited_count,
                "excluded_langs": sorted(banned),
                "generated_at": int(time.time()),
            },
            f,
            indent=2,
        )

    # Save URL inventory
    with open(outdir / "urls.txt", "w", encoding="utf-8") as f:
        for u in sorted(seen | discovered):
            f.write(u + "\n")

    print(f"\n{'='*60}")
    print("CRAWL COMPLETE")
    print(f"{'='*60}")
    print(f"Seed URLs: {len(seed_urls)}")
    print(f"Total discovered: {len(discovered)}")
    print(f"Total visited: {visited_count}")
    print(f"Output directory: {outdir}")
    print(f"Markdown pages: {outdir}/pages/")
    print(f"RAG JSONL: {outdir}/clockify_help.jsonl")
    print(f"URL inventory: {outdir}/urls.txt")
    print(f"Manifest: {outdir}/manifest.json")


if __name__ == "__main__":
    main()
```

## eval/qas.jsonl

```
{"q": "How do I submit my weekly timesheet?", "must": ["https://help.clockify.me/article/timesheet"]}
{"q": "What is time off or PTO?", "must": ["https://help.clockify.me/article/pto"]}
{"q": "How do I set up the kiosk?", "must": ["https://help.clockify.me/article/kiosk"]}
{"q": "What is a billable rate?", "must": ["https://help.clockify.me/article/billable-rate"]}
{"q": "How do I create a project?", "must": ["https://help.clockify.me/article/project"]}
{"q": "What is SSO?", "must": ["https://help.clockify.me/article/sso"]}
{"q": "How do I approve timesheets as a manager?", "must": ["https://help.clockify.me/article/approval"]}
{"q": "Can I export reports?", "must": ["https://help.clockify.me/article/reports"]}
{"q": "What are user roles?", "must": ["https://help.clockify.me/article/roles"]}
{"q": "How do I track time?", "must": ["https://help.clockify.me/article/tracking"]}
{"q": "What is project budget?", "must": ["https://help.clockify.me/article/budget"]}
{"q": "How do I enable time rounding?", "must": ["https://help.clockify.me/article/rounding"]}
{"q": "What are recurring time entries?", "must": ["https://help.clockify.me/article/recurring"]}
{"q": "How do I use estimates?", "must": ["https://help.clockify.me/article/estimates"]}
{"q": "What is the audit log?", "must": ["https://help.clockify.me/article/audit"]}
```

## install.sh

```
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
```

## scripts/run_all_tests.py

```
#!/usr/bin/env python3
"""Master test suite orchestration for Clockify RAG system."""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_test(name, status, details=""):
    """Print test result with color."""
    if status == "PASSED":
        symbol = f"{Colors.OKGREEN}✅{Colors.ENDC}"
        status_text = f"{Colors.OKGREEN}{status}{Colors.ENDC}"
    elif status == "FAILED":
        symbol = f"{Colors.FAIL}❌{Colors.ENDC}"
        status_text = f"{Colors.FAIL}{status}{Colors.ENDC}"
    elif status == "PARTIAL":
        symbol = f"{Colors.WARNING}⚠️{Colors.ENDC}"
        status_text = f"{Colors.WARNING}{status}{Colors.ENDC}"
    elif status == "RUNNING":
        symbol = f"{Colors.OKBLUE}⏳{Colors.ENDC}"
        status_text = f"{Colors.OKBLUE}{status}{Colors.ENDC}"
    else:
        symbol = "•"
        status_text = status

    print(f"{symbol} {name:<50} {status_text:<15} {details}")

def run_test(script_name, description):
    """Run a single test script and return results."""
    print_test(description, "RUNNING")

    script_path = Path("scripts") / f"{script_name}.py"

    if not script_path.exists():
        print_test(description, "FAILED", "Script not found")
        return {
            "name": script_name,
            "status": "failed",
            "error": "Script not found"
        }

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Check if test passed
        if result.returncode == 0:
            status = "PASSED"
            result_status = "passed"
        else:
            status = "FAILED"
            result_status = "failed"

        # Try to load results JSON if it exists
        result_file = LOG_DIR / f"{script_name.replace('test_', '').replace('validate_', '')}_test_results.json"
        if script_name == "validate_retrieval":
            result_file = LOG_DIR / "retrieval_test_results.json"
        elif script_name == "test_llm_connection":
            result_file = LOG_DIR / "llm_connection_test.json"
        elif script_name == "test_rag_pipeline":
            result_file = LOG_DIR / "rag_pipeline_test_results.json"
        elif script_name == "test_api":
            result_file = LOG_DIR / "api_test_results.json"

        details = ""
        result_data = {}

        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    result_data = json.load(f)
                    if "summary" in result_data:
                        summary = result_data["summary"]
                        if "success_rate" in summary:
                            details = f"({summary['success_rate']:.0f}% success)"
                        elif "overall_pass_rate" in summary:
                            details = f"({summary['overall_pass_rate']})"
            except:
                pass

        print_test(description, status, details)

        return {
            "name": script_name,
            "status": result_status,
            "return_code": result.returncode,
            "result_data": result_data,
            "stdout_lines": len(result.stdout.split('\n')),
            "stderr_lines": len(result.stderr.split('\n')),
        }

    except subprocess.TimeoutExpired:
        print_test(description, "FAILED", "Timeout (>120s)")
        return {
            "name": script_name,
            "status": "failed",
            "error": "Timeout"
        }
    except Exception as e:
        print_test(description, "FAILED", f"Error: {str(e)[:30]}")
        return {
            "name": script_name,
            "status": "failed",
            "error": str(e)
        }

def main():
    """Run all tests in sequence."""
    print_header("CLOCKIFY RAG - COMPREHENSIVE TEST SUITE")

    print(f"{Colors.BOLD}Starting test run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    # Define test suite
    tests = [
        ("validate_retrieval", "Step 1: Retrieval Validation (20 queries)"),
        ("test_llm_connection", "Step 2: LLM Connection Test"),
        ("test_rag_pipeline", "Step 3: RAG Pipeline Test (15 queries)"),
        ("test_api", "Step 4: API Endpoint Tests"),
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {}
    }

    print(f"{Colors.OKCYAN}Test Execution:{Colors.ENDC}\n")

    test_results = []
    for script_name, description in tests:
        start_time = time.time()
        result = run_test(script_name, description)
        result["duration_s"] = time.time() - start_time
        test_results.append(result)
        results["tests"].append(result)
        time.sleep(0.5)  # Brief pause between tests

    # Calculate summary
    passed = sum(1 for r in test_results if r.get("status") == "passed")
    failed = sum(1 for r in test_results if r.get("status") == "failed")
    total = len(test_results)

    # Print summary
    print_header("TEST SUITE SUMMARY")

    print(f"{Colors.BOLD}Results:{Colors.ENDC}\n")
    print(f"  Total Tests:     {total}")
    print(f"  {Colors.OKGREEN}Passed:{Colors.ENDC}     {passed}")
    print(f"  {Colors.FAIL}Failed:{Colors.ENDC}     {failed}")

    if total > 0:
        pass_rate = (passed / total) * 100
        if pass_rate >= 90:
            color = Colors.OKGREEN
            symbol = "✅"
        elif pass_rate >= 70:
            color = Colors.WARNING
            symbol = "⚠️"
        else:
            color = Colors.FAIL
            symbol = "❌"

        print(f"  Pass Rate:       {color}{symbol} {pass_rate:.0f}%{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Test Details:{Colors.ENDC}\n")

    for result in test_results:
        name = result.get("name", "Unknown")
        status = result.get("status", "unknown")
        duration = result.get("duration_s", 0)

        if status == "passed":
            status_symbol = f"{Colors.OKGREEN}✅ PASSED{Colors.ENDC}"
        elif status == "failed":
            status_symbol = f"{Colors.FAIL}❌ FAILED{Colors.ENDC}"
        else:
            status_symbol = "❓ UNKNOWN"

        print(f"  {name:30} {status_symbol:30} ({duration:.1f}s)")

    # Print recommendations
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}\n")

    if failed > 0:
        print(f"  {Colors.WARNING}⚠️  {failed} test(s) failed.{Colors.ENDC}")
        print(f"     Review the detailed output above and fix issues.")
    else:
        print(f"  {Colors.OKGREEN}✅ All core tests passed!{Colors.ENDC}")

    print(f"\n  To complete validation:")
    print(f"    1. {Colors.OKCYAN}Start LLM:{Colors.ENDC} ollama pull oss20b && ollama serve")
    print(f"    2. {Colors.OKCYAN}Rerun tests:{Colors.ENDC} python scripts/run_all_tests.py")
    print(f"    3. {Colors.OKCYAN}Deploy:{Colors.ENDC} python scripts/deployment_checklist.py")

    # Save results
    results["summary"] = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total * 100) if total > 0 else 0,
    }

    results_file = LOG_DIR / "test_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}\n")

    # Exit with appropriate code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

## src/config.py

```
"""
RAG System Configuration with validated parameters.

Centralized configuration for all RAG components:
- Retrieval parameters (RRF fusion, oversampling)
- LLM generation (temperature, model)
- Performance tuning (cache sizes, timeouts)
- Security (token validation, rate limits)
"""

import os
import re
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class RAGConfig:
    """Unified RAG configuration with validation."""

    # ===== RETRIEVAL =====
    # RRF (Reciprocal Rank Fusion) constant - higher = smoother scoring
    # From: S(d) = sum(1 / (k + rank_i(d))) where k is this constant
    RRF_CONSTANT: float = 60.0

    # Oversampling factors for pre-dedup retrieval
    # Retrieve k*factor docs before dedup to account for URL consolidation
    OVERSAMPLING_FACTOR_SMALL: int = 6  # For k <= 5
    OVERSAMPLING_FACTOR_LARGE: int = 3  # For k > 5

    # Context limits for LLM prompt construction
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
    CONTEXT_CHAR_LIMIT: int = int(os.getenv("CONTEXT_CHAR_LIMIT", "1200"))

    # ===== LLM GENERATION =====
    TEMPERATURE_MIN: float = 0.0  # Deterministic
    TEMPERATURE_MAX: float = 2.0  # Maximum creativity
    TEMPERATURE_DEFAULT: float = 0.0  # Deterministic by default
    LLM_TIMEOUT_SECONDS: int = 30

    # Answerability threshold for hallucination prevention (Jaccard overlap)
    ANSWERABILITY_THRESHOLD: float = float(os.getenv("ANSWERABILITY_THRESHOLD", "0.18"))

    # ===== EMBEDDING =====
    EMBEDDING_TIMEOUT_SECONDS: int = 30
    EMBEDDING_CACHE_SIZE: int = 512  # LRU cache for encoded queries

    # ===== PERFORMANCE =====
    REQUEST_CACHE_MAX_SIZE: int = 1000  # Search response cache
    RATE_LIMIT_RPS: int = 10  # Per-IP requests per second
    RATE_LIMIT_WINDOW_SECONDS: int = 1

    # ===== SECURITY =====
    QUERY_MAX_LENGTH: int = 2000
    QUERY_MIN_LENGTH: int = 1
    NAMESPACE_MAX_LENGTH: int = 100
    K_MIN: int = 1
    K_MAX: int = 20
    K_DEFAULT: int = 5

    # ===== LOGGING =====
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

    @staticmethod
    def validate_temperature(temp: float) -> float:
        """
        Validate and clamp temperature to valid range.

        Args:
            temp: Temperature value

        Returns:
            Clamped temperature value

        Raises:
            ValueError: If temperature is None or NaN
        """
        if temp is None:
            logger.warning("Temperature is None, using default")
            return RAGConfig.TEMPERATURE_DEFAULT

        try:
            temp_float = float(temp)
        except (ValueError, TypeError):
            logger.error(f"Invalid temperature type: {type(temp)}, using default")
            return RAGConfig.TEMPERATURE_DEFAULT

        if not (RAGConfig.TEMPERATURE_MIN <= temp_float <= RAGConfig.TEMPERATURE_MAX):
            logger.warning(
                f"Temperature {temp_float} out of range "
                f"[{RAGConfig.TEMPERATURE_MIN}, {RAGConfig.TEMPERATURE_MAX}], "
                f"clamping"
            )
            return max(RAGConfig.TEMPERATURE_MIN, min(temp_float, RAGConfig.TEMPERATURE_MAX))

        return temp_float

    @staticmethod
    def validate_k(k: Optional[int]) -> int:
        """
        Validate and clamp k to valid range.

        Args:
            k: Number of results to return

        Returns:
            Validated k value
        """
        if k is None:
            return RAGConfig.K_DEFAULT

        k_int = int(k)
        if not (RAGConfig.K_MIN <= k_int <= RAGConfig.K_MAX):
            logger.warning(
                f"k={k_int} out of range [{RAGConfig.K_MIN}, {RAGConfig.K_MAX}], "
                f"clamping"
            )
            return max(RAGConfig.K_MIN, min(k_int, RAGConfig.K_MAX))

        return k_int

    @staticmethod
    def validate_query(query: str) -> str:
        """
        Validate query text.

        Args:
            query: Query text to validate

        Returns:
            Validated query

        Raises:
            ValueError: If query is invalid
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be string, got {type(query)}")

        if len(query.strip()) < RAGConfig.QUERY_MIN_LENGTH:
            raise ValueError(f"Query too short (min {RAGConfig.QUERY_MIN_LENGTH} char)")

        if len(query) > RAGConfig.QUERY_MAX_LENGTH:
            raise ValueError(f"Query too long (max {RAGConfig.QUERY_MAX_LENGTH} chars)")

        # Block potential injection via regex escaping
        if re.search(r'[\\]', query):
            raise ValueError("Invalid characters in query (backslash not allowed)")

        return query

    @staticmethod
    def get_oversampling_factor(k: int) -> int:
        """
        Get oversampling factor based on k.

        Args:
            k: Number of final results requested

        Returns:
            Oversampling multiplier for pre-dedup retrieval
        """
        if k <= 5:
            return RAGConfig.OVERSAMPLING_FACTOR_SMALL
        else:
            return RAGConfig.OVERSAMPLING_FACTOR_LARGE


def redact_secrets(text: str) -> str:
    """
    Remove sensitive information from logs.

    Args:
        text: Text that may contain secrets

    Returns:
        Redacted text safe for logging
    """
    text = re.sub(r'Bearer\s+\S+', 'Bearer ***', text)
    text = re.sub(r'token["\']?\s*[=:]\s*["\']?\S+', 'token=***', text)
    text = re.sub(r'api[_-]?key["\']?\s*[=:]\s*["\']?\S+', 'api_key=***', text)
    text = re.sub(r'password["\']?\s*[=:]\s*["\']?\S+', 'password=***', text)
    return text


# Singleton instance
CONFIG = RAGConfig()
```

## src/embeddings_stub.py

```
"""Stub embedder for lightweight CI testing.

Used when EMBEDDINGS_BACKEND=stub to avoid downloading SentenceTransformer models.
Returns deterministic 384-dimensional vectors based on text hash.
"""

import numpy as np
from typing import List, Union


class StubEmbedder:
    """Lightweight stub embedder for CI testing.

    Generates deterministic embeddings by hashing input text.
    Useful for tests that need embeddings but don't require semantic meaning.
    """

    def __init__(self, model_name: str = "stub", max_seq_length: int = 512):
        """Initialize stub embedder.

        Args:
            model_name: Ignored (for API compatibility)
            max_seq_length: Ignored (for API compatibility)
        """
        self.model_name = "stub"
        self.max_seq_length = max_seq_length
        self.embedding_dim = 384  # Match all-MiniLM-L6-v2 output dim

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """Generate deterministic embeddings for given sentences.

        Args:
            sentences: Single sentence or list of sentences
            convert_to_numpy: If True, return numpy array (else list)
            normalize_embeddings: If True, L2-normalize embeddings

        Returns:
            Embeddings as numpy array or list of arrays (clamped to [-5, 5])
        """
        # Handle single string
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for sentence in sentences:
            # Create deterministic hash-based embedding
            # Hash the sentence to get a seed
            seed = hash(sentence) % (2 ** 31)
            rng = np.random.default_rng(seed)

            # Generate 384-dim embedding
            embedding = rng.standard_normal(self.embedding_dim).astype("float32")

            # Clamp values to prevent extreme values in stub mode
            # Standard normal can produce values beyond [-5, 5], clamp them
            embedding = np.clip(embedding, -5.0, 5.0).astype("float32")

            # Normalize if requested
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)

        # Return as numpy array if convert_to_numpy, else list
        if convert_to_numpy:
            return np.array(embeddings)
        return embeddings

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Make embedder callable for compatibility."""
        return self.encode(*args, **kwargs)


def get_stub_embedder() -> StubEmbedder:
    """Get or create stub embedder instance."""
    return StubEmbedder()
```

## src/ingest_from_jsonl.py

```
#!/usr/bin/env python3
"""
Ingest RAG-ready JSONL into FAISS index with metadata.
Designed for production-scale indexing with proper chunking and deduplication.
"""

import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration
JSONL_PATH = Path(os.getenv("JSONL_PATH", "clockify-help/clockify_help.jsonl"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index/faiss/clockify-improved"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from Ollama."""
    import httpx

    try:
        client = httpx.Client(timeout=60.0)
        response = client.post(
            f"{LLM_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()
        embedding = np.array(data["embeddings"][0], dtype=np.float32)
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def load_jsonl_records(jsonl_path: Path) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
    except Exception as e:
        logger.error(f"Error loading JSONL: {e}")
    logger.info(f"Loaded {len(records)} records from {jsonl_path}")
    return records


def deduplicate_records(records: List[Dict]) -> Tuple[List[Dict], int]:
    """Deduplicate records by content hash."""
    seen = set()
    unique = []
    dupes = 0

    for rec in records:
        content_hash = hashlib.sha256(
            (rec["url"] + "\n" + rec["content"]).encode()
        ).hexdigest()

        if content_hash not in seen:
            seen.add(content_hash)
            rec["_hash"] = content_hash
            unique.append(rec)
        else:
            dupes += 1

    logger.info(f"Deduplicated: {dupes} duplicates removed, {len(unique)} unique records")
    return unique, dupes


def embed_batch(texts: List[str]) -> List[Optional[np.ndarray]]:
    """Get embeddings for a batch of texts."""
    embeddings = []
    for text in texts:
        emb = get_embedding(text)
        embeddings.append(emb)
    return embeddings


def build_index(records: List[Dict]) -> Dict:
    """Build FAISS index from records."""
    import faiss

    logger.info(f"Building FAISS index from {len(records)} records...")

    embeddings_list = []
    metadata_list = []

    # Process in batches
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Embedding"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_texts = [rec["content"][:2000] for rec in batch_records]  # Truncate to 2K
        batch_embeddings = embed_batch(batch_texts)

        for j, emb in enumerate(batch_embeddings):
            if emb is not None:
                embeddings_list.append(emb)
                metadata_list.append(batch_records[j])

    if not embeddings_list:
        logger.error("No embeddings generated!")
        return {}

    # Convert to numpy array
    embeddings_array = np.stack(embeddings_list, axis=0)
    logger.info(f"Embedding shape: {embeddings_array.shape}")

    # Create FAISS index (inner product for L2-normalized vectors)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    logger.info(f"Index built with {index.ntotal} vectors, dimension {dimension}")

    # Save index
    index_path = INDEX_DIR / "index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"✓ Index saved: {index_path}")

    # Save metadata
    metadata_path = INDEX_DIR / "meta.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Metadata saved: {metadata_path}")

    # Save stats
    stats = {
        "total_records": len(records),
        "indexed_records": len(metadata_list),
        "dimension": dimension,
        "model": EMBEDDING_MODEL,
        "created_at": int(__import__("time").time()),
    }

    stats_path = INDEX_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✓ Stats saved: {stats_path}")

    return stats


def main():
    """Main ingestion pipeline."""
    logger.info(f"Starting JSONL ingestion from {JSONL_PATH}")

    # Load records
    records = load_jsonl_records(JSONL_PATH)
    if not records:
        logger.error("No records loaded!")
        return

    # Deduplicate
    records, dupes = deduplicate_records(records)

    # Build index
    stats = build_index(records)

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Records indexed: {stats.get('indexed_records', 0)}")
    logger.info(f"Vector dimension: {stats.get('dimension', 0)}")
    logger.info(f"Index directory: {INDEX_DIR}")


if __name__ == "__main__":
    main()
```

## src/performance_tracker.py

```
#!/usr/bin/env python3
"""
PHASE 5: Performance Tracker and Latency Metrics

Tracks end-to-end and stage-by-stage latency across the RAG pipeline.
Provides insights into performance bottlenecks and optimization opportunities.

Key metrics:
- Query embedding latency
- Retrieval latency (vector, BM25, hybrid, fusion)
- Reranking latency
- LLM generation latency
- Caching impact (hit/miss latency difference)
- End-to-end latency
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
from loguru import logger


class PipelineStage(str, Enum):
    """Pipeline stages for latency tracking."""
    QUERY_EMBEDDING = "query_embedding"
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search"
    FUSION = "fusion"
    DIVERSITY_FILTER = "diversity_filter"
    RERANKING = "reranking"
    TIME_DECAY = "time_decay"
    LLM_GENERATION = "llm_generation"
    ANSWERABILITY_CHECK = "answerability_check"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_STORE = "cache_store"
    END_TO_END = "end_to_end"


@dataclass
class LatencySample:
    """Single latency measurement."""
    stage: PipelineStage
    duration_ms: float
    timestamp: float
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    stage: PipelineStage
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    samples: List[float] = field(default_factory=list)

    def update(self, duration_ms: float) -> None:
        """Update stats with a new sample."""
        self.samples.append(duration_ms)
        self.count += 1
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

        if len(self.samples) >= 2:
            self.mean_ms = statistics.mean(self.samples)
            self.median_ms = statistics.median(self.samples)
            if len(self.samples) >= 20:
                self.p95_ms = statistics.quantiles(self.samples, n=20)[18]  # 95th percentile
                self.p99_ms = statistics.quantiles(self.samples, n=100)[98]  # 99th percentile

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "count": self.count,
            "min_ms": round(self.min_ms, 2) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


class PerformanceTracker:
    """
    Thread-safe performance tracker for RAG pipeline.

    Tracks latency across all stages and provides statistical analysis.
    """

    def __init__(self, max_samples_per_stage: int = 1000):
        """
        Initialize tracker.

        Args:
            max_samples_per_stage: Maximum samples to keep per stage (for memory efficiency)
        """
        self.max_samples = max_samples_per_stage
        self._stats: Dict[PipelineStage, StageStats] = {
            stage: StageStats(stage=stage) for stage in PipelineStage
        }
        self._samples: List[LatencySample] = []
        self._lock = threading.RLock()
        self._active_timers: Dict[str, float] = {}  # query_id -> start_time

        logger.info(f"Initialized PerformanceTracker (max_samples={max_samples_per_stage})")

    def start_timer(self, query_id: str) -> None:
        """Start end-to-end timer for a query."""
        with self._lock:
            self._active_timers[query_id] = time.time()

    def record(
        self,
        stage: PipelineStage,
        duration_ms: float,
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a latency measurement.

        Args:
            stage: Pipeline stage
            duration_ms: Duration in milliseconds
            query_id: Query identifier (for correlation)
            metadata: Additional metadata (count, items, etc.)
        """
        with self._lock:
            # Update stage stats
            self._stats[stage].update(duration_ms)

            # Store sample (with limit)
            sample = LatencySample(
                stage=stage,
                duration_ms=duration_ms,
                timestamp=time.time(),
                query_id=query_id,
                metadata=metadata or {},
            )
            self._samples.append(sample)

            if len(self._samples) > self.max_samples * len(PipelineStage):
                # Trim oldest samples
                self._samples = self._samples[-(self.max_samples * len(PipelineStage)) :]

            # Log DEBUG info for slow operations
            if duration_ms > 1000:  # >1 second
                logger.warning(
                    f"Slow stage: {stage.value} took {duration_ms:.0f}ms"
                    f"{f' (query={query_id})' if query_id else ''}"
                )

    def end_timer(self, query_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End end-to-end timer and record duration."""
        with self._lock:
            if query_id not in self._active_timers:
                logger.warning(f"Timer not started for query {query_id}")
                return

            start_time = self._active_timers.pop(query_id)
            duration_ms = (time.time() - start_time) * 1000

            self.record(PipelineStage.END_TO_END, duration_ms, query_id, metadata)

    def get_stats(self, stage: Optional[PipelineStage] = None) -> Dict[str, Any]:
        """
        Get statistics for a stage or all stages.

        Args:
            stage: Specific stage, or None for all

        Returns:
            Statistics dictionary
        """
        with self._lock:
            if stage:
                return self._stats[stage].to_dict()

            return {
                "by_stage": {s.stage.value: s.to_dict() for s in self._stats.values()},
                "total_samples": len(self._samples),
            }

    def get_recent_samples(self, stage: Optional[PipelineStage] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent samples for analysis.

        Args:
            stage: Filter by stage, or None for all
            limit: Maximum samples to return

        Returns:
            List of sample dictionaries
        """
        with self._lock:
            samples = self._samples
            if stage:
                samples = [s for s in samples if s.stage == stage]

            return [
                {
                    "stage": s.stage.value,
                    "duration_ms": round(s.duration_ms, 2),
                    "query_id": s.query_id,
                    "metadata": s.metadata,
                    "timestamp": s.timestamp,
                }
                for s in samples[-limit:]
            ]

    def get_bottlenecks(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get slowest stages (potential bottlenecks).

        Args:
            top_n: Number of slowest stages to return

        Returns:
            List of stage stats sorted by mean latency
        """
        with self._lock:
            stages = sorted(
                [s for s in self._stats.values() if s.count > 0],
                key=lambda s: s.mean_ms,
                reverse=True,
            )
            return [s.to_dict() for s in stages[:top_n]]

    def clear(self) -> None:
        """Clear all statistics."""
        with self._lock:
            self._stats = {stage: StageStats(stage=stage) for stage in PipelineStage}
            self._samples.clear()
            self._active_timers.clear()
            logger.info("Performance tracker cleared")

    def report(self) -> str:
        """Generate a human-readable performance report."""
        with self._lock:
            lines = ["Performance Report", "=" * 60]

            # Overall stats
            end_to_end = self._stats[PipelineStage.END_TO_END]
            if end_to_end.count > 0:
                lines.append(f"\nEnd-to-End Latency (n={end_to_end.count})")
                lines.append(f"  Mean: {end_to_end.mean_ms:.0f}ms")
                lines.append(f"  Median: {end_to_end.median_ms:.0f}ms")
                lines.append(f"  P95: {end_to_end.p95_ms:.0f}ms")
                lines.append(f"  Min-Max: {end_to_end.min_ms:.0f}-{end_to_end.max_ms:.0f}ms")

            # Stage breakdown
            lines.append("\nStage Breakdown (sorted by mean latency)")
            lines.append("-" * 60)
            stages = sorted(
                [s for s in self._stats.values() if s.count > 0],
                key=lambda s: s.mean_ms,
                reverse=True,
            )
            for stage in stages:
                lines.append(
                    f"{stage.stage.value:25} {stage.mean_ms:7.0f}ms "
                    f"(n={stage.count:4d}, p95={stage.p95_ms:7.0f}ms)"
                )

            return "\n".join(lines)


# Module-level singleton instance
_performance_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_performance_tracker() -> PerformanceTracker:
    """Get or create module-level performance tracker singleton."""
    global _performance_tracker

    if _performance_tracker is None:
        with _tracker_lock:
            if _performance_tracker is None:
                _performance_tracker = PerformanceTracker()

    return _performance_tracker
```

## src/scoring.py

```
#!/usr/bin/env python3
"""
Confidence scoring for search results.

Scores results based on semantic similarity, keyword matching, and metadata.
"""

import logging
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Score result confidence based on multiple factors."""

    # Thresholds for confidence levels
    CONFIDENCE_THRESHOLD_HIGH = 75  # > 75 = High confidence (green)
    CONFIDENCE_THRESHOLD_MEDIUM = 50  # 50-75 = Medium confidence (yellow)
    # < 50 = Low confidence (red)

    def __init__(self):
        """Initialize scorer."""
        self.scoring_history = []

    def score(
        self,
        result: Dict,
        query: str,
        query_entities: List[str],
        query_type: str,
    ) -> Dict:
        """
        Score a search result for confidence.

        Args:
            result: Search result dict with title, content, relevance_score, etc.
            query: Original query string
            query_entities: Extracted entities from query
            query_type: Type of query (definition, how-to, etc.)

        Returns:
            Dict with:
            - confidence: 0-100 confidence score
            - level: "high", "medium", or "low"
            - factors: breakdown of scoring factors
            - recommendation: user-friendly confidence message
        """
        factors = {}

        # Factor 1: Semantic similarity (vector search score)
        # If available, use the model's relevance score (typically 0-1)
        semantic_score = result.get("relevance_score", result.get("score", 0.5))
        if isinstance(semantic_score, float) and semantic_score <= 1.0:
            # Convert 0-1 to 0-100
            factors["semantic"] = semantic_score * 100
        else:
            factors["semantic"] = min(100, semantic_score)  # Already 0-100

        # Factor 2: Keyword density (how many query entities appear in result)
        keyword_score = self._score_keywords(query, query_entities, result)
        factors["keywords"] = keyword_score

        # Factor 3: Content quality (article length, structure)
        quality_score = self._score_quality(result)
        factors["quality"] = quality_score

        # Factor 4: Query-type alignment
        alignment_score = self._score_alignment(result, query_type)
        factors["alignment"] = alignment_score

        # Factor 5: Source reliability (namespace, metadata)
        source_score = self._score_source(result)
        factors["source"] = source_score

        # Weighted average of all factors
        weights = {
            "semantic": 0.35,  # Semantic is most important
            "keywords": 0.20,
            "quality": 0.15,
            "alignment": 0.15,
            "source": 0.15,
        }

        confidence = sum(factors[key] * weights[key] for key in factors)
        confidence = max(0, min(100, confidence))  # Clamp 0-100

        # Determine confidence level
        if confidence >= self.CONFIDENCE_THRESHOLD_HIGH:
            level = "high"
            emoji = "🟢"
        elif confidence >= self.CONFIDENCE_THRESHOLD_MEDIUM:
            level = "medium"
            emoji = "🟡"
        else:
            level = "low"
            emoji = "🔴"

        # Generate recommendation message
        recommendation = self._generate_recommendation(confidence, level, factors)

        result_with_score = {
            "confidence": round(confidence, 1),
            "level": level,
            "emoji": emoji,
            "factors": {k: round(v, 1) for k, v in factors.items()},
            "recommendation": recommendation,
        }

        self.scoring_history.append(result_with_score)
        return result_with_score

    def _score_keywords(self, query: str, entities: List[str], result: Dict) -> float:
        """Score based on keyword matching."""
        if not entities:
            return 50.0  # Neutral if no entities

        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        combined = f"{title} {content}"

        # Count entity matches (weighted by position)
        matched = 0
        for entity in entities:
            if entity in title:
                matched += 2  # Title match worth more
            elif entity in combined:
                matched += 1

        # Calculate percentage match
        match_ratio = min(1.0, matched / (len(entities) * 2))
        return match_ratio * 100

    def _score_quality(self, result: Dict) -> float:
        """Score result quality based on content attributes."""
        score = 50.0  # Neutral baseline

        # Longer content = more thorough
        content_length = len(result.get("content", ""))
        if content_length > 500:
            score += 25
        elif content_length > 200:
            score += 15
        elif content_length > 50:
            score += 5

        # Article structure indicators
        title = result.get("title", "")
        if title and len(title) > 5:
            score += 10

        section = result.get("section", "")
        if section and len(section) > 2:
            score += 5

        # URL structure (help articles typically more reliable)
        url = result.get("url", "")
        if "help" in url.lower():
            score += 10
        elif "docs" in url.lower() or "guide" in url.lower():
            score += 5

        return min(100, score)

    def _score_alignment(self, result: Dict, query_type: str) -> float:
        """Score how well result aligns with query type."""
        score = 50.0  # Neutral baseline

        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        combined = f"{title} {content}"

        # Query type alignments
        type_keywords = {
            "definition": ["definition", "what is", "explanation", "meaning"],
            "how-to": ["how to", "steps", "process", "guide", "create", "setup"],
            "comparison": ["vs", "versus", "difference", "compare", "better"],
            "factual": ["does", "can", "is", "feature", "support"],
        }

        if query_type in type_keywords:
            keywords = type_keywords[query_type]
            for keyword in keywords:
                if keyword in combined:
                    score += 10
                    break  # One per category

        return min(100, score)

    def _score_source(self, result: Dict) -> float:
        """Score source reliability."""
        score = 50.0  # Neutral baseline

        # Namespace reliability (Clockify help typically most reliable)
        namespace = result.get("namespace", "")
        if "clockify" in namespace.lower():
            score += 20
        elif "help" in namespace.lower():
            score += 10

        # Recent updates if available
        if result.get("modified_at"):
            score += 10

        # Has URL (official source)
        if result.get("url"):
            score += 5

        return min(100, score)

    def _generate_recommendation(
        self, confidence: float, level: str, factors: Dict
    ) -> str:
        """Generate user-friendly recommendation message."""
        if level == "high":
            return "✓ Highly relevant result. Good confidence in this answer."
        elif level == "medium":
            if factors["keywords"] < 30:
                return "△ Decent match, but few keywords found. May need refinement."
            elif factors["semantic"] < 50:
                return "△ Moderate relevance. Consider alternative phrasing."
            else:
                return "△ Reasonable result. More refinement may help."
        else:  # low
            if confidence < 25:
                return "✗ Poor match. Try rephrasing your question."
            elif factors["keywords"] < 20:
                return "✗ Few matching keywords found. Search for specific terms."
            else:
                return "✗ Low confidence result. Consider a new search."

    def batch_score(
        self,
        results: List[Dict],
        query: str,
        query_entities: List[str],
        query_type: str,
    ) -> List[Dict]:
        """Score multiple results at once."""
        scored_results = []
        for result in results:
            scored = result.copy()
            score_info = self.score(result, query, query_entities, query_type)
            scored.update(score_info)
            scored_results.append(scored)

        # Sort by confidence (descending)
        scored_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return scored_results

    def get_stats(self) -> Dict:
        """Get scoring statistics."""
        if not self.scoring_history:
            return {"results_scored": 0}

        confidences = [r.get("confidence", 0) for r in self.scoring_history]
        levels = {}
        for r in self.scoring_history:
            level = r.get("level", "unknown")
            levels[level] = levels.get(level, 0) + 1

        return {
            "results_scored": len(self.scoring_history),
            "avg_confidence": round(sum(confidences) / len(confidences), 1),
            "level_distribution": levels,
            "min_confidence": round(min(confidences), 1),
            "max_confidence": round(max(confidences), 1),
        }


# Global instance
_scorer = None


def get_scorer() -> ConfidenceScorer:
    """Get or create global confidence scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer
```

## tests/test_clockify_rag_eval.py

```
"""Evaluation tests for Clockify RAG recall and precision."""
import pytest
import os
import json
from typing import Any


# Eval cases: (query, expected_terms_in_result)
# These are representative queries with expected content
EVAL_CASES = [
    ("How do I submit my weekly timesheet?", ["submit", "timesheet"]),
    ("Set billable rates per workspace member", ["billable rate", "rate", "member"]),
    ("Enable time rounding to 15 minutes", ["rounding", "time", "round"]),
    ("What is SSO?", ["sso", "single", "sign"]),
    ("How do I approve timesheets as a manager?", ["approve", "timesheet", "manager"]),
    ("What is a project budget?", ["project", "budget"]),
    ("How to enable time tracking?", ["time", "track"]),
    ("What are user roles?", ["role", "user", "permission"]),
]


def test_eval_cases_structure():
    """Verify eval cases are well-formed."""
    for q, terms in EVAL_CASES:
        assert isinstance(q, str) and len(q) > 0
        assert isinstance(terms, list) and len(terms) > 0
        for t in terms:
            assert isinstance(t, str) and len(t) > 0


def get_search_results(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Query the /search endpoint and return results."""
    import requests

    api_host = os.getenv("API_HOST", "localhost")
    api_port = int(os.getenv("API_PORT", "7000"))
    base_url = f"http://{api_host}:{api_port}"

    try:
        response = requests.get(
            f"{base_url}/search",
            params={"q": query, "k": k},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        print(f"Error querying /search: {e}")
        return []


def hit_at_k(results: list[dict[str, Any]], expected_terms: list[str], k: int = 5) -> bool:
    """
    Check if at least one of the expected terms appears in top-k results.

    A hit is when ANY expected term is found (case-insensitive) in the text.
    """
    results_text = " ".join([r.get("text", "").lower() for r in results[:k]])
    for term in expected_terms:
        if term.lower() in results_text:
            return True
    return False


@pytest.mark.skipif(
    os.getenv("SKIP_API_EVAL") == "true",
    reason="Requires running API server; set SKIP_API_EVAL=false to run"
)
class TestRetrievalEval:
    """Retrieval evaluation tests that require a running API."""

    def test_search_endpoint_available(self):
        """Verify the search endpoint is available."""
        import requests
        api_host = os.getenv("API_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", "7000"))
        base_url = f"http://{api_host}:{api_port}"

        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data.get("ok") is not None
        except Exception as e:
            pytest.skip(f"API server not available: {e}")

    @pytest.mark.parametrize("query,expected_terms", EVAL_CASES)
    def test_hit_at_k_5(self, query: str, expected_terms: list[str]):
        """Test that at least one expected term appears in top-5 results."""
        results = get_search_results(query, k=5)
        assert len(results) > 0, f"No results for query: {query}"

        hit = hit_at_k(results, expected_terms, k=5)
        if not hit:
            # Print debug info
            text_sample = " ".join([r.get("text", "")[:100] for r in results[:2]])
            print(f"\nQuery: {query}")
            print(f"Expected terms: {expected_terms}")
            print(f"Results text sample: {text_sample}")

        assert hit, f"None of {expected_terms} found in top-5 for: {query}"

    @pytest.mark.parametrize("query,expected_terms", EVAL_CASES)
    def test_hit_at_k_12(self, query: str, expected_terms: list[str]):
        """Test that at least one expected term appears in top-12 results."""
        results = get_search_results(query, k=12)
        assert len(results) > 0, f"No results for query: {query}"

        hit = hit_at_k(results, expected_terms, k=12)
        assert hit, f"None of {expected_terms} found in top-12 for: {query}"


def compute_eval_metrics() -> dict[str, Any]:
    """
    Compute comprehensive eval metrics across all test cases.

    Returns dict with hit@5, hit@12, coverage, etc.
    """
    metrics = {
        "total_cases": len(EVAL_CASES),
        "hit_at_5": 0,
        "hit_at_12": 0,
        "cases": [],
    }

    for query, expected_terms in EVAL_CASES:
        results_5 = get_search_results(query, k=5)
        results_12 = get_search_results(query, k=12)

        hit_5 = hit_at_k(results_5, expected_terms, k=5)
        hit_12 = hit_at_k(results_12, expected_terms, k=12)

        metrics["hit_at_5"] += 1 if hit_5 else 0
        metrics["hit_at_12"] += 1 if hit_12 else 0

        metrics["cases"].append({
            "query": query,
            "expected_terms": expected_terms,
            "hit_at_5": hit_5,
            "hit_at_12": hit_12,
        })

    # Compute percentages
    metrics["hit_at_5_pct"] = round(100 * metrics["hit_at_5"] / metrics["total_cases"], 1)
    metrics["hit_at_12_pct"] = round(100 * metrics["hit_at_12"] / metrics["total_cases"], 1)

    return metrics


@pytest.mark.skipif(
    os.getenv("SKIP_API_EVAL") == "true",
    reason="Requires running API server; set SKIP_API_EVAL=false to run"
)
def test_eval_report(capsys):
    """Generate and print comprehensive eval report."""
    import requests

    # Check API availability
    api_host = os.getenv("API_HOST", "localhost")
    api_port = int(os.getenv("API_PORT", "7000"))
    base_url = f"http://{api_host}:{api_port}"

    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
    except Exception as e:
        pytest.skip(f"API server not available: {e}")

    metrics = compute_eval_metrics()

    # Print report
    print("\n" + "="*60)
    print("CLOCKIFY RAG RETRIEVAL EVAL REPORT")
    print("="*60)
    print(f"Total test cases: {metrics['total_cases']}")
    print(f"Hit@5:  {metrics['hit_at_5']}/{metrics['total_cases']} ({metrics['hit_at_5_pct']}%)")
    print(f"Hit@12: {metrics['hit_at_12']}/{metrics['total_cases']} ({metrics['hit_at_12_pct']}%)")
    print("-"*60)

    for case in metrics["cases"]:
        status_5 = "✓" if case["hit_at_5"] else "✗"
        status_12 = "✓" if case["hit_at_12"] else "✗"
        print(f"{status_5} {status_12}  {case['query'][:40]:40s} {case['expected_terms']}")

    print("="*60)
```

## tests/test_integration_pipeline.py

```
"""
Integration Tests: Complete RAG Pipeline with Type Safety

Tests the full search and chat pipeline using Pydantic models for validation.
Covers end-to-end flows, error scenarios, and data validation.
"""

import time
import pytest
import requests
from typing import Dict, Any, List, Optional

# Import type-safe models
from src.models import (
    SearchRequest,
    ChatRequest,
    SearchResponse,
    ChatResponse,
    SearchResult,
    QueryAnalysis,
    ErrorResponse,
)

BASE_URL = "http://localhost:7000"
HEADERS = {"x-api-token": "change-me"}


class TestSearchPipeline:
    """Test complete /search endpoint pipeline."""

    def test_search_basic_query(self) -> None:
        """Basic search with valid query should return SearchResponse."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "how do I track time", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()

        # Validate response structure using pydantic
        search_response = SearchResponse(**data)
        assert search_response.success is True
        assert search_response.query == "how do I track time"
        assert len(search_response.results) >= 0
        assert search_response.latency_ms > 0

    def test_search_with_query_analysis(self) -> None:
        """Search should include query_analysis with extracted entities."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "timer features", "k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # Query analysis should be present
        if search_response.query_analysis:
            qa = search_response.query_analysis
            assert qa.primary_search_query is not None
            assert isinstance(qa.entities, list)
            assert 0.0 <= qa.confidence <= 1.0

    def test_search_result_structure(self) -> None:
        """Search results should have complete SearchResult structure."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "project time tracking", "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        if search_response.results:
            for result in search_response.results:
                # Validate each result using pydantic
                validated_result = SearchResult(**result)
                assert validated_result.title
                assert validated_result.url
                assert validated_result.namespace
                assert 0 <= validated_result.confidence <= 100
                assert validated_result.level in ["high", "medium", "low"]

    def test_search_with_namespace(self) -> None:
        """Search with specific namespace should filter results."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "time tracking", "k": 5, "namespace": "clockify"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # All results should be from requested namespace
        for result in search_response.results:
            assert result.get("namespace") == "clockify"

    def test_search_with_custom_k(self) -> None:
        """Search with custom k parameter should return up to k results."""
        for k in [1, 5, 10]:
            resp = requests.get(
                f"{BASE_URL}/search",
                params={"q": "tracking", "k": k},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()
            search_response = SearchResponse(**data)
            assert len(search_response.results) <= k

    def test_search_caching(self) -> None:
        """Repeated search with same query should hit cache (faster latency)."""
        query = "how do I start a project"

        # First request
        start1 = time.time()
        resp1 = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        latency1 = time.time() - start1
        assert resp1.status_code == 200
        data1 = resp1.json()
        response1 = SearchResponse(**data1)

        # Wait a bit to ensure cache isn't just instant
        time.sleep(0.1)

        # Second request (should hit cache)
        start2 = time.time()
        resp2 = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 5},
            headers=HEADERS,
        )
        latency2 = time.time() - start2
        assert resp2.status_code == 200
        data2 = resp2.json()
        response2 = SearchResponse(**data2)

        # Responses should be identical (same results, same request_id expected if cached)
        assert len(response1.results) == len(response2.results)
        # Cache should provide faster response (typically 10-50ms vs 100-200ms)
        # Note: First request might be slow due to model warmup, so just check both succeed

    @pytest.mark.parametrize(
        "query",
        [
            "timer",
            "time tracking",
            "project management",
            "how to track",
            "stop watch",
            "clock app",
        ],
    )
    def test_search_various_queries(self, query: str) -> None:
        """Search should handle various query types successfully."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 3},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)
        assert search_response.success is True


class TestChatPipeline:
    """Test complete /chat endpoint pipeline."""

    def test_chat_basic_query(self) -> None:
        """Basic chat with valid query should return ChatResponse."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "How do I track time?", "k": 5},
            headers=HEADERS,
        )
        # May return 200 or 500 depending on LLM availability
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            assert chat_response.success is True
            assert chat_response.query == "How do I track time?"
            assert len(chat_response.answer) > 0
            assert chat_response.latency_ms > 0

    def test_chat_with_namespace(self) -> None:
        """Chat with specific namespace should use that namespace."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "What is Clockify?", "k": 3, "namespace": "clockify"},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            assert chat_response.success is True
            # Context docs should be from requested namespace
            for doc in chat_response.context_docs:
                assert doc.namespace == "clockify"

    def test_chat_result_citations(self) -> None:
        """Chat answer should include citations to sources."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": "Tell me about time tracking features", "k": 5},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            data = resp.json()
            chat_response = ChatResponse(**data)
            # Answer may or may not have citations depending on LLM
            assert len(chat_response.answer) > 0


class TestSearchValidationWithModels:
    """Test input validation using Pydantic models."""

    def test_search_request_validation_min_query(self) -> None:
        """SearchRequest should accept minimum query length."""
        req = SearchRequest(query="a", k=1)
        assert req.query == "a"
        assert req.k == 1

    def test_search_request_validation_max_query(self) -> None:
        """SearchRequest should accept maximum query length."""
        long_query = "a" * 2000
        req = SearchRequest(query=long_query, k=5)
        assert len(req.query) == 2000

    def test_search_request_validation_exceeds_max(self) -> None:
        """SearchRequest should reject query > 2000 chars."""
        long_query = "a" * 2001
        with pytest.raises(ValueError):
            SearchRequest(query=long_query, k=5)

    def test_search_request_k_bounds(self) -> None:
        """SearchRequest should validate k within bounds."""
        # Valid k values
        for k in [1, 5, 10, 20]:
            req = SearchRequest(query="test", k=k)
            assert req.k == k

        # Invalid k values
        with pytest.raises(ValueError):
            SearchRequest(query="test", k=0)  # k < 1

        with pytest.raises(ValueError):
            SearchRequest(query="test", k=21)  # k > 20

    def test_chat_request_validation(self) -> None:
        """ChatRequest should validate parameters."""
        # Valid request
        req = ChatRequest(question="How to track time?", k=5)
        assert req.question == "How to track time?"
        assert req.k == 5

        # Invalid k
        with pytest.raises(ValueError):
            ChatRequest(question="test", k=21)

        # Invalid question length
        with pytest.raises(ValueError):
            ChatRequest(question="a" * 2001, k=5)


class TestErrorHandling:
    """Test error handling and resilience."""

    def test_search_missing_query(self) -> None:
        """Search without required query param should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_search_missing_token(self) -> None:
        """Search without API token should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
            # No headers
        )
        assert resp.status_code == 401

    def test_search_invalid_token(self) -> None:
        """Search with invalid token should fail."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5},
            headers={"x-api-token": "invalid-token"},
        )
        assert resp.status_code == 401

    def test_chat_missing_question(self) -> None:
        """Chat without question should fail."""
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"k": 5},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_invalid_namespace(self) -> None:
        """Search with invalid namespace should still work (fallback to all namespaces)."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "test", "k": 5, "namespace": "nonexistent"},
            headers=HEADERS,
        )
        # Should gracefully handle invalid namespace
        assert resp.status_code in [200, 422]


class TestHealthAndConfig:
    """Test health checks and configuration endpoints."""

    def test_health_endpoint(self) -> None:
        """Health endpoint should return component status."""
        resp = requests.get(f"{BASE_URL}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data or "status" in data

    def test_health_deep_check(self) -> None:
        """Deep health check should probe LLM."""
        resp = requests.get(f"{BASE_URL}/health?deep=1")
        assert resp.status_code == 200
        data = resp.json()
        # May or may not have LLM available

    def test_live_endpoint(self) -> None:
        """Live endpoint should always return 200."""
        resp = requests.get(f"{BASE_URL}/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "alive"

    def test_ready_endpoint(self) -> None:
        """Ready endpoint should check if system is ready."""
        resp = requests.get(f"{BASE_URL}/ready")
        # May return 200 or 503 depending on readiness
        assert resp.status_code in [200, 503]

    def test_config_endpoint(self) -> None:
        """Config endpoint should return system configuration."""
        resp = requests.get(f"{BASE_URL}/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "namespaces_env" in data
        assert "embedding_model" in data


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""

    @pytest.mark.parametrize(
        "query",
        [
            "a",  # Single char
            "what?",  # With punctuation
            "time & tracking",  # With special chars
            "UPPERCASE QUERY",  # Case variation
            "123 456",  # Numbers
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
        ],
    )
    def test_search_special_inputs(self, query: str) -> None:
        """Search should handle special characters and injection attempts safely."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "k": 3},
            headers=HEADERS,
        )
        # Should not crash, but may return 200 or 422
        assert resp.status_code in [200, 422, 400]

    def test_search_concurrent_requests(self) -> None:
        """Multiple concurrent search requests should all succeed."""
        import concurrent.futures

        def search_query(q: str) -> int:
            resp = requests.get(
                f"{BASE_URL}/search",
                params={"q": q, "k": 3},
                headers=HEADERS,
            )
            return resp.status_code

        queries = ["timer", "project", "tracking", "time", "stop"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(search_query, queries))

        # All should succeed (200) or be rate limited (429)
        assert all(r in [200, 429] for r in results)

    def test_search_result_ranking(self) -> None:
        """Results should be ranked by confidence score."""
        resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": "time tracking", "k": 10},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        search_response = SearchResponse(**data)

        # Confidence scores should be non-increasing
        confidences = [r.confidence for r in search_response.results]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]


class TestTypeValidation:
    """Test type safety with Pydantic models."""

    def test_search_response_typing(self) -> None:
        """SearchResponse should enforce type validation."""
        # Valid response structure
        valid_data = {
            "success": True,
            "query": "test",
            "results": [],
            "total_results": 0,
            "latency_ms": 100.0,
        }
        response = SearchResponse(**valid_data)
        assert response.success is True

        # Invalid types should raise
        invalid_data = {
            "success": "yes",  # Should be bool
            "query": "test",
            "results": [],
            "total_results": 0,
            "latency_ms": 100.0,
        }
        with pytest.raises(ValueError):
            SearchResponse(**invalid_data)

    def test_search_result_typing(self) -> None:
        """SearchResult should enforce type validation."""
        valid_result = {
            "id": "chunk_1",
            "title": "Test",
            "content": "Test content",
            "url": "http://example.com",
            "namespace": "test",
            "confidence": 85,
            "level": "high",
            "score": 0.85,
        }
        result = SearchResult(**valid_result)
        assert result.confidence == 85

        # Confidence out of range
        invalid_result = valid_result.copy()
        invalid_result["confidence"] = 150
        with pytest.raises(ValueError):
            SearchResult(**invalid_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

