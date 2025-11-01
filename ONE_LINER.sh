#!/bin/bash
# Ultimate one-liner setup for RAG on VPN device
# Usage: curl -sSL https://raw.githubusercontent.com/apet97/rag/main/ONE_LINER.sh | bash
# OR: ./ONE_LINER.sh

set -e

echo "🚀 RAG One-Liner Setup"
echo "======================"
echo ""

# Detect current directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Not in RAG directory. Cloning repository..."
    cd ~
    if [ -d "rag" ]; then
        echo "⚠️  rag/ directory already exists. Using it."
        cd rag
        git pull origin main 2>/dev/null || echo "  (Using existing files)"
    else
        git clone https://github.com/apet97/rag.git
        cd rag
    fi
fi

echo "📁 Working directory: $(pwd)"
echo ""

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
echo "✓ Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt -c constraints.txt

# Create .env if missing
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env configuration..."
    cp .env.example .env
fi

# Build index if missing
if [ ! -f "index/faiss/clockify_url/meta.json" ]; then
    echo "🔨 Building search index (this takes ~2 minutes)..."
    make ingest_v2
else
    echo "✓ Index already exists, skipping build"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Starting RAG server on port 7001..."
echo "   Access UI: http://localhost:7001"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

# Launch server
uvicorn src.server:app --host 0.0.0.0 --port 7001
