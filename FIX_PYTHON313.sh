#!/bin/bash
# Fix Python 3.13 compatibility issue
# This script switches to Python 3.12 and sets up the RAG system correctly

set -e

echo "🔧 Python 3.13 Compatibility Fix"
echo "================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Not in RAG directory"
    echo "   Please run this from the rag/ directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 not found. Installing via Homebrew..."
    echo ""

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install it first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    echo "📦 Installing Python 3.12..."
    brew install python@3.12
    echo ""
fi

# Verify Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Error: Python 3.12 installation failed"
    echo "   Please install manually: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3.12 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Remove old venv
if [ -d ".venv" ]; then
    echo "🗑️  Removing old Python 3.13 virtual environment..."
    rm -rf .venv
    echo "✓ Removed .venv"
    echo ""
fi

# Create new venv with Python 3.12
echo "🐍 Creating new virtual environment with Python 3.12..."
python3.12 -m venv .venv
echo "✓ Created .venv"
echo ""

# Activate venv
echo "✓ Activating virtual environment..."
source .venv/bin/activate
echo ""

# Verify we're using the right Python
ACTIVE_PYTHON=$(python --version)
echo "✓ Active Python: $ACTIVE_PYTHON"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

echo "🚀 Ready to install dependencies!"
echo ""
echo "Next steps:"
echo "  1. Keep this terminal open"
echo "  2. Run: ./scripts/run_local.sh"
echo ""
echo "This will:"
echo "  - Install all dependencies with smart fallbacks"
echo "  - Build the search index"
echo "  - Start the RAG server on port 7001"
echo ""
echo "═══════════════════════════════════════════"
echo "Environment fixed! You can now proceed."
echo "═══════════════════════════════════════════"
echo ""

# Optionally run the setup automatically
read -p "Do you want to run ./scripts/run_local.sh now? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Launching RAG setup..."
    echo ""
    ./scripts/run_local.sh
fi
