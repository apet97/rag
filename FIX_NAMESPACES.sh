#!/bin/bash
# Fix namespace mismatch in .env file
# Updates NAMESPACES to match actual index directory (clockify_url)

set -e

echo "🔧 Namespace Configuration Fix"
echo "=============================="
echo ""

# Check if we're in the right directory
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "   Please run this from the rag/ directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo ""

# Check current namespace config
CURRENT_NS=$(grep "^NAMESPACES=" .env | cut -d= -f2)
echo "Current NAMESPACES: $CURRENT_NS"
echo ""

# Check what index directories actually exist
echo "Checking index directories..."
if [ -d "index/faiss" ]; then
    echo "Found index directories:"
    ls -1 index/faiss/ | grep -v "^\." | sed 's/^/  - /'
    echo ""
else
    echo "❌ Error: index/faiss directory not found"
    echo "   Run: python3 tools/ingest_v2.py"
    exit 1
fi

# Fix the namespace
if [ "$CURRENT_NS" != "clockify_url" ]; then
    echo "🔧 Updating NAMESPACES to 'clockify_url'..."

    # Backup current .env
    cp .env .env.backup
    echo "✓ Backed up .env to .env.backup"

    # Update NAMESPACES line
    sed -i '' 's/^NAMESPACES=.*/NAMESPACES=clockify_url/' .env

    NEW_NS=$(grep "^NAMESPACES=" .env | cut -d= -f2)
    echo "✓ Updated NAMESPACES: $NEW_NS"
    echo ""
else
    echo "✓ NAMESPACES already correct: $CURRENT_NS"
    echo ""
fi

echo "═══════════════════════════════════════════"
echo "Namespace configuration fixed!"
echo "═══════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Run: python test_rag.py"
echo "  2. Or start server: uvicorn src.server:app --host 0.0.0.0 --port 7001"
echo ""
