#!/bin/bash
# Fix embedding model configuration in .env

set -e

echo "🔧 Fixing embedding model configuration..."

# Backup current .env
cp .env .env.backup_$(date +%s)

# Fix EMBEDDING_MODEL
sed -i '' 's|EMBEDDING_MODEL=.*|EMBEDDING_MODEL=intfloat/multilingual-e5-base|' .env

# Fix EMBEDDING_DIM
sed -i '' 's/EMBEDDING_DIM=.*/EMBEDDING_DIM=768/' .env

echo ""
echo "✅ Configuration updated:"
grep "EMBEDDING_MODEL\|EMBEDDING_DIM" .env

echo ""
echo "📝 Changes made:"
echo "  - EMBEDDING_MODEL: intfloat/multilingual-e5-base"
echo "  - EMBEDDING_DIM: 768"
echo ""
echo "🚀 Ready to start server with: uvicorn src.server:app --host 0.0.0.0 --port 7001"
