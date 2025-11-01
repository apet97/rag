#!/bin/bash
# Quick launch script for VPN-enabled company device
# Usage: ./launch.sh

set -e

echo "🚀 RAG Quick Launch"
echo "==================="
echo ""

# Navigate to repo
cd "$(dirname "$0")"

# Launch via existing script
echo "Starting RAG server..."
./scripts/run_local.sh

