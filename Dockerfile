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
