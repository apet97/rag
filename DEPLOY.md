# Clockify RAG - Production Deployment Guide

## 🚀 Out-of-the-Box Deployment (3 Steps)

This system is **production-ready** with no configuration required.

---

## **Step 1: Clone Repository**

```bash
git clone <your-repo-url> clockify-rag
cd clockify-rag
```

**Verify you're on the latest version:**
```bash
git log --oneline -1
# Should show: 470bcd88 feat: Implement Harmony chat format support
```

---

## **Step 2: Build Knowledge Base**

```bash
make ingest
```

**What this does:**
- Processes Clockify + LangChain documentation
- Generates embeddings (768-dim E5 model)
- Builds FAISS vector indexes
- Creates BM25 lexical indexes

**Time:** ~5 minutes
**Output:** `index/faiss/clockify/` and `index/faiss/langchain/`

---

## **Step 3: Start Server**

```bash
make serve
```

**Server Details:**
- **URL:** http://localhost:7001
- **Port:** 7001 (configured in .env)
- **API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`
- **LLM:** Internal gpt-oss:20b at 10.127.0.192:11434 (no API key)

**That's it!** Server is running in production mode.

---

## **Step 4: Verify Deployment**

### Health Check
```bash
curl http://localhost:7001/health | python3 -m json.tool
```

**Expected Response:**
```json
{
  "ok": true,
  "namespaces": ["clockify", "langchain"],
  "embedding_ok": true,
  "llm_ok": true,
  "llm_model": "gpt-oss:20b",
  "harmony_enabled": true
}
```

### Test Search
```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=how%20to%20track%20time&k=5' \
  | python3 -m json.tool
```

### Test Chat (Full RAG with Citations)
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I create a project?", "k": 5}' \
  | python3 -m json.tool
```

---

## **Configuration Details**

All settings are production-ready in `.env`:

| Setting | Value | Description |
|---------|-------|-------------|
| `ENV` | `prod` | Production environment |
| `API_PORT` | `7001` | Server port |
| `API_TOKEN` | Pre-configured | Secure token (works out of box) |
| `LLM_BASE_URL` | `http://10.127.0.192:11434` | Internal LLM (no API key) |
| `LLM_MODEL` | `gpt-oss:20b` | Harmony-optimized model |
| `LLM_USE_HARMONY` | `auto` | Auto-detects gpt-oss models |
| `HYBRID_SEARCH` | `true` | BM25 + Vector fusion |
| `RERANK_DISABLED` | `false` | Cross-encoder reranking enabled |

**No changes required** - everything works immediately.

---

## **Optional: Web UI**

To use the demo interface:

```bash
# In a new terminal
make ui
```

Then open: **http://localhost:8080**

---

## **Production Features Enabled**

✅ **Security**
- Token authentication (HMAC constant-time comparison)
- Rate limiting (100ms per IP)
- CORS with explicit origins
- ENV=prod enforcement

✅ **Performance**
- Hybrid search (BM25 + Vector with RRF fusion)
- Cross-encoder reranking
- Semantic caching (10K query cache, 1h TTL)
- Circuit breakers (fault tolerance)

✅ **Quality**
- Harmony chat format (gpt-oss:20b optimal performance)
- Inline citations [1], [2] with source URLs
- Query decomposition for complex questions
- MMR diversity penalty

✅ **Observability**
- Prometheus metrics: `/metrics`
- Performance tracking: `/perf?detailed=true`
- Health endpoint: `/health`
- Structured logging (INFO level)

---

## **API Endpoints**

### Core Endpoints
```bash
GET  /health              # Health check
GET  /search              # Retrieval only (no LLM)
POST /chat                # Full RAG with citations
POST /chat/stream         # Streaming RAG (if enabled)
GET  /metrics             # Prometheus metrics
GET  /perf                # Performance stats
```

### Example API Call
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is kiosk mode?",
    "namespace": "clockify",
    "k": 5,
    "allow_rewrites": true,
    "allow_rerank": true
  }'
```

---

## **Troubleshooting**

### "Cannot reach LLM at 10.127.0.192:11434"
**Solution:** Ensure you're on the VPN
```bash
ping 10.127.0.192
```

### "Index not found"
**Solution:** Build the index first
```bash
make ingest
```

### "Module not found" errors
**Solution:** Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Server not responding on port 7001
**Solution:** Check if port is in use
```bash
lsof -i :7001
# If occupied, kill process or change API_PORT in .env
```

---

## **Maintenance**

### Rebuild Index (When Docs Change)
```bash
make ingest
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### View Logs
```bash
tail -f logs/api.log
```

### Run Evaluation
```bash
make eval          # Full evaluation harness
make eval-axioms   # AXIOM 1-9 compliance check
```

---

## **Architecture Overview**

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────┐
│  FastAPI (Port 7001)        │
│  - Token auth               │
│  - Rate limiting            │
│  - CORS enforcement         │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Retrieval Engine           │
│  - Hybrid (BM25 + Vector)   │
│  - Query expansion          │
│  - Cross-encoder reranking  │
│  - MMR diversity            │
└──────┬──────────────────────┘
       │
       ├─► FAISS (768-dim E5 embeddings)
       ├─► BM25 (lexical search)
       └─► Cache (semantic LRU)
       │
       ▼
┌─────────────────────────────┐
│  RAGPrompt (Harmony)        │
│  - Context building         │
│  - Citation injection       │
│  - Developer role           │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  LLM (gpt-oss:20b)          │
│  - 10.127.0.192:11434       │
│  - Harmony chat format      │
│  - Circuit breaker          │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Answer + Citations [1]     │
│  + Source URLs              │
└─────────────────────────────┘
```

---

## **Quick Reference**

| Command | Purpose |
|---------|---------|
| `make ingest` | Build knowledge base indexes |
| `make serve` | Start production server (port 7001) |
| `make ui` | Start demo web interface |
| `make eval` | Run evaluation harness |
| `curl http://localhost:7001/health` | Health check |
| `curl http://localhost:7001/metrics` | Prometheus metrics |

**API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`

---

## **Summary**

✅ **Production-ready out of the box**
✅ **No configuration required**
✅ **3-step deployment: clone → ingest → serve**
✅ **Port 7001 with secure token authentication**
✅ **Internal LLM (no API key needed)**
✅ **Harmony format for optimal gpt-oss:20b performance**

**You're ready to deploy!** 🚀
