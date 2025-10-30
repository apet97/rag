# Clockify RAG - Quick Start (3 Commands)

## Production-Ready Out of the Box ✅

**No configuration needed. Just run these 3 commands:**

---

## Step 1: Clone Repository
```bash
git clone <your-repo-url> clockify-rag
cd clockify-rag
```

---

## Step 2: Build Knowledge Base
```bash
make ingest
```
*Takes ~5 minutes. Builds FAISS indexes for Clockify + LangChain docs.*

---

## Step 3: Start Server
```bash
make serve
```
*Server starts on port 7001 with production settings.*

---

## ✅ That's It!

**Your RAG system is now running:**
- **URL:** http://localhost:7001
- **API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`
- **LLM:** Internal gpt-oss:20b (10.127.0.192:11434)

---

## Quick Test

### Health Check
```bash
curl http://localhost:7001/health | python3 -m json.tool
```

### Search Query
```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=how%20to%20track%20time&k=5'
```

### Chat with Citations
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I create a project?", "k": 5}'
```

---

## Production Features (Pre-Configured)

✅ Port 7001 (production)
✅ Secure token authentication
✅ Internal LLM (no API key needed)
✅ Harmony format (gpt-oss:20b optimized)
✅ Hybrid search (BM25 + Vector)
✅ Cross-encoder reranking
✅ Circuit breakers & fault tolerance
✅ Semantic caching (10K queries)
✅ Rate limiting & CORS

**Everything works out of the box!** 🚀

---

For detailed documentation, see: [DEPLOY.md](DEPLOY.md)
