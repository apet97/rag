# Clockify RAG - Complete Setup Guide (From Step 0)

## 🎯 **What You'll Get**

A production-ready RAG system that:
- Runs on port 7001
- Uses internal LLM (no API key needed)
- Includes web UI
- Works out of the box (no configuration)

---

# 📋 **Prerequisites**

Before starting, ensure you have:

### **1. System Requirements**
- **OS:** macOS, Linux, or Windows (with WSL2)
- **Python:** 3.11, 3.12, or 3.13 (3.12 recommended)
- **Git:** 2.x or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Disk:** 5GB free space

### **2. Network Requirements**
- **VPN:** Connected to corporate network
- **LLM Access:** Connectivity to 10.127.0.192:11434

### **3. Verify Prerequisites**

```bash
# Check Python version
python3 --version
# Should show: Python 3.11.x, 3.12.x, or 3.13.x

# Check Git
git --version
# Should show: git version 2.x.x

# Check VPN connectivity
ping -c 3 10.127.0.192
# Should show: 3 packets transmitted, 3 packets received
```

---

# 🚀 **Step-by-Step Setup**

## **Step 0: Clone Repository**

```bash
# Clone from GitHub
git clone git@github.com:apet97/rag.git clockify-rag

# Navigate into directory
cd clockify-rag

# Verify you're on the correct branch
git branch
# Should show: * main

# Verify latest commit
git log --oneline -1
# Should show: Production-ready configuration
```

**Expected:**
```
Cloning into 'clockify-rag'...
remote: Enumerating objects: X, done.
remote: Counting objects: 100% (X/X), done.
✅ Repository cloned successfully
```

---

## **Step 1: Environment Setup**

### **1A. Create Python Virtual Environment**

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# OR on Windows WSL2
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
which python3
# Should show: /path/to/clockify-rag/.venv/bin/python3
```

**Expected output:**
```bash
(.venv) user@machine:~/clockify-rag$
```

### **1B. Install Dependencies**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This installs ~50 packages including:
# - fastapi, uvicorn (API server)
# - sentence-transformers (embeddings)
# - faiss-cpu (vector search)
# - openai-harmony (gpt-oss:20b optimization)
```

**Expected:**
```
Collecting fastapi>=0.115.0
...
Successfully installed fastapi-0.115.0 uvicorn-0.30.0 [and 48 more packages]
✅ Dependencies installed
```

**Time:** ~2-3 minutes

### **1C. Verify Installation**

```bash
# Test Python imports
python3 << 'EOF'
import fastapi
import sentence_transformers
import faiss
print("✅ All dependencies installed correctly")
EOF
```

**Expected:**
```
✅ All dependencies installed correctly
```

---

## **Step 2: Build Knowledge Base**

This step processes documentation, generates embeddings, and builds vector indexes.

```bash
make ingest
```

**What happens:**
1. Scrapes Clockify Help (https://clockify.me/help/)
2. Scrapes LangChain docs (optional)
3. Cleans HTML → Markdown
4. Chunks documents (parent-child strategy)
5. Generates 768-dim embeddings (E5 model)
6. Builds FAISS vector indexes
7. Builds BM25 lexical indexes

**Expected output:**
```
Running Clockify Help ingestion (process -> chunk -> embed)...
📥 Loading documentation from data/clockify/
✓ Loaded 1,234 documents
🔪 Chunking into parent-child nodes...
✓ Generated 3,456 chunks
🧠 Generating embeddings (intfloat/multilingual-e5-base)...
✓ Created 3,456 embeddings (768-dim)
💾 Building FAISS index...
✓ FAISS index built: index/faiss/clockify/index.faiss
✓ Metadata saved: index/faiss/clockify/meta.json
✨ Ingestion complete!
```

**Time:** ~5-7 minutes

**Troubleshooting:**
```bash
# If you see "Module not found" errors:
pip install -r requirements.txt

# If you see "Permission denied":
chmod +x scripts/*.py

# If FAISS build fails:
pip install faiss-cpu --upgrade
```

---

## **Step 3: Start RAG Server**

```bash
make serve
```

**What happens:**
- Starts FastAPI server on port 7001
- Loads FAISS indexes into memory
- Initializes LLM client (connects to 10.127.0.192:11434)
- Enables Harmony format for gpt-oss:20b
- Activates security (token auth, rate limiting)

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
🚀 LLM Client initialized in PRODUCTION_MODE (ollama) [Harmony enabled]
   Endpoint: http://10.127.0.192:11434/api/chat
   Model: gpt-oss:20b
✓ Loaded 2 namespaces: ['clockify', 'langchain']
✓ Cached 3,456 vectors for namespace 'clockify'
✓ Cached 2,100 vectors for namespace 'langchain'
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7001 (Press CTRL+C to quit)
```

**Server is now running!** ✅

**Keep this terminal open.** Open a new terminal for testing.

---

## **Step 4: Verify Deployment**

Open a **new terminal** (keep server running).

### **4A. Health Check**

```bash
curl http://localhost:7001/health | python3 -m json.tool
```

**Expected:**
```json
{
  "ok": true,
  "namespaces": ["clockify", "langchain"],
  "index_ok": true,
  "embedding_ok": true,
  "llm_ok": true,
  "llm_model": "gpt-oss:20b",
  "harmony_enabled": true,
  "mode": "production",
  "cache_hit_rate_pct": 0.0,
  "indexed_vectors": 5556
}
```

### **4B. Test Search (Retrieval Only)**

```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=how%20to%20track%20time&k=5' \
  | python3 -m json.tool
```

**Expected:**
```json
{
  "results": [
    {
      "rank": 1,
      "title": "Time Tracking > Getting Started",
      "url": "https://clockify.me/help/time-tracking/...",
      "score": 0.89,
      "text": "To track time in Clockify..."
    }
  ],
  "count": 5,
  "latency_ms": 45
}
```

### **4C. Test Chat (Full RAG with Citations)**

```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How do I create a project?",
    "k": 5
  }' | python3 -m json.tool
```

**Expected:**
```json
{
  "answer": "To create a project in Clockify [1]:\n1. Go to Projects\n2. Click 'Create Project'\n3. Enter project name...",
  "sources": [
    {
      "title": "Projects > Creating Projects",
      "url": "https://clockify.me/help/projects/creating-projects",
      "score": 0.92
    }
  ],
  "latency_ms": {
    "retrieval": 45,
    "llm": 1200,
    "total": 1245
  },
  "citations_found": 3
}
```

✅ **All systems operational!**

---

## **Step 5: Start Web UI**

Open a **third terminal** (keep server running).

```bash
# Navigate to project directory
cd clockify-rag

# Activate virtual environment
source .venv/bin/activate

# Start UI server
make ui
```

**Expected output:**
```
Starting demo UI on http://localhost:8080...
Press Ctrl+C to stop
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

**UI is now running!** ✅

### **5A. Open UI in Browser**

```bash
# Open browser to:
http://localhost:8080
```

**What you'll see:**
- **Search Tab:** Test retrieval with relevance scores
- **Chat Tab:** Full RAG with inline citations [1], [2]
- **Config Panel:** System status (click to expand)

**UI Features:**
- Default API: `http://localhost:7001` ✅
- Default Token: `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0` ✅
- Default k: 5 (adjustable)

**No configuration needed** - works immediately!

### **5B. Test UI**

1. **Search Tab:**
   - Enter: "how to track time"
   - Click "Search"
   - See top 5 results with relevance scores

2. **Chat Tab:**
   - Enter: "How do I create a project?"
   - Click "Ask" (or Ctrl+Enter)
   - See answer with inline citations [1], [2]
   - See source URLs below answer

3. **Config Panel:**
   - Click "▼ Configuration" to expand
   - See system health status
   - See loaded namespaces, LLM model, cache stats

---

# 📊 **Architecture Overview**

```
┌─────────────────┐
│  Browser UI     │ ← http://localhost:8080
│  (port 8080)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  FastAPI Server             │ ← http://localhost:7001
│  (port 7001)                │
│  - Token auth               │
│  - Rate limiting            │
│  - CORS                     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Retrieval Engine           │
│  - Hybrid (BM25 + Vector)   │
│  - Query expansion          │
│  - Reranking                │
│  - MMR diversity            │
└────────┬────────────────────┘
         │
         ├─► FAISS Index (768-dim E5)
         ├─► BM25 Index (lexical)
         └─► Semantic Cache (10K)
         │
         ▼
┌─────────────────────────────┐
│  LLM (gpt-oss:20b)          │
│  10.127.0.192:11434         │
│  - Harmony format           │
│  - Circuit breaker          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Answer + Citations [1]     │
│  + Source URLs              │
└─────────────────────────────┘
```

---

# 🔧 **Configuration Reference**

All settings are in `.env` (production-ready, no changes needed):

| Setting | Value | Description |
|---------|-------|-------------|
| `ENV` | `prod` | Production environment |
| `API_PORT` | `7001` | Server port |
| `API_TOKEN` | `05yBp...` | Secure authentication token |
| `LLM_BASE_URL` | `http://10.127.0.192:11434` | Internal LLM (no API key) |
| `LLM_MODEL` | `gpt-oss:20b` | Model name |
| `LLM_USE_HARMONY` | `auto` | Auto-detect Harmony format |
| `HYBRID_SEARCH` | `true` | BM25 + Vector fusion |
| `RERANK_DISABLED` | `false` | Cross-encoder enabled |
| `CACHE_SIZE` | `1000` | Response cache size |

**Note:** `.env` file is not committed to Git (in .gitignore).

---

# 🧪 **Quality Assurance Checklist**

Run these checks to ensure everything is working:

### **1. Syntax Validation**
```bash
python3 -m py_compile src/server.py src/embeddings.py src/prompt.py
echo "✅ Syntax valid"
```

### **2. Import Check**
```bash
python3 << 'EOF'
from src.llm.harmony_encoder import HarmonyEncoder
from src.prompt import RAGPrompt
from src.server import app
print("✅ All imports successful")
EOF
```

### **3. Health Endpoint**
```bash
curl http://localhost:7001/health | grep -q '"ok": true' && echo "✅ Health OK"
```

### **4. Search Endpoint**
```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=test&k=3' \
  | grep -q '"results"' && echo "✅ Search OK"
```

### **5. Chat Endpoint**
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{"question": "test", "k": 3}' \
  | grep -q '"answer"' && echo "✅ Chat OK"
```

### **6. UI Accessibility**
```bash
curl -s http://localhost:8080 | grep -q "Clockify Help RAG" && echo "✅ UI OK"
```

---

# ❌ **Troubleshooting**

## **Problem: "Cannot connect to LLM"**

**Symptoms:**
```
❌ LLM returned status 500
Cannot connect to http://10.127.0.192:11434
```

**Solution:**
```bash
# 1. Check VPN connection
ping 10.127.0.192
# If fails: Connect to VPN

# 2. Test LLM endpoint
curl http://10.127.0.192:11434/api/tags
# Should return: {"models": [...]}

# 3. If still failing, use mock mode for testing
export MOCK_LLM=true
make serve
```

---

## **Problem: "Index not found"**

**Symptoms:**
```
RuntimeError: Index for namespace 'clockify' not found
```

**Solution:**
```bash
# Rebuild index
make ingest

# Verify index exists
ls -lh index/faiss/clockify/
# Should show: index.faiss, meta.json
```

---

## **Problem: "Module not found: openai_harmony"**

**Symptoms:**
```
ImportError: No module named 'openai_harmony'
```

**Solution:**
```bash
# Install Harmony support
pip install openai-harmony

# Restart server
make serve
```

---

## **Problem: "Port 7001 already in use"**

**Symptoms:**
```
ERROR: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port 7001
lsof -i :7001

# Kill the process
kill -9 <PID>

# Or change port in .env
export API_PORT=7002
make serve
```

---

## **Problem: "Permission denied" during ingestion**

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'data/clockify'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 data/
chmod -R 755 index/

# Retry
make ingest
```

---

# 📚 **Additional Resources**

- **Quick Start:** `QUICKSTART.md` - 3-command deployment
- **Production Guide:** `DEPLOY.md` - Detailed production setup
- **Main README:** `README.md` - Architecture & features
- **API Documentation:** http://localhost:7001/docs (when server is running)

---

# 🎓 **Next Steps**

**You've successfully deployed the RAG system!** Here's what to do next:

1. **Try Custom Queries:**
   - Test with domain-specific questions
   - Verify citation accuracy
   - Check source URLs

2. **Monitor Performance:**
   ```bash
   curl http://localhost:7001/metrics  # Prometheus metrics
   curl http://localhost:7001/perf?detailed=true  # Performance stats
   ```

3. **Run Evaluation:**
   ```bash
   make eval  # Full evaluation harness
   ```

4. **Explore Advanced Features:**
   - Query decomposition for complex questions
   - Hybrid search (BM25 + Vector)
   - Cross-encoder reranking
   - Semantic caching

---

# ✅ **Summary: You're Done!**

**What you have:**
✅ RAG server running on port 7001
✅ Web UI running on port 8080
✅ Internal LLM connected (10.127.0.192:11434)
✅ Harmony format enabled (optimal gpt-oss:20b performance)
✅ Hybrid search (BM25 + Vector)
✅ Cross-encoder reranking
✅ Production security (token auth, rate limiting)
✅ Semantic caching (10K queries, 1h TTL)

**Total setup time:** ~10-15 minutes

**Commands to remember:**
```bash
make ingest  # Rebuild knowledge base
make serve   # Start RAG server
make ui      # Start web UI
make eval    # Run evaluation
```

🚀 **Your RAG system is production-ready!**
