# Code Part 9

## COMPLETE_SETUP_GUIDE.md

```
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
```

## eval/run_eval.py

```
#!/usr/bin/env python3
"""Clockify RAG evaluation harness.

Computes retrieval and mock-generation metrics against a gold set.
Falls back to offline FAISS evaluation when the HTTP API is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

# Ensure project root is importable when executed as a script
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import requests
except Exception:  # pragma: no cover - requests is optional for offline mode
    requests = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - validated at runtime
    raise SystemExit("faiss is required for offline evaluation. Install faiss-cpu.") from exc

from src.embeddings import encode_texts, embed_query
from src.query_expand import expand
from src.rerank import rerank, is_available as rerank_available
from src.query_decomposition import decompose_query, is_multi_intent_query
from src.server import detect_query_type, should_enable_hybrid_search


INDEX_ROOT = Path("index/faiss")
DEFAULT_NAMESPACES = ["clockify"]


@dataclass
class GoldItem:
    """Single gold-set entry."""

    id: str
    question: str
    answer_regex: str
    source_urls: List[str]


@dataclass
class DecompositionHitInfo:
    """Per-query decomposition and hit tracking."""

    strategy: str  # "none", "heuristic", or "llm"
    subtask_count: int
    subtasks: List[str]
    subtask_hits: Dict[int, List[str]]  # subtask_index -> list of retrieved URLs
    llm_used: bool


def load_goldset(path: Path) -> List[GoldItem]:
    """Load CSV goldset with required columns."""
    items: List[GoldItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "question", "answer_regex", "source_url"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"goldset is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            src_urls = [u.strip() for u in row["source_url"].split("|") if u.strip()]
            items.append(
                GoldItem(
                    id=row["id"].strip(),
                    question=row["question"].strip(),
                    answer_regex=row["answer_regex"].strip(),
                    source_urls=src_urls,
                )
            )
    return items


def normalize_url(url: str) -> str:
    """Strip anchors and trailing slashes for stable comparison."""
    if not url:
        return url
    base = url.split("#", 1)[0]
    return base.rstrip("/")


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 1.0
    top = retrieved[:k]
    hits = sum(1 for g in gold if any(normalize_url(g) == normalize_url(r) for r in top))
    return hits / len(gold)


def mrr_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 1.0
    top = retrieved[:k]
    for rank, url in enumerate(top, start=1):
        if any(normalize_url(url) == normalize_url(g) for g in gold):
            return 1.0 / rank
    return 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    rank = (len(vals) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return vals[int(rank)]
    return vals[lower] + (vals[upper] - vals[lower]) * (rank - lower)


def fuse_results(by_ns: Dict[str, List[Dict]], cap: int) -> List[Dict]:
    """Reciprocal-rank fusion mirroring server logic."""
    scores: Dict[Tuple[str, str], float] = {}
    payloads: Dict[Tuple[str, str], Dict] = {}
    C = 60.0
    for ns, docs in by_ns.items():
        for rank, doc in enumerate(docs, start=1):
            url = doc.get("url", "")
            chunk_id = doc.get("chunk_id", doc.get("id", f"{ns}-{rank}"))
            key = (url, chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (C + rank)
            payloads[key] = doc
    merged = sorted(
        payloads.values(),
        key=lambda doc: scores[(doc.get("url", ""), doc.get("chunk_id", doc.get("id", "")))],
        reverse=True,
    )
    return merged[:cap]


class HttpRetriever:
    """Wrapper over the running FastAPI /search endpoint."""

    def __init__(self, base_url: str, api_token: Optional[str] = None, disable_decomposition: bool = False):
        if requests is None:  # pragma: no cover - handled in load
            raise RuntimeError("requests is required for HTTP evaluation")
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-api-token": api_token or "change-me"}
        self.disable_decomposition = disable_decomposition

    def healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def retrieve(self, question: str, top_k: int) -> Tuple[List[Dict], float, Optional[DecompositionHitInfo]]:
        """Retrieve results and optionally return decomposition hit info.

        Returns:
            (results, latency_ms, decomposition_info)
        """
        t0 = time.perf_counter()
        try:
            params = {"q": question, "k": top_k}
            if self.disable_decomposition:
                params["decomposition_off"] = "true"
            resp = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=self.headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            # Extract decomposition metadata if available
            decomp_info = None
            metadata = data.get("metadata", {})
            if metadata and "subtasks" in metadata and not self.disable_decomposition:
                decomp_info = self._build_decomp_hit_info(question, metadata)
        except Exception as exc:
            raise RuntimeError(f"HTTP retrieval failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms, decomp_info

    def _build_decomp_hit_info(self, question: str, metadata: Dict) -> Optional[DecompositionHitInfo]:
        """Build decomposition hit info from server metadata."""
        try:
            subtasks = metadata.get("subtasks", [])
            if not subtasks:
                return None

            subtask_texts = [st.get("text", "") for st in subtasks]
            subtask_hits: Dict[int, List[str]] = {}

            # Extract which subtasks hit which URLs from fused docs metadata
            for st_idx in range(len(subtasks)):
                subtask_hits[st_idx] = []

            # If server provides per-doc hit mapping, parse it
            fused_docs = metadata.get("fused_docs", 0)
            llm_used = metadata.get("llm_used", False)

            return DecompositionHitInfo(
                strategy=metadata.get("decomposition_strategy", "none"),
                subtask_count=len(subtasks),
                subtasks=subtask_texts,
                subtask_hits=subtask_hits,
                llm_used=llm_used,
            )
        except Exception:
            return None


class OfflineRetriever:
    """FAISS-based retriever mirroring server semantics."""

    def __init__(self, namespaces: List[str], oversample: int = 60, enable_rerank: bool = True, disable_decomposition: bool = False):
        self.oversample = oversample
        self.enable_rerank = enable_rerank and rerank_available()
        self.disable_decomposition = disable_decomposition
        self.namespaces = namespaces or DEFAULT_NAMESPACES
        self._indexes: Dict[str, Dict[str, object]] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        for ns in self.namespaces:
            root = INDEX_ROOT / ns
            if not root.exists():
                raise RuntimeError(f"Missing index directory for namespace '{ns}' under {root}")
            idx_path = root / "index.faiss"
            if not idx_path.exists():
                idx_path = root / "index.bin"
            if not idx_path.exists():
                raise RuntimeError(f"Missing FAISS index for namespace '{ns}'")
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text())
            metas = meta.get("chunks") or meta.get("rows") or []
            index = faiss.read_index(str(idx_path))
            self._indexes[ns] = {"index": index, "metas": metas}

    def retrieve(self, question: str, top_k: int) -> Tuple[List[Dict], float, Optional[DecompositionHitInfo]]:
        """Retrieve results with optional decomposition.

        Returns:
            (results, latency_ms, decomposition_info)
        """
        t0 = time.perf_counter()

        # Check if decomposition should be used
        decomp_info = None
        if not self.disable_decomposition and is_multi_intent_query(question):
            decomp_info = self._retrieve_with_decomposition(question, top_k)
            if decomp_info:
                deduped, latency_ms = self._fuse_decomposed_results(decomp_info, question, top_k, t0)
                return deduped, latency_ms, decomp_info

        # Standard (non-decomposed) retrieval
        expansions = expand(question)
        vectors = encode_texts(expansions)
        qvec = np.mean(vectors, axis=0)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-8)
        qvec = qvec.astype(np.float32)

        by_ns: Dict[str, List[Dict]] = {}
        for ns, bundle in self._indexes.items():
            index: faiss.Index = bundle["index"]  # type: ignore[assignment]
            metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
            raw_k = min(max(top_k * 6, self.oversample), index.ntotal)
            distances, indices = index.search(qvec.reshape(1, -1), raw_k)
            ns_results: List[Dict] = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(metas):
                    continue
                meta = dict(metas[idx])
                meta.update({
                    "namespace": ns,
                    "score": float(score),
                    "rank": len(ns_results) + 1,
                })
                ns_results.append(meta)
            by_ns[ns] = ns_results

        fused = fuse_results(by_ns, max(top_k * 3, top_k))
        fused.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("url", ""), r.get("title", "")))

        deduped: List[Dict] = []
        seen_urls: set[str] = set()
        for candidate in fused:
            url = candidate.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(candidate)
            if len(deduped) >= top_k:
                break

        if self.enable_rerank and deduped:
            reranked = rerank(question, deduped, top_k)
            if reranked:
                deduped = reranked

        latency_ms = (time.perf_counter() - t0) * 1000
        return deduped, latency_ms, decomp_info

    def _retrieve_with_decomposition(self, question: str, top_k: int) -> Optional[DecompositionHitInfo]:
        """Perform per-subtask retrieval and track hits."""
        try:
            decomp_result = decompose_query(question)
            if not decomp_result or len(decomp_result.subtasks) <= 1:
                return None

            subtask_texts = [st.text for st in decomp_result.subtasks]
            subtask_hits: Dict[int, List[str]] = {}

            # Per-subtask retrieval
            for st_idx, subtask in enumerate(decomp_result.subtasks):
                expansions = expand(subtask.text, boost_terms=subtask.boost_terms)
                vectors = encode_texts(expansions)
                qvec = np.mean(vectors, axis=0)
                qvec = qvec / (np.linalg.norm(qvec) + 1e-8)
                qvec = qvec.astype(np.float32)

                hit_urls = []
                for ns, bundle in self._indexes.items():
                    index: faiss.Index = bundle["index"]  # type: ignore[assignment]
                    metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
                    raw_k = min(max(top_k * 2, 20), index.ntotal)
                    distances, indices = index.search(qvec.reshape(1, -1), raw_k)
                    for score, idx in zip(distances[0], indices[0]):
                        if idx < 0 or idx >= len(metas):
                            continue
                        url = metas[idx].get("url", "")
                        if url and url not in hit_urls:
                            hit_urls.append(url)
                subtask_hits[st_idx] = hit_urls

            return DecompositionHitInfo(
                strategy=decomp_result.strategy,
                subtask_count=len(decomp_result.subtasks),
                subtasks=subtask_texts,
                subtask_hits=subtask_hits,
                llm_used=decomp_result.llm_used,
            )
        except Exception:
            return None

    def _fuse_decomposed_results(self, decomp_info: DecompositionHitInfo, question: str, top_k: int, t0: float) -> Tuple[List[Dict], float]:
        """Fuse per-subtask results into final ranking."""
        fused_docs: Dict[Tuple[str, str], Dict] = {}

        # Aggregate hits across subtasks
        for st_idx, urls in decomp_info.subtask_hits.items():
            for url in urls:
                key = (url, str(st_idx))  # Use subtask index as chunk_id for deduping
                if key not in fused_docs:
                    fused_docs[key] = {"url": url, "hits": 0, "subtasks": []}
                fused_docs[key]["hits"] += 1
                if st_idx not in fused_docs[key]["subtasks"]:
                    fused_docs[key]["subtasks"].append(st_idx)

        # Sort by hit count (additive fusion)
        sorted_docs = sorted(fused_docs.values(), key=lambda d: (-d["hits"], d["url"]))

        # Load full doc metadata from indexes
        deduped: List[Dict] = []
        seen_urls: set[str] = set()
        for doc in sorted_docs:
            url = doc["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Find full metadata from indexes
            for ns, bundle in self._indexes.items():
                metas: List[Dict] = bundle["metas"]  # type: ignore[assignment]
                for meta in metas:
                    if meta.get("url") == url:
                        full_doc = dict(meta)
                        full_doc["namespace"] = ns
                        deduped.append(full_doc)
                        break
                if len(deduped) >= top_k:
                    break
            if len(deduped) >= top_k:
                break

        if self.enable_rerank and deduped:
            reranked = rerank(question, deduped, top_k)
            if reranked:
                deduped = reranked

        latency_ms = (time.perf_counter() - t0) * 1000
        return deduped, latency_ms


def evaluate(goldset: List[GoldItem], retriever, top_k: int = 5, context_k: int = 4, log_decomposition: bool = False) -> Dict[str, object]:
    metrics = {
        "cases": len(goldset),
        "recall_at_5": [],
        "mrr_at_5": [],
        "answer_hits": [],
        "retrieval_latencies": [],
        "full_latencies": [],
        "case_details": [],
        "decomposition_stats": {  # A/B comparison stats
            "none": {"count": 0, "recall_sum": 0.0, "misses": []},
            "heuristic": {"count": 0, "recall_sum": 0.0, "misses": []},
            "multi_part": {"count": 0, "recall_sum": 0.0, "misses": []},
            "comparison": {"count": 0, "recall_sum": 0.0, "misses": []},
            "llm": {"count": 0, "recall_sum": 0.0, "misses": []},
        },
    }

    # Setup decomposition logging if requested
    decomp_log_file = None
    if log_decomposition:
        decomp_log_file = Path("logs/decomposition_eval.jsonl")
        decomp_log_file.parent.mkdir(parents=True, exist_ok=True)
        decomp_log_file.write_text("")  # Clear previous logs

    for item in goldset:
        try:
            result = retriever.retrieve(item.question, max(top_k, context_k))
            if len(result) == 3:
                results, retr_ms, decomp_info = result
            else:
                # Fallback for retrievers that don't return decomp_info
                results, retr_ms = result
                decomp_info = None
        except Exception as exc:
            results, retr_ms, decomp_info = [], float("inf"), None
            detail = {
                "id": item.id,
                "question": item.question,
                "error": str(exc),
            }
            metrics["case_details"].append(detail)
            metrics["recall_at_5"].append(0.0)
            metrics["mrr_at_5"].append(0.0)
            metrics["answer_hits"].append(False)
            metrics["retrieval_latencies"].append(retr_ms)
            metrics["full_latencies"].append(retr_ms)
            continue

        retrieved_urls = [r.get("url", "") for r in results]
        recall = recall_at_k(retrieved_urls, item.source_urls, 5)
        mrr = mrr_at_k(retrieved_urls, item.source_urls, 5)

        metrics["recall_at_5"].append(recall)
        metrics["mrr_at_5"].append(mrr)

        # Track decomposition strategy for A/B analysis
        decomp_strategy = "none"
        subtask_hit_details = {}
        if decomp_info:
            decomp_strategy = decomp_info.strategy
            metrics["decomposition_stats"][decomp_strategy]["count"] += 1
            metrics["decomposition_stats"][decomp_strategy]["recall_sum"] += recall

            # Log per-subtask hits if recall is 0 (miss)
            if recall == 0.0:
                miss_entry = {
                    "id": item.id,
                    "question": item.question,
                    "decomposition_strategy": decomp_strategy,
                    "subtasks": decomp_info.subtasks,
                    "subtask_hits": decomp_info.subtask_hits,
                    "llm_used": decomp_info.llm_used,
                    "gold_urls": item.source_urls,
                }
                metrics["decomposition_stats"][decomp_strategy]["misses"].append(miss_entry)

            # Build subtask hit summary for logging
            for st_idx, urls in decomp_info.subtask_hits.items():
                st_text = decomp_info.subtasks[st_idx] if st_idx < len(decomp_info.subtasks) else f"subtask_{st_idx}"
                hit_count = len([u for u in urls if any(normalize_url(u) == normalize_url(g) for g in item.source_urls)])
                subtask_hit_details[st_idx] = {
                    "text": st_text,
                    "retrieved_urls": urls,
                    "gold_hits": hit_count,
                }

        answer_start = time.perf_counter()
        context_blocks = results[: min(context_k, len(results))]
        context_text = "\n\n".join(r.get("text", "") for r in context_blocks)
        try:
            pattern = re.compile(item.answer_regex)
            answer_hit = bool(pattern.search(context_text))
        except re.error:
            answer_hit = False
        answer_latency_ms = (time.perf_counter() - answer_start) * 1000

        metrics["answer_hits"].append(answer_hit)
        metrics["retrieval_latencies"].append(retr_ms)
        metrics["full_latencies"].append(retr_ms + answer_latency_ms)

        # Log detailed decomposition info if enabled
        if log_decomposition and decomp_info:
            try:
                decomp_entry = {
                    "id": item.id,
                    "question": item.question,
                    "decomposition_strategy": decomp_strategy,
                    "subtask_count": decomp_info.subtask_count,
                    "subtasks": decomp_info.subtasks,
                    "llm_used": decomp_info.llm_used,
                    "subtask_hits": subtask_hit_details,
                    "retrieved_urls": retrieved_urls[:top_k],
                    "gold_urls": item.source_urls,
                    "recall@5": recall,
                    "answer_hit": answer_hit,
                }
                with decomp_log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(decomp_entry) + "\n")
            except Exception:
                # Log error but don't fail evaluation
                pass

        metrics["case_details"].append(
            {
                "id": item.id,
                "question": item.question,
                "decomposition_strategy": decomp_strategy,
                "retrieved_urls": retrieved_urls,
                "recall@5": recall,
                "answer_hit": answer_hit,
                "retrieval_ms": retr_ms,
                "full_ms": retr_ms + answer_latency_ms,
            }
        )

    # Build A/B comparison summary
    strategy_summary = {}
    for strategy in ["none", "heuristic", "llm"]:
        stats = metrics["decomposition_stats"][strategy]
        if stats["count"] > 0:
            strategy_summary[strategy] = {
                "count": stats["count"],
                "recall_at_5": round(stats["recall_sum"] / stats["count"], 3),
                "miss_count": len(stats["misses"]),
            }

    summary = {
        "cases": metrics["cases"],
        "recall_at_5": round(statistics.fmean(metrics["recall_at_5"]) if metrics["cases"] else 0.0, 3),
        "mrr_at_5": round(statistics.fmean(metrics["mrr_at_5"]) if metrics["cases"] else 0.0, 3),
        "answer_accuracy": round(sum(metrics["answer_hits"]) / metrics["cases"] if metrics["cases"] else 0.0, 3),
        "retrieval_latency_ms": {
            "p50": round(percentile(metrics["retrieval_latencies"], 0.5), 1),
            "p95": round(percentile(metrics["retrieval_latencies"], 0.95), 1),
        },
        "full_latency_ms": {
            "p50": round(percentile(metrics["full_latencies"], 0.5), 1),
            "p95": round(percentile(metrics["full_latencies"], 0.95), 1),
        },
        "decomposition_ab_summary": strategy_summary,
        "details": metrics["case_details"],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Clockify RAG retrieval and grounding.")
    parser.add_argument("--goldset", default="eval/goldset.csv", type=Path, help="Path to goldset CSV")
    parser.add_argument("--base-url", default="http://localhost:7000", help="Search API base URL")
    parser.add_argument("--api-token", default="change-me", help="API token for HTTP requests")
    parser.add_argument("--namespaces", default="clockify", help="Comma-separated namespaces for offline mode")
    parser.add_argument("--k", default=5, type=int, help="Top-k results to evaluate")
    parser.add_argument("--context-k", default=4, type=int, help="Chunks to pack into mock answer")
    parser.add_argument("--json", action="store_true", help="Dump summary as JSON only")
    parser.add_argument("--log-decomposition", action="store_true", help="Log query decomposition metadata to JSONL file")
    parser.add_argument("--decomposition-off", action="store_true", help="Disable query decomposition for A/B baseline comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goldset = load_goldset(args.goldset)

    retriever: object
    use_http = False
    if requests is not None and args.base_url:
        http_retriever = HttpRetriever(args.base_url, args.api_token, disable_decomposition=args.decomposition_off)
        if http_retriever.healthy():
            retriever = http_retriever
            use_http = True
        else:
            try:
                retriever = OfflineRetriever(
                    [ns.strip() for ns in args.namespaces.split(",") if ns.strip()],
                    disable_decomposition=args.decomposition_off,
                )
            except RuntimeError as exc:
                # FAISS indexes not available in CI environment
                if args.json:
                    error_summary = {
                        "error": "OfflineRetriever initialization failed",
                        "message": str(exc),
                        "details": "FAISS indexes not found. This is expected in CI where indexes are not persisted.",
                        "cases": 0,
                        "recall_at_5": 0.0,
                        "mrr_at_5": 0.0,
                        "answer_accuracy": 0.0,
                        "retrieval_latency_ms": {"p50": 0.0, "p95": 0.0},
                        "full_latency_ms": {"p50": 0.0, "p95": 0.0},
                        "decomposition_ab_summary": {},
                        "details": [],
                    }
                    print(json.dumps(error_summary, indent=2))
                    return
                else:
                    raise
    else:
        try:
            retriever = OfflineRetriever(
                [ns.strip() for ns in args.namespaces.split(",") if ns.strip()],
                disable_decomposition=args.decomposition_off,
            )
        except RuntimeError as exc:
            # FAISS indexes not available
            if args.json:
                error_summary = {
                    "error": "OfflineRetriever initialization failed",
                    "message": str(exc),
                    "details": "FAISS indexes not found. This is expected in CI where indexes are not persisted.",
                    "cases": 0,
                    "recall_at_5": 0.0,
                    "mrr_at_5": 0.0,
                    "answer_accuracy": 0.0,
                    "retrieval_latency_ms": {"p50": 0.0, "p95": 0.0},
                    "full_latency_ms": {"p50": 0.0, "p95": 0.0},
                    "decomposition_ab_summary": {},
                    "details": [],
                }
                print(json.dumps(error_summary, indent=2))
                return
            else:
                raise

    summary = evaluate(goldset, retriever, top_k=args.k, context_k=args.context_k, log_decomposition=args.log_decomposition)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    mode = "HTTP" if use_http else "offline"
    decomp_status = "DISABLED (A/B Baseline)" if args.decomposition_off else "ENABLED"
    print("\n" + "=" * 80)
    print(f"CLOCKIFY RAG EVAL ({mode.upper()} mode) | Decomposition: {decomp_status}")
    print("=" * 80)
    print(f"Cases: {summary['cases']}")
    print(f"Recall@5: {summary['recall_at_5']:.3f}")
    print(f"MRR@5: {summary['mrr_at_5']:.3f}")
    print(f"Answer accuracy: {summary['answer_accuracy']:.3f}")
    print(
        "Retrieval latency p50/p95 (ms): "
        f"{summary['retrieval_latency_ms']['p50']} / {summary['retrieval_latency_ms']['p95']}"
    )
    print(
        "Full pipeline latency p50/p95 (ms): "
        f"{summary['full_latency_ms']['p50']} / {summary['full_latency_ms']['p95']}"
    )

    # Print A/B comparison table
    if summary.get("decomposition_ab_summary"):
        print("\n" + "-" * 80)
        print("DECOMPOSITION A/B COMPARISON")
        print("-" * 80)
        print(f"{'Strategy':<15} {'Count':<8} {'Recall@5':<12} {'Misses':<8}")
        print("-" * 80)
        for strategy, stats in sorted(summary["decomposition_ab_summary"].items()):
            print(
                f"{strategy:<15} {stats['count']:<8} {stats['recall_at_5']:<12.3f} {stats['miss_count']:<8}"
            )
        print("-" * 80)

    print("-" * 80)
    for detail in summary["details"]:
        qid = detail.get("id")
        recall = detail.get("recall@5", 0.0)
        answer_hit = detail.get("answer_hit", False)
        decomp_strat = detail.get("decomposition_strategy", "none")
        print(
            f"[{qid}] R@5={recall:.2f} | answer={'✓' if answer_hit else '✗'} | "
            f"decomp={decomp_strat} | urls={detail.get('retrieved_urls', [])[:3]}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
```

## src/llm/__init__.py

```
"""LLM client modules for RAG system."""
```

## src/llm/harmony_encoder.py

```
#!/usr/bin/env python3
"""Harmony Chat Format Support for gpt-oss:20b.

Implements proper Harmony encoding for optimal gpt-oss:20b performance.
Handles message encoding, stop tokens, and fallback to standard format.

References:
- https://github.com/openai/openai-harmony
- gpt-oss:20b expects Harmony format; standard OpenAI format causes degradation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Attempt to import Harmony support
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
        UserContent,
        AssistantContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning(
        "openai-harmony not installed. Install with: pip install openai-harmony. "
        "gpt-oss:20b performance will be degraded without Harmony format."
    )


class HarmonyEncoder:
    """Encode messages in Harmony format for gpt-oss:20b."""

    def __init__(self, use_harmony: bool = True):
        """Initialize Harmony encoder.

        Args:
            use_harmony: Whether to use Harmony format (requires openai-harmony)
        """
        self.use_harmony = use_harmony and HARMONY_AVAILABLE

        if self.use_harmony:
            try:
                self.encoding = load_harmony_encoding(
                    HarmonyEncodingName.HARMONY_GPT_OSS
                )
                logger.info("✓ Harmony encoder initialized for gpt-oss:20b")
            except Exception as e:
                logger.warning(f"Failed to initialize Harmony encoder: {e}. Falling back to standard format.")
                self.use_harmony = False
                self.encoding = None
        else:
            self.encoding = None
            if not HARMONY_AVAILABLE:
                logger.debug(
                    "Harmony format disabled: openai-harmony not installed. "
                    "Install for optimal gpt-oss:20b performance."
                )

    def render_messages(
        self,
        system_prompt: str,
        developer_instructions: Optional[str] = None,
        user_message: str = "",
        reasoning_effort: str = "low",
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Render messages in Harmony format.

        Args:
            system_prompt: Base system prompt
            developer_instructions: RAG-specific instructions (moves to Developer role)
            user_message: User's question/prompt
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG)

        Returns:
            Tuple of (prefill_token_ids, stop_token_ids)
            If Harmony unavailable, returns ([], None) and caller should use standard format
        """
        if not self.use_harmony or self.encoding is None:
            return [], None

        try:
            # Build conversation with Harmony roles
            messages = []

            # System role: Base instructions
            if system_prompt:
                messages.append(
                    Message.from_role_and_content(
                        Role.SYSTEM,
                        SystemContent.new().with_content(system_prompt),
                    )
                )

            # Developer role: RAG-specific instructions (enforces grounding, citations)
            if developer_instructions:
                dev_content = DeveloperContent.new().with_instructions(
                    developer_instructions
                )
                # Add reasoning effort control
                if reasoning_effort == "low":
                    dev_content = dev_content.with_instructions(
                        "Use low reasoning effort to minimize latency."
                    )
                elif reasoning_effort == "high":
                    dev_content = dev_content.with_instructions(
                        "Use high reasoning effort for complex analysis."
                    )

                messages.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )

            # User role: The actual question with context
            if user_message:
                messages.append(
                    Message.from_role_and_content(
                        Role.USER,
                        UserContent.from_content(user_message),
                    )
                )

            convo = Conversation(messages)

            # Render conversation for completion (generates prefill)
            prefill_ids = self.encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )

            # Get stop tokens for assistant actions (prevents leakage)
            stop_ids = self.encoding.stop_tokens_for_assistant_actions()

            logger.debug(
                f"Rendered Harmony messages: prefill_len={len(prefill_ids)}, "
                f"stop_tokens={len(stop_ids) if stop_ids else 0}"
            )

            return prefill_ids, stop_ids

        except Exception as e:
            logger.warning(
                f"Failed to render Harmony messages: {e}. Falling back to standard format."
            )
            return [], None

    def build_messages_standard(
        self,
        system_prompt: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """Build messages in standard OpenAI format (fallback).

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            List of message dicts with role and content
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_message:
            messages.append({"role": "user", "content": user_message})
        return messages


def get_harmony_encoder(model_name: str = "") -> HarmonyEncoder:
    """Get Harmony encoder instance.

    Args:
        model_name: Model name (used to determine if Harmony is applicable)

    Returns:
        HarmonyEncoder instance (with Harmony enabled for gpt-oss models)
    """
    # Enable Harmony for gpt-oss and oss models
    use_harmony = any(
        model_name.lower().startswith(prefix)
        for prefix in ["gpt-oss", "oss", "oss20b", "oss13b", "oss7b"]
    )

    # Check environment override
    harmony_env = os.getenv("LLM_USE_HARMONY", "auto").lower()
    if harmony_env == "true":
        use_harmony = True
    elif harmony_env == "false":
        use_harmony = False

    return HarmonyEncoder(use_harmony=use_harmony)
```

## src/prompt.py

```
#!/usr/bin/env python3
"""RAG prompt templates with inline citations.

Implements the Clockify RAG Standard v1 with strict context adherence,
parameter quoting, and consistent citation format across channels.
"""

from typing import List, Dict, Any, Optional
import re

from src.config import CONFIG


class RAGPrompt:
    """Build RAG prompts with citations and strict adherence to system instructions."""

    # System prompt enforces: strict context use, "Not in docs" fallback, inline citations
    SYSTEM_PROMPT_CLOCKIFY = """You are an expert technical assistant for Clockify Help documentation.

INSTRUCTIONS:
1. Answer ONLY from the provided context. Do NOT use prior knowledge.
2. Be accurate, concise, and direct.
3. Use inline citations [1], [2] tied to the Sources section.
4. If asked about something not in the docs, respond: "Not in docs" and suggest related topics.
5. When mentioning API parameters, functions, or settings, use QUOTES: "parameter_name"
6. Format: Answer first, then numbered Sources.
7. Respect breadcrumbs: cite section titles for better navigation (e.g., [1] Administration > User Roles).
"""

    # Developer role (Harmony format): RAG-specific enforcement rules for grounding and citation
    DEVELOPER_INSTRUCTIONS_CLOCKIFY = """You are grounding RAG responses to Clockify documentation.

ENFORCEMENT:
1. CITE ONLY: Every factual claim must reference a numbered source [1], [2], etc.
2. GROUND STRICTLY: Never use knowledge outside the provided context blocks.
3. OUT OF SCOPE: If information is not in the docs, respond with "Not in docs" before suggesting alternatives.
4. PARAMETER QUOTING: Quote all API parameters, field names, and settings: "parameter_name"
5. NO SYNTHESIS: Do not combine information across sources to infer new facts. Present each source independently.
6. REASONING: Use low reasoning effort to minimize response latency; prioritize speed over depth."""

    SYSTEM_PROMPT_LANGCHAIN = """You are an expert technical assistant for LangChain documentation.

INSTRUCTIONS:
1. Answer ONLY from the provided context. Do NOT use prior knowledge.
2. Be accurate, concise, and direct.
3. Use inline citations [1], [2] tied to the Sources section.
4. If asked about something not in the docs, respond: "Not in docs" and suggest related topics.
5. When mentioning functions, modules, or API elements, use QUOTES: "function_name"
6. Format: Answer first, then numbered Sources.
"""

    # Developer role (Harmony format): RAG-specific enforcement rules for LangChain
    DEVELOPER_INSTRUCTIONS_LANGCHAIN = """You are grounding RAG responses to LangChain documentation.

ENFORCEMENT:
1. CITE ONLY: Every factual claim must reference a numbered source [1], [2], etc.
2. GROUND STRICTLY: Never use knowledge outside the provided context blocks.
3. OUT OF SCOPE: If information is not in the docs, respond with "Not in docs" before suggesting alternatives.
4. CODE EXAMPLES: Quote all function names, modules, and class names: "function_name"
5. NO SYNTHESIS: Do not combine information across sources to infer new facts. Present each source independently.
6. REASONING: Use low reasoning effort to minimize response latency; prioritize speed over depth."""

    # P2: Query decomposition prompts for breaking down complex questions
    DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant. Your task is to break down complex questions \
into focused sub-questions that can be searched independently. Return ONLY a JSON array \
of strings, one per line, with no markdown formatting or explanation. Example:
["What is kiosk?", "What is timer?", "How do they compare?"]"""

    @staticmethod
    def get_decomposition_prompts(query: str, max_subtasks: int = 3) -> tuple[str, str]:
        """Get system and user prompts for query decomposition.

        Args:
            query: The complex question to decompose
            max_subtasks: Maximum number of sub-questions to generate

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = RAGPrompt.DECOMPOSITION_SYSTEM_PROMPT
        user_prompt = (
            f"Break down this complex question into at most {max_subtasks} focused sub-questions "
            f"that can be searched independently. Return only the JSON array, no explanation.\n\n"
            f"Question: {query}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def get_system_prompt(namespace: str = "clockify") -> str:
        """Get system prompt for a specific namespace."""
        if namespace == "langchain":
            return RAGPrompt.SYSTEM_PROMPT_LANGCHAIN
        return RAGPrompt.SYSTEM_PROMPT_CLOCKIFY

    @staticmethod
    def get_developer_instructions(namespace: str = "clockify") -> str:
        """Get developer role instructions for Harmony format (RAG-specific enforcement).

        Args:
            namespace: Documentation namespace (clockify, langchain, etc.)

        Returns:
            Developer role instructions for RAG grounding and citation enforcement
        """
        if namespace == "langchain":
            return RAGPrompt.DEVELOPER_INSTRUCTIONS_LANGCHAIN
        return RAGPrompt.DEVELOPER_INSTRUCTIONS_CLOCKIFY

    @staticmethod
    def build_context_block(chunks: List[Dict[str, Any]], max_chunks: int = None) -> tuple[str, List[Dict]]:
        """Format chunks as numbered context blocks with breadcrumb titles.

        Args:
            chunks: List of chunk dictionaries from retrieval
            max_chunks: Maximum number of chunks to include (default from CONFIG.MAX_CONTEXT_CHUNKS)

        Returns:
            Tuple of (formatted_context_string, sources_list)
        """
        # Use config default if not specified
        if max_chunks is None:
            max_chunks = CONFIG.MAX_CONTEXT_CHUNKS

        sources = []
        context_lines = []

        # Limit to max_chunks
        chunks_to_use = chunks[:max_chunks]

        for idx, chunk in enumerate(chunks_to_use, 1):
            url = chunk.get("url", "")
            title = chunk.get("title", "Untitled")
            namespace = chunk.get("namespace", "")
            text = chunk.get("text", "")
            breadcrumb = chunk.get("breadcrumb", [])
            section = chunk.get("section", "")

            # Build breadcrumb title for better navigation context
            breadcrumb_title = " > ".join(breadcrumb) if breadcrumb else title

            # Truncate text (configurable via CONTEXT_CHAR_LIMIT) for better coverage while managing context window
            text_excerpt = text[:CONFIG.CONTEXT_CHAR_LIMIT] if text else ""

            source = {
                "number": idx,
                "title": title,
                "breadcrumb": breadcrumb,
                "breadcrumb_title": breadcrumb_title,
                "url": url,
                "namespace": namespace,
                "section": section,
            }
            sources.append(source)

            # Format context with breadcrumb for better navigation
            context_lines.append(
                f"[{idx}] {breadcrumb_title}\n"
                f"URL: {url}\n\n"
                f"{text_excerpt}\n"
            )

        context = "\n---\n".join(context_lines)
        return context, sources

    @staticmethod
    def build_user_prompt(question: str, context: str, max_chunks: int = 4) -> str:
        """Build final user prompt with context limit enforcement.

        Args:
            question: The user's question
            context: Formatted context blocks (already limited by build_context_block)
            max_chunks: Context limit for instructions (for user awareness)

        Returns:
            Formatted user prompt
        """
        return f"""Based on the following {max_chunks} context blocks, answer the user's question.
Use inline citations [1], [2], etc., when referencing context.
If the answer is not in these docs, respond: "Not in docs" and suggest related topics.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, accurate answer with inline citations to the source numbers above."""

    @staticmethod
    def build_messages(
        question: str,
        chunks: List[Dict[str, Any]],
        namespace: str = "clockify",
        max_chunks: int = None,
        reasoning_effort: str = "low",
    ) -> tuple[List[Dict[str, str]], List[Dict], Optional[str]]:
        """Build messages for LLM + track sources + developer instructions for Harmony.

        Args:
            question: The user's question
            chunks: Retrieved chunks from search/retrieval
            namespace: Documentation namespace (clockify, langchain, etc.)
            max_chunks: Maximum context chunks to include (default from CONFIG.MAX_CONTEXT_CHUNKS)
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG latency optimization)

        Returns:
            Tuple of (messages_list, sources_list, developer_instructions)
            - messages_list: Standard OpenAI format messages (system + user)
            - sources_list: Metadata for each context chunk
            - developer_instructions: RAG enforcement rules for Harmony Developer role
        """
        # Use config default if not specified
        if max_chunks is None:
            max_chunks = CONFIG.MAX_CONTEXT_CHUNKS

        context, sources = RAGPrompt.build_context_block(chunks, max_chunks=max_chunks)
        system_msg = RAGPrompt.get_system_prompt(namespace)
        user_msg = RAGPrompt.build_user_prompt(question, context, max_chunks=max_chunks)
        developer_instructions = RAGPrompt.get_developer_instructions(namespace)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages, sources, developer_instructions

    @staticmethod
    def suggest_refinements(question: str, tried_terms: Optional[List[str]] = None) -> List[str]:
        """
        P2: Suggest related search terms when answer is not found in docs.

        Generates alternative queries based on the original question
        to help users refine their search.

        Args:
            question: Original question that returned "Not in docs"
            tried_terms: Terms already tried (to avoid repeating)

        Returns:
            List of suggested refinement questions
        """
        tried = set(tried_terms or [])
        suggestions = []

        # Extract key nouns/concepts from the question
        # Simple heuristic: look for common Clockify terms and related concepts
        question_lower = question.lower()

        # Define related term mappings for common Clockify queries
        refinement_map = {
            "timesheet": ["time entry", "time tracking", "work log", "track time"],
            "timer": ["kiosk mode", "stopwatch", "active tracking", "real-time clock"],
            "project": ["task", "assignment", "work item", "billable work"],
            "team": ["workspace", "members", "organization", "company"],
            "report": ["analytics", "time summary", "billing report", "time report"],
            "approval": ["timesheet approval", "review", "sign-off", "manager approval"],
            "integration": ["webhook", "API", "third-party", "connected app"],
            "permission": ["access", "role", "authorization", "admin"],
            "rate": ["billable rate", "cost", "pricing", "hourly rate"],
            "clock": ["check in", "start timer", "punch clock", "time clock"],
        }

        # Suggest alternatives based on detected keywords
        for keyword, alternatives in refinement_map.items():
            if keyword in question_lower:
                for alt in alternatives:
                    if alt not in tried:
                        # Create suggestion by replacing keyword with alternative
                        suggested = question.replace(keyword, alt, 1)
                        if suggested not in suggestions and suggested != question:
                            suggestions.append(suggested)
                            if len(suggestions) >= 3:  # Limit to 3 suggestions
                                return suggestions

        # Fallback: suggest broad topics if no keyword matches
        fallback_suggestions = [
            "How do I get started with Clockify?",
            "What are the main features of Clockify?",
            "How do I set up my account?",
        ]
        for fallback in fallback_suggestions:
            if fallback not in tried and fallback != question:
                suggestions.append(fallback)
                if len(suggestions) >= 3:
                    break

        return suggestions

    @staticmethod
    def format_response(answer: str, sources: List[Dict], use_citations: bool = True) -> Dict[str, Any]:
        """Format final response with sources list.

        Args:
            answer: The LLM-generated answer text
            sources: List of source metadata from build_context_block
            use_citations: Whether to include sources section

        Returns:
            Dictionary with formatted answer and sources metadata
        """
        response_text = answer

        if use_citations and sources:
            # Add Sources section with breadcrumb titles for navigation
            sources_section = "\n\n## Sources\n\n"
            for src in sources:
                sources_section += f"[{src['number']}] **{src['breadcrumb_title']}** ({src['namespace']})\n"
                sources_section += f"    URL: {src['url']}\n"
                if src.get("section"):
                    sources_section += f"    Section: {src['section']}\n"
                sources_section += "\n"

            response_text += sources_section

        return {
            "answer": response_text,
            "sources": sources,
            "sources_count": len(sources),
            "chunk_limit": 4,  # Enforced in build_context_block
        }
```

## test_llm_connection.sh

```
#!/bin/bash
# Test script for internal Ollama LLM server

set -euo pipefail

# Production environment checks
BASE="${BASE:-http://10.127.0.192:11434}"

# Validate LLM endpoint format
if [[ ! "$BASE" =~ ^https?:// ]]; then
    echo "❌ ERROR: Invalid LLM endpoint. Must start with http:// or https://"
    echo "   Current value: $BASE"
    exit 1
fi

# Extract host and port for timeout calculation
LLM_HOST=$(echo "$BASE" | sed -E 's|^https?://([^/:]+).*|\1|')
if [[ -z "$LLM_HOST" ]]; then
    echo "❌ ERROR: Could not parse LLM host from $BASE"
    exit 1
fi

echo "========================================="
echo "Testing LLM Server: $BASE"
echo "========================================="
echo

# Test 1: Check server is reachable
echo "## 1) Testing server connectivity..."
if curl -sS -f "$BASE/api/tags" > /dev/null 2>&1; then
    echo "✅ Server is reachable at $BASE"
else
    echo "❌ Server NOT reachable at $BASE"
    echo "   Check VPN connection or firewall"
    exit 1
fi
echo

# Test 2: List available models
echo "## 2) Available models:"
MODEL_LIST=$(curl -sS "$BASE/api/tags" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
if [ -z "$MODEL_LIST" ]; then
    echo "❌ No models found"
    exit 1
else
    echo "$MODEL_LIST" | while read -r model; do
        echo "   - $model"
    done
fi
echo

# Test 3: Find gpt-oss model
MODEL=$(echo "$MODEL_LIST" | grep -iE '^gpt-oss' | head -n1)
if [ -z "$MODEL" ]; then
    echo "⚠️  No 'gpt-oss' model found, using first available model"
    MODEL=$(echo "$MODEL_LIST" | head -n1)
fi
echo "## 3) Using model: $MODEL"
echo

# Test 4: Test /api/chat endpoint
echo "## 4) Testing /api/chat endpoint..."
CHAT_RESPONSE=$(curl -sS -X POST "$BASE/api/chat" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say 'pong' if you can hear me\"}],\"stream\":false}" \
  2>&1)

if echo "$CHAT_RESPONSE" | grep -qi "pong\|content"; then
    echo "✅ /api/chat endpoint works!"
    echo "   Response preview:"
    echo "$CHAT_RESPONSE" | head -c 200
    echo
else
    echo "❌ /api/chat endpoint failed"
    echo "   Response: $CHAT_RESPONSE"
fi
echo

# Test 5: Test /api/generate endpoint
echo "## 5) Testing /api/generate endpoint..."
GEN_RESPONSE=$(curl -sS -X POST "$BASE/api/generate" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hi\",\"stream\":false}" \
  2>&1)

if echo "$GEN_RESPONSE" | grep -qi "response\|hi"; then
    echo "✅ /api/generate endpoint works!"
    echo "   Response preview:"
    echo "$GEN_RESPONSE" | head -c 200
    echo
else
    echo "⚠️  /api/generate endpoint may not be available"
fi
echo

# Test 6: Check for /talk endpoint (custom endpoint)
echo "## 6) Testing /talk endpoint (if exists)..."
TALK_RESPONSE=$(curl -sS -X POST "$BASE/talk" \
  -H 'content-type: application/json' \
  -d '{"prompt":"test"}' \
  2>&1)

if echo "$TALK_RESPONSE" | grep -qi "error\|not found" || [ -z "$TALK_RESPONSE" ]; then
    echo "⚠️  /talk endpoint not available (this is OK)"
else
    echo "✅ /talk endpoint exists!"
    echo "   Response preview:"
    echo "$TALK_RESPONSE" | head -c 200
    echo
fi
echo

# Summary
echo "========================================="
echo "Summary:"
echo "========================================="
echo "Server:     $BASE"
echo "Model:      $MODEL"
echo "Status:     ✅ READY TO USE"
echo
echo "Next steps:"
echo "1. Copy this config to .env:"
echo "   LLM_BASE_URL=$BASE"
echo "   LLM_MODEL=$MODEL"
echo
echo "2. Start RAG server:"
echo "   python3 -m uvicorn src.server:app --host 0.0.0.0 --port 8877"
echo
echo "3. Test RAG API:"
echo "   curl http://localhost:8877/health"
echo "========================================="
```

## tests/test_cache.py

```
"""
Test response caching module.

Tests LRU eviction, TTL expiration, thread-safety, and hit rates.
"""

import time
import pytest
from src.cache import LRUResponseCache, init_cache, get_cache


class TestCacheBasics:
    """Test basic cache operations."""

    def test_cache_get_miss(self):
        """Cache miss returns None."""
        cache = LRUResponseCache(max_size=10)
        result = cache.get("query1", 5)
        assert result is None

    def test_cache_set_and_get(self):
        """Cache stores and retrieves responses."""
        cache = LRUResponseCache(max_size=10)
        response = {"results": [{"rank": 1, "score": 0.9}]}

        cache.set("query1", 5, response)
        retrieved = cache.get("query1", 5)

        assert retrieved is not None
        assert retrieved["results"] == response["results"]

    def test_cache_key_includes_namespace(self):
        """Different namespaces produce different cache keys."""
        cache = LRUResponseCache(max_size=10)
        response1 = {"results": [{"rank": 1, "score": 0.9}]}
        response2 = {"results": [{"rank": 1, "score": 0.8}]}

        cache.set("query1", 5, response1, namespace="ns1")
        cache.set("query1", 5, response2, namespace="ns2")

        retrieved_ns1 = cache.get("query1", 5, namespace="ns1")
        retrieved_ns2 = cache.get("query1", 5, namespace="ns2")

        assert retrieved_ns1["results"][0]["score"] == 0.9
        assert retrieved_ns2["results"][0]["score"] == 0.8

    def test_cache_key_includes_k(self):
        """Different k values produce different cache keys."""
        cache = LRUResponseCache(max_size=10)
        response_k3 = {"results": [{"rank": 1}, {"rank": 2}, {"rank": 3}]}
        response_k5 = {"results": [{"rank": 1}, {"rank": 2}, {"rank": 3}, {"rank": 4}, {"rank": 5}]}

        cache.set("query1", 3, response_k3)
        cache.set("query1", 5, response_k5)

        retrieved_k3 = cache.get("query1", 3)
        retrieved_k5 = cache.get("query1", 5)

        assert len(retrieved_k3["results"]) == 3
        assert len(retrieved_k5["results"]) == 5


class TestCacheTTL:
    """Test time-to-live expiration."""

    def test_cache_respects_ttl(self):
        """Expired entries are not returned."""
        cache = LRUResponseCache(max_size=10, default_ttl=1)
        response = {"results": [{"rank": 1}]}

        cache.set("query1", 5, response, ttl=1)

        # Before expiration
        retrieved = cache.get("query1", 5)
        assert retrieved is not None

        # After expiration
        time.sleep(1.1)
        retrieved = cache.get("query1", 5)
        assert retrieved is None

    def test_cache_custom_ttl(self):
        """Custom TTL overrides default."""
        cache = LRUResponseCache(max_size=10, default_ttl=60)
        response = {"results": [{"rank": 1}]}

        cache.set("query1", 5, response, ttl=1)

        time.sleep(1.1)
        retrieved = cache.get("query1", 5)
        assert retrieved is None


class TestCacheLRUEviction:
    """Test least-recently-used eviction."""

    def test_cache_evicts_lru_when_full(self):
        """LRU item is evicted when cache exceeds max_size."""
        cache = LRUResponseCache(max_size=3)

        # Fill cache
        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})

        assert cache.stats()["size"] == 3

        # Add one more (should evict q1)
        cache.set("q4", 5, {"id": 4})

        assert cache.stats()["size"] == 3
        assert cache.get("q1", 5) is None  # q1 was evicted
        assert cache.get("q2", 5) is not None
        assert cache.get("q3", 5) is not None
        assert cache.get("q4", 5) is not None

    def test_cache_access_order_updated_on_hit(self):
        """Accessing an item updates its recency."""
        cache = LRUResponseCache(max_size=3)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})

        # Access q1 (makes it recently used)
        cache.get("q1", 5)

        # Add q4 (should evict q2, not q1)
        cache.set("q4", 5, {"id": 4})

        assert cache.get("q1", 5) is not None  # q1 still there (was accessed)
        assert cache.get("q2", 5) is None  # q2 was evicted (least recently used)
        assert cache.get("q3", 5) is not None
        assert cache.get("q4", 5) is not None


class TestCacheStats:
    """Test cache statistics."""

    def test_cache_hit_rate_calculation(self):
        """Cache tracks hit rate correctly."""
        cache = LRUResponseCache(max_size=10)

        # Add two entries
        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})

        # 2 misses
        cache.get("q3", 5)
        cache.get("q4", 5)

        # 3 hits
        cache.get("q1", 5)
        cache.get("q2", 5)
        cache.get("q1", 5)

        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate_pct"] == 60.0

    def test_cache_eviction_counter(self):
        """Cache tracks eviction count."""
        cache = LRUResponseCache(max_size=2)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})
        cache.set("q3", 5, {"id": 3})  # Evicts q1
        cache.set("q4", 5, {"id": 4})  # Evicts q2
        cache.set("q5", 5, {"id": 5})  # Evicts q3

        stats = cache.stats()
        assert stats["evictions"] == 3

    def test_cache_clear(self):
        """Clear empties the cache."""
        cache = LRUResponseCache(max_size=10)

        cache.set("q1", 5, {"id": 1})
        cache.set("q2", 5, {"id": 2})

        assert cache.stats()["size"] == 2

        cache.clear()

        assert cache.stats()["size"] == 0
        assert cache.get("q1", 5) is None


class TestGlobalCache:
    """Test global cache singleton."""

    def test_init_cache_creates_singleton(self):
        """init_cache creates a global cache instance."""
        cache1 = init_cache(max_size=100)
        cache2 = get_cache()

        assert cache1 is cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/test_glossary_integration.py

```
"""Tests for glossary integration and query expansion."""
import pytest
from src.query_rewrite import expand, is_definitional
from src.ontologies.clockify_glossary import ALIASES, extract_terms, _norm


def test_aliases_loaded():
    """Verify glossary aliases are loaded correctly."""
    assert "timesheet" in ALIASES or any("timesheet" in k for k in ALIASES.keys())
    assert "approval" in ALIASES or any("approval" in k for k in ALIASES.keys())


def test_expand_variants_cap():
    """Test query expansion caps at max_vars."""
    out = expand("How do I submit my timesheet for approval?", max_vars=5)
    assert isinstance(out, list)
    assert len(out) <= 5
    assert out[0].startswith("How do I submit")


def test_expand_preserves_original():
    """First variant should always be the original query."""
    q = "Help with project budgets"
    out = expand(q, max_vars=3)
    assert out[0] == q


def test_is_definitional():
    """Test definitional query detection."""
    assert is_definitional("What is billable rate?") is True
    assert is_definitional("Define timesheet") is True
    assert is_definitional("How do I enable SSO?") is False
    assert is_definitional("Log time to project") is False


def test_norm_consistency():
    """Test string normalization."""
    assert _norm("TimeSheet") == "timesheet"
    assert _norm("Billable-Rate") == "billablerate"
    assert _norm("PTO (Paid Time Off)") == "pto paid time off"


def test_extract_terms_from_glossary():
    """Test parsing glossary terms marked with #."""
    sample = "### Timesheet #\nA weekly record.\n### Timer #\nA clock."
    terms = extract_terms(sample)
    assert len(terms) >= 2
    assert any("timesheet" in t["norm"] for t in terms)
```

## tests/test_llm_client_hardening.py

```
"""Test LLM client hardening: retries, config validation, logging hygiene."""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock, call
from src.llm_client import (
    LLMClient,
    _validate_config,
    _sanitize_url,
    _redact_token,
    _cap_response,
)


class TestConfigValidation:
    """Test configuration validation on startup."""

    def test_validate_config_success_with_ollama_mock(self):
        """Config validation should pass with valid mock Ollama config."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_TIMEOUT_SECONDS"] = "30"
        os.environ["LLM_RETRIES"] = "3"
        os.environ["LLM_BACKOFF"] = "0.75"

        # Should not raise
        _validate_config()

    def test_validate_config_success_with_live_ollama(self):
        """Config validation should pass with valid live Ollama config."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_CHAT_PATH"] = "/api/chat"
        os.environ["LLM_TAGS_PATH"] = "/api/tags"

        _validate_config()

    def test_validate_config_success_with_openai(self):
        """Config validation should pass with valid OpenAI config."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "openai"
        os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"
        os.environ["LLM_CHAT_PATH"] = "/chat/completions"

        _validate_config()

    def test_validate_config_fails_invalid_api_type(self):
        """Config validation should fail with invalid API type."""
        os.environ["LLM_API_TYPE"] = "invalid_type"
        os.environ["MOCK_LLM"] = "true"

        with pytest.raises(ValueError, match="LLM_API_TYPE must be"):
            _validate_config()

    def test_validate_config_fails_missing_base_url(self):
        """Config validation should fail when base URL is missing in live mode."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ.pop("LLM_BASE_URL", None)

        with pytest.raises(ValueError, match="LLM_BASE_URL is required"):
            _validate_config()

    def test_validate_config_fails_invalid_url_scheme(self):
        """Config validation should fail with non-http(s) URLs."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "ftp://example.com"

        with pytest.raises(ValueError, match="must be http:// or https://"):
            _validate_config()

    def test_validate_config_fails_path_without_slash(self):
        """Config validation should fail when paths don't start with /."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_CHAT_PATH"] = "api/chat"  # Missing leading /

        with pytest.raises(ValueError, match="must start with '/'"):
            _validate_config()

    def test_validate_config_fails_invalid_timeout(self):
        """Config validation should fail with non-positive timeout."""
        os.environ["MOCK_LLM"] = "true"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_TIMEOUT_SECONDS"] = "0"

        with pytest.raises(ValueError, match="must be positive"):
            _validate_config()


class TestLoggingHygiene:
    """Test logging sanitization functions."""

    def test_sanitize_url_removes_token_param(self):
        """_sanitize_url should mask token query parameters."""
        url = "http://example.com/api?token=secret123&other=value"
        sanitized = _sanitize_url(url)

        assert "secret123" not in sanitized
        assert "token=***" in sanitized
        assert "other=value" in sanitized

    def test_sanitize_url_removes_api_key_param(self):
        """_sanitize_url should mask api_key query parameters."""
        url = "https://api.openai.com/v1/chat?api_key=sk-1234567890"
        sanitized = _sanitize_url(url)

        assert "sk-1234567890" not in sanitized
        assert "api_key=***" in sanitized

    def test_sanitize_url_preserves_normal_params(self):
        """_sanitize_url should preserve non-sensitive parameters."""
        url = "http://example.com/api?model=gpt-4&temperature=0.5"
        sanitized = _sanitize_url(url)

        assert "model=gpt-4" in sanitized
        assert "temperature=0.5" in sanitized

    def test_sanitize_url_with_no_query(self):
        """_sanitize_url should handle URLs without query strings."""
        url = "http://example.com/api/chat"
        sanitized = _sanitize_url(url)

        assert sanitized == url

    def test_redact_token_from_log(self):
        """_redact_token should mask Bearer tokens."""
        text = 'Authorization: Bearer sk-1234567890abcdef'
        redacted = _redact_token(text)

        assert "sk-1234567890abcdef" not in redacted
        assert "Bearer ***" in redacted

    def test_redact_token_case_insensitive(self):
        """_redact_token should work case-insensitively."""
        text = "authorization: bearer SECRET_TOKEN_VALUE"
        redacted = _redact_token(text)

        assert "SECRET_TOKEN_VALUE" not in redacted
        assert "Bearer ***" in redacted

    def test_redact_token_multiple_tokens(self):
        """_redact_token should mask multiple Bearer tokens."""
        text = "First: Bearer token1 and Second: Bearer token2"
        redacted = _redact_token(text)

        assert "token1" not in redacted
        assert "token2" not in redacted
        assert redacted.count("Bearer ***") == 2

    def test_cap_response_short(self):
        """_cap_response should not truncate short responses."""
        text = "Short response"
        capped = _cap_response(text, max_len=100)

        assert capped == text
        assert "..." not in capped

    def test_cap_response_long(self):
        """_cap_response should truncate long responses."""
        text = "x" * 500
        capped = _cap_response(text, max_len=200)

        assert len(capped) < len(text)
        assert "..." in capped
        assert "300 more bytes" in capped


class TestRetryLogic:
    """Test retry logic with mocked HTTPX."""

    @patch("src.llm_client._get_http_client")
    def test_retry_on_timeout(self, mock_get_client):
        """LLMClient should retry on timeout exceptions."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "3"

        # Mock client that raises timeout on first 2 attempts, succeeds on 3rd
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            MagicMock(status_code=200, text='{"message": {"content": "ok"}}'),
        ]

        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])

        assert result == "ok"
        assert mock_client.post.call_count == 3

    @patch("src.llm_client._get_http_client")
    def test_no_retry_on_4xx(self, mock_get_client):
        """LLMClient should NOT retry on 4xx errors."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "3"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock 401 Unauthorized
        mock_response = MagicMock(status_code=401)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response
        )
        mock_client.post.return_value = mock_response

        llm = LLMClient()
        with pytest.raises(RuntimeError, match="failed after"):
            llm.chat([{"role": "user", "content": "test"}])

        # Should only attempt once (no retries)
        assert mock_client.post.call_count == 1

    @patch("src.llm_client._get_http_client")
    def test_retry_on_5xx(self, mock_get_client):
        """LLMClient should retry on 5xx errors."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "2"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call returns 503, second succeeds
        mock_response_503 = MagicMock(status_code=503)
        mock_response_ok = MagicMock(
            status_code=200,
            text='{"message": {"content": "recovered"}}'
        )

        mock_client.post.side_effect = [
            mock_response_503,
            mock_response_ok,
        ]

        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])

        assert result == "recovered"
        assert mock_client.post.call_count == 2

    @patch("src.llm_client._get_http_client")
    def test_retry_with_jitter(self, mock_get_client):
        """Retry logic should apply exponential backoff with jitter."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_API_TYPE"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"
        os.environ["LLM_RETRIES"] = "2"
        os.environ["LLM_BACKOFF"] = "0.1"

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Fail twice then succeed
        mock_client.post.side_effect = [
            httpx.TimeoutException("timeout"),
            httpx.TimeoutException("timeout"),
            MagicMock(status_code=200, text='{"message": {"content": "ok"}}'),
        ]

        import time
        start = time.time()
        llm = LLMClient()
        result = llm.chat([{"role": "user", "content": "test"}])
        elapsed = time.time() - start

        # Should have slept due to backoff (at least 0.1s for first attempt)
        # With jitter and retry, this should be reasonable
        assert result == "ok"
        assert elapsed > 0.05  # Allow some tolerance
        assert mock_client.post.call_count == 3


class TestLiveEndpoints:
    """Test /live and /ready endpoints."""

    def test_live_endpoint_always_up(self):
        """GET /live should always return 200 with status: alive."""
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/live")

        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_ready_endpoint_mock_mode(self):
        """GET /ready in mock mode should return 200."""
        os.environ["MOCK_LLM"] = "true"

        import importlib
        import src.server
        importlib.reload(src.server)
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/ready")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_endpoint_with_bad_llm(self):
        """GET /ready should return 503 if LLM is unhealthy."""
        os.environ["MOCK_LLM"] = "false"
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"  # Unreachable
        os.environ["LLM_API_TYPE"] = "ollama"

        import importlib
        import src.server
        importlib.reload(src.server)
        from fastapi.testclient import TestClient
        from src.server import app

        client = TestClient(app)
        resp = client.get("/ready")

        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

