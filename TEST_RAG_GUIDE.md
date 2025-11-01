# RAG Testing Guide - Bypass HTTP Server Issues

If you're experiencing HTTP middleware errors (500 Internal Server Error), use this direct testing script instead.

## Quick Start

```bash
# Navigate to rag directory
cd ~/rag

# Activate virtual environment
source .venv/bin/activate

# Run test with default query
python test_rag.py

# Run test with custom query
python test_rag.py "How do I track time in Clockify?"
```

## What It Does

The `test_rag.py` script:
1. ✅ **Bypasses HTTP server** - No middleware issues
2. ✅ **Tests search directly** - Uses FAISS index directly
3. ✅ **Shows relevance scores** - See how well docs match
4. ✅ **Optionally tests LLM** - Full RAG pipeline with answer generation

## Expected Output

```
================================================================================
RAG DIRECT TEST (Bypassing HTTP Server)
================================================================================

📝 Query: How do I create a project in Clockify?

🔧 Initializing...
✓ Embedding model loaded
✓ Index manager loaded

🔍 Embedding query...
✓ Query embedded (dimension=768)

🔎 Searching index (namespace=clockify_url)...
✓ Search complete: Found 5 results

📊 SEARCH RESULTS
================================================================================

Result #1
  Score: 0.8234
  URL: https://clockify.me/help/projects/create-project
  Title: How to Create a Project in Clockify
  Preview: To create a project, navigate to the Projects page...

Result #2
  Score: 0.7891
  URL: https://clockify.me/help/projects/project-settings
  Title: Project Settings and Configuration
  Preview: Configure your project settings including billable rates...

...

================================================================================
✅ TEST COMPLETE
================================================================================
```

## Usage Examples

### Test Search Only

```bash
python test_rag.py "timesheet export"
```

### Test Full RAG (Search + LLM)

```bash
python test_rag.py "How do I export timesheets?"
# When prompted: y
```

### Custom Queries

```bash
# Single word
python test_rag.py tracking

# Phrase
python test_rag.py "client billing rates"

# Question
python test_rag.py "What are workspace settings?"
```

## Troubleshooting

### Error: "No module named 'src'"

```bash
# Ensure you're in the rag directory
cd ~/rag
pwd  # Should show: /Users/aleksandar/rag

# Ensure venv is activated
source .venv/bin/activate
```

### Error: "Index not found"

```bash
# Rebuild index
python3 tools/ingest_v2.py

# Verify index exists
ls -lah index/faiss/clockify_url/
# Should show: index.bin, meta.json
```

### Error: "No such file or directory: 'index/faiss/clockify/meta.json'"

**Cause:** Namespace mismatch - your `.env` has wrong `NAMESPACES` value.

**Quick fix:**
```bash
# Automated fix
./FIX_NAMESPACES.sh

# Or manual fix
sed -i '' 's/NAMESPACES=.*/NAMESPACES=clockify_url/' .env

# Verify
grep NAMESPACES .env
# Should show: NAMESPACES=clockify_url

# Check what index directories exist
ls -1 index/faiss/
# Should match your NAMESPACES value
```

**Why this happens:** The index is built with namespace `clockify_url`, but your `.env` might have `NAMESPACES=clockify` or `NAMESPACES=clockify,langchain`. They must match exactly.

### Error: "Embedding model not found"

```bash
# Check if model is cached
ls -lah ~/.cache/huggingface/hub/

# If missing, download will happen automatically (requires internet)
# Or use stub embeddings:
export EMBEDDINGS_BACKEND=stub
python test_rag.py
```

### LLM Test Fails

```bash
# Check VPN connection
curl http://10.127.0.192:11434/api/tags

# Or use mock mode
export MOCK_LLM=true
python test_rag.py
```

## Advantages Over HTTP Server

| Feature | HTTP Server | test_rag.py |
|---------|-------------|-------------|
| **Middleware bugs** | ❌ Affected | ✅ Bypassed |
| **Setup complexity** | High (CORS, auth, etc) | Low (just run) |
| **Speed** | Slower (HTTP overhead) | Faster (direct) |
| **Debugging** | Hard (async stack traces) | Easy (simple stack) |
| **Dependencies** | Needs Starlette/FastAPI compat | Just core libs |

## When to Use

**Use `test_rag.py` when:**
- HTTP server has middleware errors
- Quick testing without server setup
- Debugging search relevance
- Batch testing multiple queries
- CI/CD pipeline testing

**Use HTTP server when:**
- Need web UI
- Integrating with external apps
- Production deployment
- Need API endpoints

## Advanced Usage

### Batch Testing

```bash
# Create a test file
cat > queries.txt << EOF
How do I create a project?
Export timesheet to Excel
Track billable hours
Workspace settings
EOF

# Run tests
while read query; do
  echo "Testing: $query"
  python test_rag.py "$query"
  echo ""
done < queries.txt
```

### JSON Output

```python
# Modify test_rag.py to output JSON
import json

results_data = []
for i, (doc_id, score) in enumerate(results, 1):
    meta = mgr.get_metadata(namespace, doc_id)
    results_data.append({
        "rank": i,
        "score": float(score),
        "url": meta.get('url'),
        "title": meta.get('title')
    })

print(json.dumps(results_data, indent=2))
```

### Python Integration

```python
# Use in your own Python code
from test_rag import test_search, test_chat

# Search only
test_search("How do I track time?", k=10)

# Full RAG
test_chat("What are project templates?")
```

## See Also

- **FIX_PYTHON313.sh** - Fix Python version issues
- **PYTHON_VERSION_GUIDE.md** - Python compatibility guide
- **QUICKSTART_VPN.md** - Full setup guide
- **VPN_SMOKE.md** - VPN connectivity tests

---

**Summary:** Use `test_rag.py` when HTTP server has issues. It works identically but bypasses problematic middleware.
