# RAG Launch Guide - VPN Company Device

## 🚀 Quick Start (30 seconds)

### One Command to Launch
```bash
cd /Users/15x/Downloads/rag && ./scripts/run_local.sh
```

**Access UI**: http://localhost:7001

---

## 📋 What This Does

1. ✅ Activates Python virtual environment
2. ✅ Verifies/installs dependencies
3. ✅ Uses prebuilt index (no rebuild needed)
4. ✅ Connects to VPN LLM: `http://10.127.0.192:11434`
5. ✅ Starts API server on port 7001

**Expected startup time**: ~30 seconds

---

## ✓ Health Checks

After launching, verify everything works:

```bash
# 1. Check server health
curl http://localhost:7001/healthz

# 2. Test search
curl -H 'x-api-token: change-me' \
  'http://localhost:7001/search?q=create%20project&k=5'

# 3. Test VPN LLM (direct)
curl http://10.127.0.192:11434/api/tags
```

**Expected responses**:
- `/healthz` → `{"ok":true,"index_present":true,...}`
- `/search` → JSON array with search results
- `/api/tags` → `{"models":[{"name":"gpt-oss:20b",...}]}`

---

## 🎯 Usage Examples

### Using the UI
1. Open http://localhost:7001 in browser
2. Type question: "How do I create a project in Clockify?"
3. Get AI-powered answer with citations

### Using cURL
```bash
# Search endpoint
curl -H 'x-api-token: change-me' \
  'http://localhost:7001/search?q=timesheet&k=10'

# Chat endpoint (RAG with citations)
curl -X POST http://localhost:7001/chat \
  -H 'Content-Type: application/json' \
  -H 'x-api-token: change-me' \
  -d '{"query":"How do I export a timesheet?","namespace":"clockify"}'
```

---

## 🛠️ Configuration (Optional)

### Default Settings (Already Configured)
```bash
LLM_BASE_URL=http://10.127.0.192:11434  # VPN LLM endpoint
LLM_MODEL=gpt-oss:20b                   # Model name
API_PORT=7001                           # Server port
API_TOKEN=change-me                     # Dev token (change for prod)
NAMESPACE=clockify                      # Default corpus
```

### To Customize
```bash
# Edit .env file
cp .env.example .env
nano .env  # Change settings as needed
```

---

## 🔧 Troubleshooting

### Issue: "Cannot connect to LLM"
**Cause**: VPN disconnected or LLM server unreachable

**Fix**:
```bash
# 1. Check VPN connection
curl http://10.127.0.192:11434/api/tags

# 2. If VPN down, use mock mode
export MOCK_LLM=true
./scripts/run_local.sh
```

### Issue: "Module not found"
**Cause**: Dependencies not installed

**Fix**:
```bash
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

### Issue: "Index not found"
**Cause**: Prebuilt index missing

**Fix**:
```bash
make ingest_v2  # Rebuild index (~2 minutes)
```

### Issue: Port 7001 already in use
**Fix**:
```bash
# Use different port
export API_PORT=7002
uvicorn src.server:app --host 0.0.0.0 --port 7002
```

---

## 🚫 Stopping the Server

```bash
# In terminal where server is running:
Ctrl+C

# Or kill process:
pkill -f "uvicorn src.server:app"
```

---

## 📊 Monitoring

### Logs
```bash
# View API logs
tail -f logs/api.log

# View metrics (JSONL format)
tail -f logs/rag_metrics.jsonl
```

### Prometheus Metrics
Visit: http://localhost:7001/metrics

### Health Dashboard
Visit: http://localhost:7001/health?deep=1&detailed=1

---

## 🎁 Bonus: Shell Alias (Optional)

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Quick RAG commands
alias rag-start="cd /Users/15x/Downloads/rag && ./scripts/run_local.sh"
alias rag-test="curl http://localhost:7001/healthz"
alias rag-logs="tail -f /Users/15x/Downloads/rag/logs/api.log"
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

Usage:
```bash
rag-start  # Launch RAG
rag-test   # Health check
rag-logs   # View logs
```

---

## 📚 Additional Resources

- **Full Documentation**: See `codex/RUNBOOK_v2.md`
- **VPN Smoke Tests**: See `VPN_SMOKE.md`
- **Configuration Reference**: See `.env.example`
- **Architecture**: See `codex/HANDOFF_NEXT_AI.md`

---

## ✅ Success Checklist

After first launch, verify:

- [ ] Server starts without errors
- [ ] `/healthz` returns `{"ok":true}`
- [ ] `/search` endpoint returns results
- [ ] UI loads at http://localhost:7001
- [ ] VPN LLM responds at `/api/tags`
- [ ] Chat endpoint generates answers with citations

**All checks pass? You're ready to go!** 🎉

---

**Need help?** Check `codex/RUNBOOK_v2.md` or `VPN_SMOKE.md`
