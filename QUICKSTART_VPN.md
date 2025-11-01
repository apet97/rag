# RAG Quickstart - From Zero to Running (VPN Device)

**Complete setup guide for VPN-enabled company device**

---

## 📋 Prerequisites

Before starting, ensure you have:
- [ ] Python 3.10 or 3.11 installed
- [ ] Git installed
- [ ] Connected to company VPN (for LLM access)
- [ ] Terminal/command line access

**Check versions:**
```bash
python3 --version    # Should show 3.10.x or 3.11.x
git --version        # Should show git version
```

---

## 🚀 Complete Setup (5 Minutes)

### Step 1: Clone the Repository

```bash
# Navigate to where you want the project
cd ~/Projects  # Or your preferred location

# Clone the repository
git clone https://github.com/apet97/rag.git

# Enter the directory
cd rag
```

**Verify:**
```bash
ls -la  # Should see: README.md, requirements.txt, src/, etc.
```

---

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# You should see (.venv) in your prompt
```

**Expected output:**
```
(.venv) your-name@hostname:~/Projects/rag$
```

---

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt -c constraints.txt
```

**This takes ~2 minutes.** You'll see packages installing.

**Expected final output:**
```
Successfully installed fastapi-0.110.0 uvicorn-0.27.1 httpx-0.27.0 ...
```

---

### Step 4: Configure Environment

```bash
# Copy example config to .env
cp .env.example .env

# (Optional) Edit .env if you want to customize
# nano .env
```

**Default settings work perfectly for VPN!** No editing needed.

**What's already configured:**
```
LLM_BASE_URL=http://10.127.0.192:11434  ✓ VPN endpoint
LLM_MODEL=gpt-oss:20b                   ✓ Model name
API_PORT=7001                           ✓ Server port
NAMESPACE=clockify                      ✓ Corpus name
```

---

### Step 5: Build Search Index

```bash
# Build the FAISS index from Clockify corpus
make ingest_v2
```

**This takes ~1-2 minutes.** You'll see progress:
```
Loading enriched corpus...
Processing 150+ URLs...
Building FAISS index...
✓ Index saved to index/faiss/clockify_url/
```

**Expected output files:**
```
index/faiss/clockify_url/
├── index.bin     ✓ FAISS index
├── meta.json     ✓ Metadata
└── stats.json    ✓ Ingestion stats
```

---

### Step 6: Verify Quality (Optional but Recommended)

```bash
# Run offline evaluation
make offline_eval
```

**Takes ~30 seconds.** Validates index quality.

**Expected output:**
```
Hit@5: 0.87 ✓ (threshold: 0.85)
Metadata coverage: 95% ✓ (threshold: 90%)
All quality gates passed!
```

---

### Step 7: Launch the Server

```bash
# Start the API server
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7001
```

**Server is now running!** 🎉

---

## ✅ Verification Steps

### Step 8: Test in Browser

Open browser and navigate to:
```
http://localhost:7001
```

You should see the RAG UI with:
- Search box
- "Ask a question" input
- Example queries

**Try asking:** "How do I create a project in Clockify?"

---

### Step 9: Test with cURL (New Terminal)

Open a **new terminal** (keep server running in first one):

```bash
# Test 1: Health check
curl http://localhost:7001/healthz

# Expected output:
# {"ok":true,"namespace":"clockify","index_present":true,...}

# Test 2: Search
curl -H 'x-api-token: change-me' \
  'http://localhost:7001/search?q=create%20project&k=5'

# Expected: JSON array with search results

# Test 3: Chat (RAG with citations)
curl -X POST http://localhost:7001/chat \
  -H 'Content-Type: application/json' \
  -H 'x-api-token: change-me' \
  -d '{"query":"How do I create a project?","namespace":"clockify"}'

# Expected: JSON with answer and citations
```

---

### Step 10: Test VPN LLM Connection

```bash
# Direct LLM health check
curl http://10.127.0.192:11434/api/tags

# Expected output:
# {"models":[{"name":"gpt-oss:20b",...}]}
```

**If this fails:** Check VPN connection is active.

---

## 🎯 Quick Reference

### Start Server (After Initial Setup)

```bash
cd ~/Projects/rag  # Or wherever you cloned it
source .venv/bin/activate
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

### Stop Server

Press `Ctrl+C` in the terminal where server is running.

---

## 📝 Complete Command Summary

**Full setup from zero:**

```bash
# 1. Clone
git clone https://github.com/apet97/rag.git
cd rag

# 2. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt

# 3. Configure
cp .env.example .env

# 4. Build index
make ingest_v2

# 5. (Optional) Validate quality
make offline_eval

# 6. Launch
uvicorn src.server:app --host 0.0.0.0 --port 7001

# 7. Test (new terminal)
curl http://localhost:7001/healthz
```

**Access UI:** http://localhost:7001

---

## 🛠️ Troubleshooting

### Issue: `python3: command not found`
**Fix:**
```bash
# Check if Python is installed
which python3
python --version

# Install Python 3.11 if missing (macOS):
brew install python@3.11
```

### Issue: `git: command not found`
**Fix:**
```bash
# Install git (macOS):
brew install git

# Or download from: https://git-scm.com/download
```

### Issue: `pip install` fails with "no module named pip"
**Fix:**
```bash
# Recreate venv with pip
python3 -m venv .venv --clear
source .venv/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip
```

### Issue: `make: command not found`
**Fix:**
```bash
# Install make (macOS):
xcode-select --install

# Or run commands manually:
python3 tools/ingest_v2.py  # Instead of make ingest_v2
```

### Issue: VPN LLM not reachable
**Symptom:** Timeout on `curl http://10.127.0.192:11434/api/tags`

**Fix:**
```bash
# 1. Check VPN is connected
# 2. Ping the LLM server
ping 10.127.0.192

# 3. If still unreachable, use mock mode:
export MOCK_LLM=true
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

### Issue: Port 7001 already in use
**Fix:**
```bash
# Use different port
uvicorn src.server:app --host 0.0.0.0 --port 7002

# Then access at http://localhost:7002
```

### Issue: Import errors when starting server
**Fix:**
```bash
# Ensure venv is activated (should see (.venv) in prompt)
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt -c constraints.txt
```

---

## ⚡ Alternative: One-Script Setup

After cloning, you can use the automated script:

```bash
cd rag
./scripts/run_local.sh
```

This script does steps 2-7 automatically!

---

## 📊 What You Get

After setup, you'll have:

- ✅ **API Server** running on http://localhost:7001
- ✅ **Web UI** at http://localhost:7001
- ✅ **Search Endpoint** at `/search`
- ✅ **RAG Chat Endpoint** at `/chat`
- ✅ **Health Checks** at `/healthz` and `/health`
- ✅ **Metrics** at `/metrics` (Prometheus format)
- ✅ **150+ Clockify docs** indexed and searchable
- ✅ **VPN LLM** connected at http://10.127.0.192:11434

---

## 🎁 Bonus: Shell Aliases

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# RAG shortcuts
alias rag-cd="cd ~/Projects/rag"  # Change path to match yours
alias rag-start="cd ~/Projects/rag && source .venv/bin/activate && uvicorn src.server:app --host 0.0.0.0 --port 7001"
alias rag-test="curl http://localhost:7001/healthz"
alias rag-logs="tail -f ~/Projects/rag/logs/api.log"
```

Then:
```bash
source ~/.zshrc  # Reload config
rag-start        # Launch RAG from anywhere!
```

---

## 📚 Next Steps

### Learn More
- **Full Documentation:** `codex/RUNBOOK_v2.md`
- **VPN Testing:** `VPN_SMOKE.md`
- **Architecture:** `codex/HANDOFF_NEXT_AI.md`
- **API Reference:** Visit http://localhost:7001/docs (when running)

### Advanced Usage
```bash
# Run tests
pytest tests/

# Rebuild index
make ingest_v2

# Check quality gates
make offline_eval

# View logs
tail -f logs/api.log
tail -f logs/rag_metrics.jsonl
```

---

## ✅ Success Checklist

- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] `.env` file created
- [ ] Index built successfully
- [ ] Offline eval passed
- [ ] Server starts without errors
- [ ] Browser UI loads at http://localhost:7001
- [ ] `/healthz` returns `{"ok":true}`
- [ ] Search returns results
- [ ] VPN LLM responds at `/api/tags`
- [ ] Chat endpoint generates answers

**All checked?** You're production-ready! 🚀

---

## 🆘 Need Help?

**Common issues:**
1. **Dependencies fail:** Use `./scripts/run_local.sh` for auto-fallback
2. **VPN not working:** Set `MOCK_LLM=true` for offline testing
3. **Port conflict:** Change port with `--port 7002`
4. **Index missing:** Run `make ingest_v2` to rebuild

**Still stuck?** Check:
- `LAUNCH_VPN.md` - Detailed launch guide
- `VPN_SMOKE.md` - VPN connectivity tests
- `codex/RUNBOOK_v2.md` - Operations manual

---

**Total setup time:** 5 minutes (mostly dependency installation)

**Enjoy your RAG system!** 🎉
