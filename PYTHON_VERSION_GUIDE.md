# Python Version Compatibility Guide

## ⚠️ Python 3.13 Issues (macOS ARM64)

**Problem:** Python 3.13 is too new for some pinned dependencies.

**Error you'll see:**
```
ERROR: Cannot install faiss-cpu==1.7.4 because these package versions have conflicting dependencies.
Additionally, some packages in these conflicts have no matching distributions available for your environment:
  faiss-cpu
```

**Root cause:** faiss-cpu 1.7.4 doesn't have pre-built wheels for Python 3.13 on macOS ARM64.

---

## ✅ Supported Python Versions

| Python Version | Status | Notes |
|----------------|--------|-------|
| **3.12** | ✅ **Recommended** | Best compatibility, production-tested |
| **3.11** | ✅ **Fully Supported** | CI tested, official runtime.txt |
| **3.10** | ✅ Supported | Works but less tested |
| **3.13** | ⚠️ **Not Compatible** | faiss-cpu wheels missing |
| **3.9** | ⚠️ Deprecated | May work but not tested |

---

## 🔧 Quick Fix (If You Hit Python 3.13 Error)

### Option 1: Automated Fix Script (Easiest)

```bash
cd ~/rag
./FIX_PYTHON313.sh
```

This script will:
1. Install Python 3.12 via Homebrew
2. Remove old .venv
3. Create new venv with Python 3.12
4. Optionally run the full setup

### Option 2: Manual Fix

```bash
# Install Python 3.12
brew install python@3.12

# Remove old venv
cd ~/rag
rm -rf .venv

# Create new venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Verify version
python --version  # Should show Python 3.12.x

# Install and run
./scripts/run_local.sh
```

---

## 🎯 How to Check Your Python Version

```bash
# Check system Python
python3 --version

# Check specific version
python3.12 --version
python3.11 --version

# Check active venv Python
source .venv/bin/activate
python --version
```

---

## 🐛 Why Does This Happen?

**Python Package Wheels:**
- Python packages distribute pre-compiled binaries called "wheels"
- Wheels are built for specific Python versions and platforms
- faiss-cpu 1.7.4 was released before Python 3.13 existed
- No Python 3.13 wheels → pip tries to compile from source → fails

**Why Not Just Upgrade faiss-cpu?**
- Newer faiss-cpu (1.8+) has Python 3.13 wheels
- BUT: Untested with this project
- Risk of API changes or subtle bugs
- Dependencies pinned for stability

---

## 🔄 Alternative: Use Stub Embeddings (Advanced)

**If you can't/won't change Python version:**

```bash
# Install minimal deps (no faiss-cpu)
source .venv/bin/activate
pip install fastapi uvicorn httpx orjson loguru python-dotenv rank-bm25 prometheus-client

# Use stub embeddings (no semantic search)
export EMBEDDINGS_BACKEND=stub
python3 tools/ingest_v2.py
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

**Trade-offs:**
- ❌ No semantic search (hash-based random embeddings)
- ❌ Poor RAG quality
- ✅ API works for testing
- ✅ No Python version change needed

**Use cases:**
- Quick API structure testing
- CI/CD pipelines
- Completely offline environments
- When you just need the server running

---

## 📋 Verification After Fix

```bash
# 1. Check Python version in venv
source .venv/bin/activate
python --version
# Expected: Python 3.12.x or 3.11.x

# 2. Try installing dependencies
pip install -r requirements.txt -c constraints.txt
# Should succeed without errors

# 3. Verify FAISS imports
python -c "import faiss; print('FAISS OK')"
# Expected: FAISS OK
```

---

## 💡 Pro Tips

### Always Specify Python Version

```bash
# Bad (uses system default, might be 3.13)
python3 -m venv .venv

# Good (explicit version)
python3.12 -m venv .venv
```

### Check Before Creating Venv

```bash
# Check what python3 points to
python3 --version

# If it's 3.13, use explicit version
python3.12 -m venv .venv
```

### Multiple Python Versions on macOS

```bash
# List installed Python versions
ls -la /usr/local/bin/python3*
ls -la /opt/homebrew/bin/python3*

# Install specific version
brew install python@3.12
brew install python@3.11

# Use specific version
python3.12 --version
python3.11 --version
```

---

## 🆘 Still Having Issues?

### Issue: "python3.12: command not found"

```bash
# Install Python 3.12
brew install python@3.12

# If Homebrew not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Issue: Dependencies still fail with Python 3.12

```bash
# Use the robust fallback script
./scripts/run_local.sh

# This has 4-tier fallback strategy:
# Tier 1: Try exact versions
# Tier 2: Try platform-compatible versions
# Tier 3: Strip optional deps (lxml, etc)
# Tier 4: Minimal runtime install
```

### Issue: Can't install Homebrew / No sudo access

**Use stub embeddings mode** (see Alternative section above)

---

## 📚 References

- Official runtime: `runtime.txt` (specifies Python 3.11.9)
- CI configuration: `.github/workflows/rag-ci.yml` (uses Python 3.11)
- Bootstrap script: `scripts/bootstrap.sh` (recommends Python 3.11+)
- Fallback strategy: `scripts/run_local.sh` (multi-tier installation)

---

**Summary:** Use Python 3.12 or 3.11 for best results. Avoid Python 3.13 until dependencies catch up.
