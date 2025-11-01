# Troubleshooting Guide

## Common Issues

### "I don't have enough information" Responses

**Symptom:** LLM refuses to answer despite having correct documents

**Solution:** See [Answerability Tuning Guide](ANSWERABILITY_TUNING.md)

### Search Returns Wrong Documents

**Symptom:** Similar queries return different results, search misses obvious docs

**Solution:** See [Retrieval Tuning Guide](RETRIEVAL_TUNING_GUIDE.md)

### Server Won't Start

**Symptom:** Error on startup, server crashes immediately

**Common causes:**
1. **Embedding dimension mismatch**
   ```
   Error: Embedding dimension mismatch for namespace 'clockify_url'
   Index built with: dim=768
   Encoder provides: dim=384
   ```
   **Fix:** Update `.env` to match index:
   ```bash
   EMBEDDING_MODEL=intfloat/multilingual-e5-base
   EMBEDDING_DIM=768
   ```

2. **Missing index files**
   ```
   Error: No such file or directory: 'index/faiss/clockify_url/meta.json'
   ```
   **Fix:** Rebuild index:
   ```bash
   python3 tools/ingest_v2.py
   ```

3. **Port already in use**
   ```
   Error: [Errno 48] Address already in use
   ```
   **Fix:** Kill existing process:
   ```bash
   pkill -f uvicorn
   # Or use different port
   uvicorn src.server:app --port 7002
   ```

### Slow Performance

**Symptom:** Search takes >1 second, diversity filter warnings in logs

**Solution:**
- Check logs for "Slow stage: diversity_filter"
- Latest code has performance fix (skips MMR for large result sets)
- Ensure you're on latest main branch

### Chat Returns 429 "Too Many Requests"

**Symptom:** Parallel requests from browser UI fail

**Solution:** Already fixed in latest code (localhost exempt from rate limiting)

### Missing Documents in Search

**Symptom:** Know docs exist but search doesn't find them

**Solutions:**
1. **Check if docs are indexed:**
   ```bash
   python3 -c "
   import json
   with open('index/faiss/clockify_url/meta.json') as f:
       meta = json.load(f)
   print(f'Total docs: {len(meta)}')
   for i, doc in enumerate(meta[:5]):
       print(f'{i}: {doc.get(\"url\", \"N/A\")}')"
   ```

2. **Check ALLOWLIST/DENYLIST:**
   ```bash
   grep "problem-url" codex/ALLOWLIST.txt
   grep "problem-url" codex/DENYLIST.txt
   ```

3. **Rebuild index:**
   ```bash
   python3 tools/ingest_v2.py
   ```

## Getting More Help

1. **Check logs:**
   ```bash
   tail -f logs/server.log
   ```

2. **Enable debug logging:**
   ```bash
   export LOG_LEVEL=DEBUG
   ```

3. **Test without HTTP server:**
   ```bash
   python test_rag.py "your question"
   ```

4. **Open GitHub issue:** https://github.com/apet97/rag/issues

## See Also

- [Answerability Tuning](ANSWERABILITY_TUNING.md)
- [Retrieval Tuning](RETRIEVAL_TUNING_GUIDE.md)
- [Add Documents](ADD_NEW_DOCUMENTS.md)
- [Runbook](../codex/RUNBOOK_v2.md)
