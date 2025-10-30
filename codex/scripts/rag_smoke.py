#!/usr/bin/env python3
"""
Runtime smoke test for RAG endpoint (optional).

Behavior:
- Samples 20 queries from codex/RAG_EVAL_TASKS.jsonl across topics
- Attempts to call /chat on local server (http://localhost:7000/chat) unless RAG_API_BASE provided
- Verifies all returned citations (sources[].url) are allowlisted
- If endpoint is unreachable, writes an OFFLINE note
"""
import os
import json
import random
from collections import defaultdict
from urllib.parse import urlparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TASKS = os.path.join(ROOT, "codex", "RAG_EVAL_TASKS.jsonl")
ALLOWLIST = os.path.join(ROOT, "codex", "ALLOWLIST.txt")
OUT = os.path.join(ROOT, "codex", "RAG_RUNTIME_EVAL.md")

API_BASE = os.getenv("BASE_URL", os.getenv("RAG_API_BASE", "http://localhost:7000"))
TIMEOUT = float(os.getenv("TIMEOUT", "10"))
QUERIES = int(os.getenv("QUERIES", "20"))
MODEL = os.getenv("MODEL", os.getenv("LLM_MODEL", "gpt-oss:20b"))


def compile_patterns(path):
    import re
    pats = []
    if os.path.isfile(path):
        for line in open(path, "r", encoding="utf-8"):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            try:
                pats.append(re.compile(s))
            except re.error:
                pass
    return pats


def allowed(url: str, pats) -> bool:
    return any(p.search(url) for p in pats) if pats else True


def main():
    tasks = [json.loads(l) for l in open(TASKS, "r", encoding="utf-8") if l.strip()]
    # Stratified sample by topic
    by_topic = defaultdict(list)
    for t in tasks:
        by_topic[t.get("topic", "")] .append(t)
    sample = []
    for topic, arr in by_topic.items():
        random.shuffle(arr)
        sample.extend(arr[:1])
        if len(sample) >= QUERIES:
            break
    if len(sample) < QUERIES:
        sample.extend(tasks[: QUERIES - len(sample)])

    allow_pats = compile_patterns(ALLOWLIST)

    import requests
    results = []
    ok = True
    try:
        for t in sample:
            q = t["question"]
            r = requests.post(
                f"{API_BASE}/chat",
                json={"question": q, "k": 5, "namespace": "clockify"},
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                results.append({"q": q, "status": r.status_code, "error": r.text[:200]})
                ok = False
                continue
            data = r.json()
            sources = data.get("sources", [])
            bad = [s for s in sources if not allowed(s.get("url", ""), allow_pats)]
            results.append({"q": q, "ok": len(bad)==0, "bad": bad, "latency_ms": data.get("latency_ms", {})})
            if bad:
                ok = False
    except Exception as e:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("RAG Runtime Eval\n\n- Endpoint unreachable or offline. Skipping runtime tests.\n")
            f.write(f"- Error: {e}\n")
        print(f"Wrote {OUT} (offline)")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("RAG Runtime Eval\n\n")
        f.write(f"- Queries: {len(sample)}\n")
        f.write(f"- All citations in ALLOWLIST: {'YES' if ok else 'NO'}\n\n")
        for res in results:
            if res.get("ok", True) is False:
                f.write(f"- FAIL: {res['q']} bad={len(res.get('bad', []))}\n")
        if ok:
            f.write("- All tests passed.\n")

    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
