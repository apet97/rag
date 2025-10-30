# Work Mac usage (VPN required)
- Do not assume RAG/Ollama availability.
- Default .env: ALLOW_NETWORK=0, ENABLE_STARTUP_PROBES=0, empty OLLAMA_BASE.
- When on VPN: set OLLAMA_BASE=http://10.127.0.192:11434 and ALLOW_NETWORK=1.
- All network calls must include timeouts and graceful messages.
