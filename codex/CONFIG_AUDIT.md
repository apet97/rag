# Config Audit

This document enumerates environment/config variables discovered, with defaults, usage files, and notes.

| VAR | default | files | required? | notes |
| --- | --- | --- | --- | --- |
| LLM_BASE_URL | http://10.127.0.192:11434 (server), http://localhost:11434 (ingest), http://ollama:11434 (Docker) | src/server.py, src/embeddings_async.py, src/ingest_from_jsonl.py, Dockerfile, README.md, install.sh | yes (for live LLM) | Ollama-compatible; endpoint unreachable in this environment (see ENDPOINT_CHECK.log). |
| LLM_MODEL | gpt-oss-20b | README.md, install.sh, src/llm/local_client.py | no | For Ollama/vLLM; can vary. |
| LLM_API_TYPE | ollama | src/llm_client.py, README.md | no | Supported: ollama, openai. |
| LLM_TEMPERATURE | 0.0 | src/server.py, config docs | no | Determinism default. |
| LLM_TIMEOUT_SECONDS | 30 | src/config.py | no | LLM request timeout. |
| LLM_RETRIES | 2–3 | code_md tests, install.sh | no | Retries/backoff in LLM client. |
| EMBEDDINGS_BACKEND | real/model | src/embeddings.py, src/rerank.py, README.md | no | Set to `stub` for CI/testing (deterministic). |
| EMBEDDING_MODEL | intfloat/multilingual-e5-base | src/embeddings.py, src/embed.py, src/ingest_from_jsonl.py, Dockerfile | yes (for real backend) | Used by SentenceTransformer and metadata. |
| EMBEDDING_DIM | 768 | src/embeddings.py, src/embeddings_async.py (now imports from embeddings), server health | yes | Single source of truth is src/embeddings.EMBEDDING_DIM. |
| STUB_EMBEDDING_DIM | 384 | src/embeddings.py | no | Only for `EMBEDDINGS_BACKEND=stub`. |
| EMBEDDING_BATCH_SIZE | 32 | src/embeddings_async.py, src/embed.py | no | Batch size for embedding operations. |
| MAX_CONTEXT_CHUNKS | 8 | src/config.py | no | Max chunks in prompt. |
| CONTEXT_CHAR_LIMIT | 1200 | src/config.py | no | Character budget per chunk. |
| ANSWERABILITY_THRESHOLD | 0.18 | src/config.py | no | Jaccard overlap threshold guard. |
| LOG_LEVEL | INFO | src/config.py | no | Logging level. |
| LOG_FILE | (unset) | src/config.py | no | Optional log sink. |
| API_TOKEN | change-me | src/server.py, README.md | yes | Required for all endpoints. Must not be `change-me` in prod. |
| API_HOST | 0.0.0.0 | src/server.py | no | FastAPI bind host. |
| API_PORT | 7000 | src/server.py | no | FastAPI port. |
| RETRIEVAL_K | 20 | src/server.py | no | Default K for retrieval. |
| RAG_INDEX_ROOT | index/faiss | src/server.py, tests | yes | Location of prebuilt indexes. |
| NAMESPACES | auto-derived or `clockify` fallback | src/server.py, Dockerfile, install.sh | no | Auto-detects from `RAG_INDEX_ROOT`; Docker default is `clockify`. |
| INDEX_MODE | single | src/server.py | no | Reserved for future multi-index modes. |
| CORS_ALLOWED_ORIGINS | defaults (localhost, etc.) | src/server.py | yes (prod) | No wildcards; must be explicit. |
| ENV | dev | src/server.py | no | If `prod`, enforces API_TOKEN guard. |
| MOCK_LLM | false | src/server.py, src/llm_client.py | no | Skip live LLM and use mock flow. |
| RERANK_DISABLED | false/0 | src/rerank.py, src/server.py | no | Disables FlagEmbedding reranker. Auto-disabled in stub mode. |
| SEMANTIC_CACHE_MAX_SIZE | 10000 | src/semantic_cache.py, README.md | no | Max entries in semantic cache. |
| SEMANTIC_CACHE_TTL_SECONDS | 3600 | src/semantic_cache.py, README.md | no | TTL for cache entries. |
| JSONL_PATH | clockify-help/clockify_help.jsonl | src/ingest_from_jsonl.py | no | Input for legacy ingestion script. |
| INDEX_DIR | index/faiss or index/faiss/clockify-improved | src/embed.py, src/ingest_from_jsonl.py | yes (when running those scripts) | Output dir for builders. |
| BATCH_SIZE | 32 | src/ingest_from_jsonl.py | no | Batch size for ingest embeddings. |
| GLOSSARY_PATH | data/glossary.csv | src/glossary.py | no | Optional glossary for expansion. |
| PARENT_CHUNK_TOKENS | 3200 | src/chunk.py | no | Parent chunk target tokens. |
| PARENT_CHUNK_OVERLAP_TOKENS | 240 | src/chunk.py | no | Parent overlap. |
| CHILD_CHUNK_TOKENS | 640 | src/chunk.py | no | Child chunk tokens. |
| CHILD_CHUNK_OVERLAP_TOKENS | 140 | src/chunk.py | no | Child overlap. |
| CHILD_CHUNK_MIN_TOKENS | 100 | src/chunk.py | no | Minimum child size. |

Notes:
- embeddings_async previously parsed `EMBEDDING_DIM` independently; patched to import from `src.embeddings` to prevent drift.
- index_manager previously hard-raised on reconstruct; patched to degrade gracefully without cached embeddings.
- ingest_from_jsonl uses `/api/embed` with `{input: ...}` payload; Ollama typically expects `/api/embeddings` with `{model, prompt}`. Keep in mind if you use that script.

Safe example `.env` (written to codex/.env.example):

LLM_BASE_URL=http://10.127.0.192:11434
LLM_MODEL=gpt-oss-20b
LLM_API_TYPE=ollama
TOP_K=5
NAMESPACES=clockify
EMBEDDINGS_BACKEND=real
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768
JSON_MODE=false
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
INDEX_PATH=./index/faiss
DATA_DIR=./data
MAX_TOKENS=1024
API_TOKEN=change-me
RERANK_DISABLED=false
SEMANTIC_CACHE_MAX_SIZE=10000
SEMANTIC_CACHE_TTL_SECONDS=3600

