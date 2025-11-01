# RAG Documentation Index

## Quick Navigation

**First time here?** Start with [Getting Started](#getting-started)

**Tuning the system?** See [Configuration & Tuning](#configuration--tuning)

**Production deployment?** Check [Operations](#operations)

---

## Getting Started

### [Quick Start](../README.md#quickstart)
Get running in 3 commands. Start here if you're new to the project.

### Setup & Deployment
- **[Main README](../README.md)** - Project overview, features, architecture
- **[Deployment Guide](../DEPLOY.md)** - Production deployment instructions
- **[VPN Setup](../VPN_SMOKE.md)** - Connect to internal LLM (for Clockify users)

---

## Configuration & Tuning

### Core Tuning Guides

#### **[Answerability Tuning](ANSWERABILITY_TUNING.md)** ⭐ CRITICAL
Fix "I don't have enough information" responses. Learn when and how to adjust the threshold to balance hallucination prevention vs answer quality.

**Use this guide when:**
- LLM refuses to answer despite having correct docs
- You see `Answerability check failed` in logs
- Users complain about unhelpful responses

#### **[Retrieval Tuning Guide](RETRIEVAL_TUNING_GUIDE.md)** ⭐ CRITICAL
Optimize search quality. Learn to balance semantic vs keyword matching for your use case.

**Use this guide when:**
- Similar queries return different results
- Search misses obvious documents
- Want to improve precision or recall

#### **[Add New Documents](ADD_NEW_DOCUMENTS.md)** ⭐ CRITICAL
Expand your knowledge base. Step-by-step guide to adding URLs, running ingestion, and validating results.

**Use this guide when:**
- Need to index new documentation
- Update existing content
- Add a new knowledge domain

### Reference Documentation

#### [Parameter Reference](../README.md#configuration)
Complete list of all environment variables and their defaults.

#### [API Reference](../README.md#api-endpoints)
HTTP endpoint schemas, request/response formats, authentication.

---

## Operations

### [Runbook](../codex/RUNBOOK_v2.md)
Production operations guide. Health checks, monitoring, incident response.

### [Troubleshooting](TROUBLESHOOTING.md)
Common issues and their solutions. Check here first when things go wrong.

### [Performance Monitoring](../docs/PERFORMANCE_MONITORING.md)
Metrics to track, alert thresholds, performance baselines.

---

## Advanced Topics

### [Integration Examples](../docs/INTEGRATION_EXAMPLES.md)
Code samples for integrating RAG into your application.

### [Testing Guide](../TEST_RAG_GUIDE.md)
How to test the RAG system without the HTTP server (useful for debugging).

---

## Architecture & Development

### [Handoff Document](../codex/HANDOFF_NEXT_AI.md)
Technical architecture, design decisions, operational context. Read this to understand how the system works.

### [Improvement Plan](../codex/IMPROVEMENT_PLAN.md)
Roadmap of planned enhancements, prioritized by impact.

### [Contributing](../CONTRIBUTING.md)
Guidelines for contributing to the project.

---

## Decision Trees

### "My RAG system isn't working well"

```
START: What's the problem?

├─ "I don't have enough information" responses?
│  └─ Read: Answerability Tuning Guide
│
├─ Search returns wrong documents?
│  └─ Read: Retrieval Tuning Guide
│
├─ Missing recent documentation?
│  └─ Read: Add New Documents
│
├─ Server errors / crashes?
│  └─ Read: Troubleshooting
│
└─ Slow performance?
   └─ Read: Performance Monitoring
```

### "I want to customize my RAG system"

```
START: What do you want to change?

├─ Different knowledge base?
│  └─ Read: Add New Documents
│
├─ Different LLM model?
│  └─ Read: README > Configuration > LLM Settings
│
├─ Different embedding model?
│  ├─ Read: Add New Documents > Full Rebuild
│  └─ Update: EMBEDDING_MODEL and EMBEDDING_DIM in .env
│
├─ Different chunking strategy?
│  ├─ Read: Add New Documents > Custom Chunking
│  └─ Update: CHUNK_STRATEGY in .env
│
└─ Different search behavior?
   └─ Read: Retrieval Tuning Guide
```

---

## Support

### Getting Help

1. **Check existing docs** - Most questions are answered in this documentation
2. **Search issues** - [GitHub Issues](https://github.com/apet97/rag/issues)
3. **Create issue** - If problem persists, open a detailed issue

### Providing Feedback

Found a bug? Have a feature request? Want to improve docs?

**Open an issue:** https://github.com/apet97/rag/issues/new

**Include:**
- What you were trying to do
- What happened
- What you expected
- Logs (if applicable)
- Configuration (`.env` settings)

---

## Documentation Hierarchy

```
.
├── README.md              # Main entry point
├── DEPLOY.md              # Production deployment
├── CONTRIBUTING.md        # Contribution guidelines
│
├── docs/                  # Comprehensive guides
│   ├── README.md          # This file (navigation)
│   ├── ANSWERABILITY_TUNING.md
│   ├── RETRIEVAL_TUNING_GUIDE.md
│   ├── ADD_NEW_DOCUMENTS.md
│   ├── TROUBLESHOOTING.md
│   └── PERFORMANCE_MONITORING.md
│
└── codex/                 # Operational artifacts
    ├── HANDOFF_NEXT_AI.md  # Technical architecture
    ├── RUNBOOK_v2.md       # Operations guide
    ├── IMPROVEMENT_PLAN.md # Roadmap
    └── ALLOWLIST.txt       # URL policies
```

---

Last updated: 2025-11-01
