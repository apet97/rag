# RAG Evaluation Framework

This directory contains tools for evaluating and tracking RAG retrieval quality.

## Key Components

### `run_eval.py` - Core Evaluation Script
Runs the evaluation harness against a goldset of Q&A pairs.

**Basic usage:**
```bash
# Baseline evaluation (decomposition disabled)
python3 eval/run_eval.py --decomposition-off

# With query decomposition enabled
python3 eval/run_eval.py

# JSON output only
python3 eval/run_eval.py --json

# Log decomposition metadata to JSONL
python3 eval/run_eval.py --log-decomposition
```

**Key metrics:**
- `Recall@5`: Percentage of cases where ground truth appears in top-5 results
- `MRR@5`: Mean Reciprocal Rank (average position of first correct result)
- `Answer accuracy`: Whether the LLM answer matches expected output
- `Retrieval latency p50/p95`: Retrieval timing statistics

### `track_eval.py` - Evaluation Tracking & Versioning
Automatically versions evaluation results and performs A/B comparisons.

**Usage:**
```bash
# Run baseline only
python3 eval/track_eval.py --baseline --label "session5c"

# Run with decomposition
python3 eval/track_eval.py --with-decomposition --label "session5c"

# Run both and compare automatically
python3 eval/track_eval.py --both --label "session5c"
```

**Output:**
- Results saved to `logs/evals/` with timestamps
- Latest results symlinked to `*_latest.json`
- JSON format for programmatic analysis
- A/B comparison showing delta in Recall, Accuracy, Latency

### `diagnose_misses.py` - Miss Case Analysis
Categorizes and analyzes failed evaluation cases.

**Usage:**
```bash
# Analyze baseline results
python3 eval/diagnose_misses.py logs/evals/baseline_latest.json

# Analyze with-decomposition results
python3 eval/diagnose_misses.py logs/evals/with_decomposition_latest.json
```

**Output:**
- Miss breakdown by decomposition strategy
- Miss breakdown by query intent (howto, question, comparison, etc.)
- Sample failed cases with retrieved URLs
- Failure pattern extraction (API gaps, generic titles, multi-intent failures)

## Evaluation Results

All evaluation results are stored in `logs/evals/` with automatic versioning:

```
logs/evals/
├── baseline_2025-10-25T19-30-45.json
├── baseline_latest.json -> baseline_2025-10-25T19-30-45.json
├── with_decomposition_2025-10-25T19-32-10.json
└── with_decomposition_latest.json -> with_decomposition_2025-10-25T19-32-10.json
```

## Goldset

The evaluation goldset is defined in `eval/goldset.csv` with columns:
- `id`: Case identifier
- `question`: Query to evaluate
- `ground_truth_urls`: Expected result URLs (pipe-separated)
- `expected_answer`: Expected LLM answer (for answer accuracy eval)

## Metrics Interpretation

### Recall@5 = 0.32 (current baseline)
- Only 8 of 25 test cases have ground truth in top-5 results
- Indicates fundamental retrieval gaps
- Main causes identified:
  - **API vocabulary gaps**: Queries using API-specific terms (webhook, curl)
  - **Generic titles**: Results pointing to index pages rather than specific guides
  - **Multi-intent failures**: Comparison queries not properly decomposed

### Answer Accuracy = 0.36
- LLM answers correct for ~9 of 25 cases
- Directly correlates with retrieval quality (better retrieval → better answers)
- After embedding fix (Session 5c): maintains 0.36 with decomposition (previously dropped to 0.12)

## Next Steps for Improvement

Based on diagnosed failure patterns, prioritized options:

1. **Synonym-heavy glossary expansion** (quick win)
   - Add more API-related synonyms (webhook→event, curl→API request)
   - Reduce generic title hits by adding negative terms
   - Estimated impact: +3-5% recall

2. **Cross-encoder reranking** (medium effort)
   - Use cross-encoder to re-rank top-20 by semantic relevance
   - Demote generic docs, promote specific guides
   - Estimated impact: +5-10% recall

3. **Chunk title rewriting** (higher effort)
   - Rewrite generic chunk titles to be more specific
   - Example: "How to Create" → "How to Create a Project in Clockify"
   - Estimated impact: +10-15% recall

4. **LLM-powered decomposition endpoint** (addresses multi-intent)
   - Wire actual LLM for query decomposition (currently heuristic-only)
   - Better handling of comparison/contrast queries
   - Estimated impact: +5-10% recall for multi-intent queries

## CI Integration

Tests are located in `tests/test_search_chat.py` and validate:
- API response contract (success, total_results, latency_ms, metadata)
- Score normalization ([0, 1] range)
- Decomposition metadata structure
- Latency bounds (< 5s for /search with decomposition)

Run tests with:
```bash
pytest tests/test_search_chat.py -xvs
```
