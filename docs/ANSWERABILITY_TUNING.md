# Answerability Tuning Guide

## Overview

The answerability check prevents the LLM from hallucinating answers that aren't grounded in your knowledge base. It validates that the LLM's response actually uses information from the retrieved documents.

**How it works:** After the LLM generates an answer, we calculate the **Jaccard similarity** (token overlap) between the answer and the source documents. If the overlap is below the threshold, the answer is rejected and replaced with "I don't have enough information."

```
Answerability Score = (tokens in both answer and context) / (total unique tokens)
```

## Current Configuration

- **Threshold:** 0.18 (18% token overlap required)
- **Location:** `ANSWERABILITY_THRESHOLD` in `.env` or `src/config.py`
- **Historical note:** Lowered from 0.25 to allow more natural paraphrasing

## Symptoms of Incorrect Threshold

### Threshold Too Strict (Current: 0.18)

**Symptoms:**
- LLM says "I don't have enough information" even when docs contain the answer
- Many valid answers get rejected
- Users complain about unhelpful responses
- Log shows: `Answerability check failed (score=0.095)`

**Example from logs:**
```
Chat question: 'how to create a project'
Answerability: 0.095 (threshold=0.18, passed=False)
Original answer: "To create a new project: 1. Go to Projects tab..."
Replaced with: "I don't have enough information"
```

**When this happens:**
- LLM is paraphrasing heavily (good writing, but low token overlap)
- Docs use different terminology than LLM's answer
- Answer is synthesized from multiple sources (each contributes <18%)

### Threshold Too Lenient

**Symptoms:**
- LLM generates plausible-sounding but incorrect answers
- Answers cite sources but don't actually reflect source content
- Hallucinations slip through validation

**Example:**
```
Question: "How do I delete a workspace?"
Answer: "Simply go to Settings > Workspaces > Delete" (score=0.22, passes)
Reality: Docs don't mention deletion, LLM made it up
```

## Tuning Methodology

### Step 1: Measure Current Distribution

Run queries and collect answerability scores:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run test queries
curl -X POST http://localhost:7001/chat \
  -H "Content-Type: application/json" \
  -H "x-api-token: $API_TOKEN" \
  -d '{"question": "how to create a project", "k": 5}'

# Check logs for scores
grep "Answerability:" logs/*.log
```

**Sample output:**
```
Answerability: 0.095 (passed=False)  <- Rejected, but valid answer
Answerability: 0.143 (passed=False)  <- Rejected, but valid answer
Answerability: 0.201 (passed=True)   <- Passed
Answerability: 0.087 (passed=False)  <- Rejected
```

### Step 2: Analyze Rejected Answers

For each rejected answer, manually verify if it was actually correct:

```bash
# Find rejected answers in logs
grep -A5 "Answerability check failed" logs/*.log | less
```

Calculate your **false negative rate:**
```
False Negative Rate = (Valid answers rejected) / (Total rejected)
```

**Target:** <10% false negatives

### Step 3: Adjust Threshold

**Lower threshold IF:**
- False negative rate >10%
- Many good paraphrased answers are rejected
- Your LLM writes in a different style than source docs

**Raise threshold IF:**
- Hallucinations are getting through
- Answers sound plausible but aren't grounded
- Users report incorrect information

**Recommended adjustments:**
- **Conservative:** 0.15 (allows more paraphrasing, some hallucination risk)
- **Current:** 0.18 (balanced)
- **Strict:** 0.22 (prevents most hallucinations, may reject valid answers)

### Step 4: Update Configuration

Edit `.env`:
```bash
ANSWERABILITY_THRESHOLD=0.15
```

Or edit `src/config.py`:
```python
class _Compat:
    ANSWERABILITY_THRESHOLD: float = _parse_float("ANSWERABILITY_THRESHOLD", 0.15)
```

Restart server:
```bash
pkill -f uvicorn
uvicorn src.server:app --host 0.0.0.0 --port 7001
```

### Step 5: Validate

Re-run your test queries and measure:
1. **False negative rate** (should decrease if lowering threshold)
2. **Manual spot-check:** Read 10-20 answers, verify they're grounded
3. **User feedback:** Monitor complaints about incorrect info

## Production Monitoring

Track these metrics:

### 1. Answerability Score Distribution
```python
# In logs, parse scores
scores = [0.095, 0.143, 0.201, ...]
avg_score = mean(scores)  # Target: >0.18
rejection_rate = (scores < 0.18).sum() / len(scores)  # Target: <30%
```

### 2. User Refusal Rate
```sql
SELECT
  COUNT(*) FILTER (WHERE answer LIKE '%don''t have enough information%') / COUNT(*) as refusal_rate
FROM chat_logs
WHERE timestamp > NOW() - INTERVAL '1 day';
```

**Target:** <20% refusal rate

### 3. Manual Quality Audits
- Sample 50 answers/week
- Verify grounding in sources
- Track hallucination rate (target: <5%)

## Advanced: Per-Query-Type Thresholds

Different question types may need different thresholds:

**Factual questions** (who/what/when):
- Strict threshold (0.22) - Must be precise

**How-to questions** (how/why):
- Lenient threshold (0.15) - Allow paraphrasing

**Implementation:**
```python
# In src/llm_client.py
if query_type == "factual":
    threshold = 0.22
elif query_type == "how_to":
    threshold = 0.15
else:
    threshold = CONFIG.ANSWERABILITY_THRESHOLD
```

## Troubleshooting

### "I don't have enough information" for obvious questions

**Cause:** Threshold too strict or documents use different terminology

**Fix:**
1. Lower threshold to 0.15
2. Check if retrieved docs actually contain the answer:
   ```bash
   curl "http://localhost:7001/search?q=how+to+create+project&k=5"
   ```
3. If docs are retrieved but mismatch terminology, consider:
   - Adding synonym expansion in query_expand.py
   - Tuning SEARCH_LEXICAL_WEIGHT (current: 0.35)

### Hallucinations passing through

**Cause:** Threshold too lenient or LLM copying keywords without understanding

**Fix:**
1. Raise threshold to 0.22
2. Enable citation validation (already active)
3. Check LLM temperature (should be low, ~0.1)

### Scores consistently <0.10

**Cause:** LLM is not using retrieved context at all

**Fix:**
1. Check prompt engineering in `src/prompt.py`
2. Verify context is being passed to LLM:
   ```bash
   grep "Retrieved chunks:" logs/*.log
   ```
3. Increase MAX_CONTEXT_CHUNKS if too little context

## Quick Reference

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| **0.12** | Creative/conversational | High paraphrasing freedom, risk of hallucination |
| **0.15** | How-to guides, tutorials | Balanced for instructional content |
| **0.18** | General purpose (current) | Conservative default |
| **0.22** | Factual/legal/medical | Strict grounding, may reject valid paraphrases |
| **0.25** | High-stakes (previous) | Very strict, many false negatives |

## See Also

- [Retrieval Tuning Guide](RETRIEVAL_TUNING_GUIDE.md) - Improve document retrieval
- [Parameter Reference](../README.md#configuration) - All config options
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
