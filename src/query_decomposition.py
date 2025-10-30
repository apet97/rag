"""
Query decomposition for multi-intent and procedural questions.

Decomposes complex queries into focused sub-queries to improve retrieval recall
for comparison, procedural (how-to), and multi-part questions.

Features:
- Heuristic-based decomposition (comparison, multi-part, procedural)
- LLM fallback when heuristics find <=1 subtask (with 750ms timeout)
- Per-subtask intent detection and boost term extraction
- Subtask normalization (punctuation trimming, context reattachment)
- Rich metadata for logging and analysis
"""

import re
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.prompt import RAGPrompt
import time

from loguru import logger

# Import for LLM fallback and query type detection
try:
    from src.llm_client import LLMClient
except ImportError:
    LLMClient = None

try:
    from src.search_improvements import detect_query_type
except ImportError:
    detect_query_type = None


@dataclass
class QuerySubtask:
    """Single subtask from decomposed query."""
    text: str
    reason: str
    weight: float = 1.0
    boost_terms: List[str] = field(default_factory=list)
    intent: Optional[str] = None  # "factual", "how_to", "comparison", "definition", "general"
    llm_generated: bool = False  # Whether this subtask came from LLM fallback

    def to_dict(self):
        return asdict(self)

    def to_log_payload(self) -> Dict[str, Any]:
        """Return enriched payload for logging and analysis."""
        return {
            "text": self.text,
            "reason": self.reason,
            "weight": self.weight,
            "boost_terms": self.boost_terms,
            "intent": self.intent,
            "llm_generated": self.llm_generated,
        }


@dataclass
class QueryDecompositionResult:
    """Result of query decomposition."""
    original_query: str
    subtasks: List[QuerySubtask]
    strategy: str
    timed_out: bool = False
    llm_used: bool = False  # Whether LLM fallback was invoked

    def to_dict(self):
        return {
            "original_query": self.original_query,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "strategy": self.strategy,
            "timed_out": self.timed_out,
            "llm_used": self.llm_used,
        }

    def to_strings(self) -> List[str]:
        """Return subtask texts as list of strings."""
        return [st.text for st in self.subtasks]

    def to_log_payload(self) -> Dict[str, Any]:
        """Return enriched payload for eval logging."""
        return {
            "original_query": self.original_query,
            "strategy": self.strategy,
            "llm_used": self.llm_used,
            "timed_out": self.timed_out,
            "subtask_count": len(self.subtasks),
            "subtasks": [st.to_log_payload() for st in self.subtasks],
        }


def _load_glossary() -> dict:
    """Load glossary from data/domain/glossary.json."""
    path = Path("data/domain/glossary.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load glossary for decomposition: {e}")
            return {}
    return {}


def _extract_boost_terms_from_glossary(query: str, glossary: dict) -> List[str]:
    """Extract glossary terms and their synonyms from query as boost terms."""
    boost_terms = []
    query_lower = query.lower()

    for term, synonyms in glossary.items():
        if term.lower() in query_lower:
            boost_terms.append(term)
            for syn in synonyms:
                syn_clean = syn.strip().lower()
                if syn_clean and syn_clean not in boost_terms:
                    boost_terms.append(syn_clean)

    return boost_terms[:8]  # Cap boost terms


def _detect_comparison_query(query: str) -> Optional[tuple]:
    """
    Detect comparison queries (X vs Y, difference between X and Y).

    Returns:
        (entity1, entity2) if comparison detected, else None
    """
    query_lower = query.lower()

    # Try "X vs Y" or "X versus Y"
    match = re.search(r"([\w\s]+?)\s+(?:vs\.?|versus)\s+([\w\s]+)", query_lower)
    if match:
        return (match.group(1).strip(), match.group(2).strip())

    # Try "difference between X and Y"
    match = re.search(
        r"difference\s+between\s+([\w\s]+?)\s+and\s+([\w\s]+)", query_lower
    )
    if match:
        return (match.group(1).strip(), match.group(2).strip())

    # Try "compare X and Y"
    match = re.search(r"compare\s+([\w\s]+?)\s+and\s+([\w\s]+)", query_lower)
    if match:
        return (match.group(1).strip(), match.group(2).strip())

    return None


def _detect_multi_part_query(query: str) -> Optional[List[str]]:
    """
    Detect multi-part queries with conjunctions or enumerations.

    Returns:
        List of parts if multi-part detected, else None
    """
    query_lower = query.lower()

    # Look for procedural steps: "first...", "then...", "next..."
    steps = []
    for keyword in ["first", "then", "next", "finally", "after"]:
        if keyword in query_lower:
            # Split on the keyword and collect parts
            parts = re.split(rf"\b{keyword}\b", query_lower)
            if len(parts) > 1:
                # Filter out empty parts and recombine context
                for part in parts[1:]:
                    part = part.strip()
                    if part and len(part) > 3:
                        steps.append(part)
    if steps and len(steps) >= 2:
        return steps

    # Look for "and" conjunctions (but not "and" in single clause)
    # e.g., "export timesheets and invoices"
    parts = re.split(r"\band\b", query_lower)
    if len(parts) >= 2:
        # Filter for meaningful parts (>3 chars, not stop words only)
        meaningful = [p.strip() for p in parts if len(p.strip()) > 3]
        if len(meaningful) >= 2:
            return meaningful

    return None


def _normalize_subtask(text: str, head_verb: Optional[str] = None) -> str:
    """
    Normalize a subtask by trimming punctuation and reattaching shared context.

    Args:
        text: Raw subtask text (may have trailing punctuation, incomplete phrases)
        head_verb: Optional head verb to prepend (e.g., "export" for "timesheets, invoices")

    Returns:
        Normalized subtask text
    """
    # Trim trailing punctuation and whitespace
    text = text.rstrip('.,;:!?')
    text = text.strip()

    # Prepend head verb if provided and text doesn't already contain it
    if head_verb and head_verb.lower() not in text.lower():
        text = f"{head_verb} {text}"

    return text


def _extract_head_verb(query: str) -> Optional[str]:
    """
    Extract the main verb from a query for context reattachment.

    Examples:
        "export timesheets and invoices" -> "export"
        "How do I set up workspace then configure time off" -> "set up"
    """
    # Try common head verb patterns
    verbs = [
        "export", "import", "create", "delete", "set up", "configure", "setup",
        "enable", "disable", "add", "remove", "update", "change", "manage",
        "track", "log", "report", "view", "show", "check"
    ]
    query_lower = query.lower()
    for verb in verbs:
        if verb in query_lower:
            return verb
    return None


def _get_subtask_intent(subtask_text: str) -> Optional[str]:
    """
    Detect the query intent for a subtask.

    Returns one of: "factual", "how_to", "comparison", "definition", "general"
    """
    if detect_query_type is None:
        return None

    try:
        detected = detect_query_type(subtask_text)
        return detected
    except Exception:
        return None


def _get_subtask_boost_terms(subtask_text: str, glossary: dict) -> List[str]:
    """
    Extract boost terms specific to a subtask fragment.

    Different from global boost terms extraction - this looks only at the
    subtask's own words, not the entire query.
    """
    boost_terms = []
    subtask_lower = subtask_text.lower()

    for term, synonyms in glossary.items():
        if term.lower() in subtask_lower:
            boost_terms.append(term)
            for syn in synonyms:
                syn_clean = syn.strip().lower()
                if syn_clean and syn_clean not in boost_terms:
                    boost_terms.append(syn_clean)

    return boost_terms[:6]  # Cap per-subtask boost terms lower than global


def _llm_decompose_fallback(
    query: str, timeout_seconds: float = 0.5, max_subtasks: int = 3
) -> Optional[List[str]]:
    """
    LLM-based decomposition fallback for when heuristics fail.

    Asks the LLM to split a complex question into sub-questions.
    Falls back gracefully if LLM is unavailable, in MOCK mode, or times out.

    Returns:
        List of sub-question strings if successful, None otherwise
    """
    if LLMClient is None:
        logger.debug("LLMClient not available for decomposition fallback")
        return None

    # Don't use LLM if in MOCK mode
    if os.getenv("MOCK_LLM", "false").lower() == "true":
        logger.debug("MOCK_LLM enabled, skipping LLM decomposition fallback")
        return None

    try:
        start_time = time.time()
        client = LLMClient()

        # P2: Use centralized RAGPrompt for query decomposition
        system_prompt, user_prompt = RAGPrompt.get_decomposition_prompts(query, max_subtasks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM with timeout
        response = client.chat(messages, max_tokens=200, temperature=0.0, stream=False)
        elapsed = time.time() - start_time

        if elapsed > timeout_seconds:
            logger.warning(
                f"LLM decomposition timed out ({elapsed:.2f}s > {timeout_seconds}s), "
                f"falling back to heuristics"
            )
            return None

        # Parse JSON response
        response_clean = response.strip()
        if response_clean.startswith("[") and response_clean.endswith("]"):
            sub_questions = json.loads(response_clean)
            if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                logger.debug(
                    f"LLM decomposition successful: {len(sub_questions)} sub-questions "
                    f"in {elapsed:.2f}s"
                )
                return sub_questions[:max_subtasks]

        logger.warning(f"LLM returned invalid format: {response_clean[:100]}")
        return None

    except Exception as e:
        logger.debug(f"LLM decomposition fallback failed: {e}, using heuristics")
        return None


def decompose_query(
    query: str, max_subtasks: int = 3, timeout_seconds: float = 0.75
) -> QueryDecompositionResult:
    """
    Decompose query into focused sub-queries.

    Uses heuristics to detect and split comparison, multi-part, and procedural
    questions. Falls back to LLM if heuristics find <=1 subtask.
    Always includes the original query as a subtask.

    Args:
        query: Original query text
        max_subtasks: Maximum number of subtasks to generate
        timeout_seconds: Timeout for decomposition (LLM fallback: 0.5s of 0.75s total)

    Returns:
        QueryDecompositionResult with subtasks and strategy
    """
    start_time = time.time()
    glossary = _load_glossary()
    boost_terms_global = _extract_boost_terms_from_glossary(query, glossary)
    head_verb = _extract_head_verb(query)

    subtasks = []
    strategy = "none"
    llm_used = False

    # Try comparison detection
    comparison = _detect_comparison_query(query)
    if comparison:
        entity1, entity2 = comparison
        # Normalize entities and extract per-subtask boost terms
        entity1_norm = _normalize_subtask(entity1)
        entity2_norm = _normalize_subtask(entity2)

        intent1 = _get_subtask_intent(entity1_norm)
        intent2 = _get_subtask_intent(entity2_norm)
        boost1 = _get_subtask_boost_terms(entity1_norm, glossary)
        boost2 = _get_subtask_boost_terms(entity2_norm, glossary)

        subtasks.append(
            QuerySubtask(
                text=entity1_norm,
                reason="comparison_entity_1",
                weight=1.0,
                boost_terms=boost1 or boost_terms_global,
                intent=intent1,
                llm_generated=False,
            )
        )
        subtasks.append(
            QuerySubtask(
                text=entity2_norm,
                reason="comparison_entity_2",
                weight=1.0,
                boost_terms=boost2 or boost_terms_global,
                intent=intent2,
                llm_generated=False,
            )
        )
        strategy = "comparison"

    # Try multi-part detection
    if not subtasks:
        multi_parts = _detect_multi_part_query(query)
        if multi_parts and len(multi_parts) >= 2:
            for i, part in enumerate(multi_parts[:max_subtasks]):
                # Normalize part and reattach head verb if needed
                part_norm = _normalize_subtask(part, head_verb)
                intent = _get_subtask_intent(part_norm)
                boost = _get_subtask_boost_terms(part_norm, glossary)

                subtasks.append(
                    QuerySubtask(
                        text=part_norm,
                        reason=f"procedural_step_{i+1}",
                        weight=0.9,
                        boost_terms=boost or boost_terms_global,
                        intent=intent,
                        llm_generated=False,
                    )
                )
            strategy = "multi_part"

    # LLM fallback: if heuristics found <=1 subtask, try LLM with remaining timeout
    remaining_timeout = timeout_seconds - (time.time() - start_time)
    if len(subtasks) <= 1 and remaining_timeout > 0.1:
        logger.debug(f"Attempting LLM fallback for query: {query}")
        llm_subtasks = _llm_decompose_fallback(query, min(remaining_timeout - 0.1, 0.5), max_subtasks)
        if llm_subtasks and len(llm_subtasks) >= 2:
            subtasks = []  # Clear any single heuristic result
            for i, llm_q in enumerate(llm_subtasks):
                # Normalize LLM-generated subtask
                llm_q_norm = _normalize_subtask(llm_q)
                intent = _get_subtask_intent(llm_q_norm)
                boost = _get_subtask_boost_terms(llm_q_norm, glossary)

                subtasks.append(
                    QuerySubtask(
                        text=llm_q_norm,
                        reason=f"llm_generated_{i+1}",
                        weight=0.95,
                        boost_terms=boost or boost_terms_global,
                        intent=intent,
                        llm_generated=True,
                    )
                )
            strategy = "llm"
            llm_used = True
            logger.info(f"LLM decomposition successful for: {query}")

    # Ensure original query is always included with highest weight
    original_subtask = QuerySubtask(
        text=query,
        reason="original",
        weight=1.0,
        boost_terms=boost_terms_global,
        intent=_get_subtask_intent(query),
        llm_generated=False,
    )

    if not any(st.text.lower() == query.lower() for st in subtasks):
        subtasks.insert(0, original_subtask)
    else:
        # Replace any exact match with the fully formed original
        subtasks = [
            original_subtask if st.text.lower() == query.lower() else st
            for st in subtasks
        ]

    timed_out = (time.time() - start_time) > timeout_seconds

    if timed_out:
        logger.warning(f"Query decomposition timed out ({time.time() - start_time:.2f}s): {query}")

    result = QueryDecompositionResult(
        original_query=query,
        subtasks=subtasks[: max(1, max_subtasks)],
        strategy=strategy,
        timed_out=timed_out,
        llm_used=llm_used,
    )

    logger.debug(
        f"Decomposed query '{query}' into {len(result.subtasks)} subtasks "
        f"(strategy={strategy}, llm_used={llm_used}, timed_out={timed_out})"
    )

    return result


def is_multi_intent_query(query: str) -> bool:
    """
    Detect if query has multiple intents or comparisons.

    Uses conjunction keywords, enumerations, comparison phrases, and
    entity count from glossary to determine if decomposition is beneficial.
    """
    query_lower = query.lower()

    # Comparison indicators
    if any(
        keyword in query_lower
        for keyword in [" vs ", " versus ", "difference between", "compare"]
    ):
        return True

    # Procedural indicators
    if any(
        keyword in query_lower
        for keyword in ["first", "then", "next", "finally", "step"]
    ):
        return True

    # Multiple conjunctions with short phrases between them
    and_parts = query_lower.split(" and ")
    if len(and_parts) >= 2:
        # Check if parts are meaningful (not just adjectives)
        meaningful_count = sum(1 for part in and_parts if len(part.strip()) > 5)
        if meaningful_count >= 2:
            return True

    # Check glossary entity count
    glossary = _load_glossary()
    entity_count = sum(1 for term in glossary.keys() if term.lower() in query_lower)
    if entity_count >= 2:
        return True

    return False


if __name__ == "__main__":
    # Quick test
    test_queries = [
        "What is the difference between kiosk and timer?",
        "How do I export timesheets and invoices?",
        "First step to set up workspace, then configure time off",
        "approvals workflow API",
    ]

    for q in test_queries:
        result = decompose_query(q)
        print(f"\nQuery: {q}")
        print(f"  Strategy: {result.strategy}")
        print(f"  Subtasks: {result.to_strings()}")
        print(f"  Multi-intent: {is_multi_intent_query(q)}")
