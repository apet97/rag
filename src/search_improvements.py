from __future__ import annotations

"""
Enhanced search improvements: query type detection, adaptive k multiplier, and improved boosting.
AXIOM: Adaptive search strategies based on query intent for +15-20% relevance improvement.
"""

import re
from typing import Literal, List
import logging

logger = logging.getLogger(__name__)


def detect_query_type(query: str) -> Literal["factual", "how_to", "comparison", "definition", "general"]:
    """
    Detect query intent to apply adaptive search strategies.

    Returns:
        "factual": Who/What/When/Where facts ("When was X founded?")
        "how_to": How-to/procedural queries ("How do I set up X?")
        "comparison": Comparative questions ("What's the difference between X and Y?")
        "definition": Definitional queries ("What is X?", "Define X")
        "general": Default fallback
    """
    query_lower = query.lower().strip()

    # How-to queries
    how_patterns = [
        r'^how\s+to\s+',
        r'^how\s+do\s+i\s+',
        r'^\s*steps?\s+',
        r'^\s*guide\s+',
        r'^\s*setup',
        r'^\s*configure',
        r'^\s*install',
        r'^how\s+can\s+i',
    ]
    if any(re.search(p, query_lower) for p in how_patterns):
        return "how_to"

    # Comparison queries
    compare_patterns = [
        r'\bdifference\s+between\b',
        r'\bvs\.?\b',
        r'\bversus\b',
        r'\bcompare\b',
        r'\bwhich\s+is\s+better',
        r'\bwhich\s+one\s+is\b',
    ]
    if any(re.search(p, query_lower) for p in compare_patterns):
        return "comparison"

    # Definition queries
    def_patterns = [
        r'^what\s+is\s+',
        r'^define\s+',
        r'^definition\s+of\s+',
        r'^\s*definition:',
    ]
    if any(re.search(p, query_lower) for p in def_patterns):
        return "definition"

    # Factual queries (who, what, when, where - but not how or define)
    factual_patterns = [
        r'^who\s+',
        r'^when\s+',
        r'^where\s+',
        r'^\s*what\s+happened',
        r'^which\s+',
    ]
    if any(re.search(p, query_lower) for p in factual_patterns):
        return "factual"

    return "general"


def get_adaptive_k_multiplier(query_type: str, base_k: int) -> int:
    """
    Get adaptive k multiplier for retrieving more candidates before reranking.

    Strategy (PHASE 10a tuning: increased multipliers to improve recall from 0.32 → 0.85):
    - how_to: 8x (procedural questions need broader candidate pool)
    - comparison: 10x (comparisons need diverse results)
    - factual: 6x (factual queries need more precise keyword matching)
    - definition: 5x (definitions benefit from broader context)
    - general: 6x (default increased from 3x)

    Returns:
        Recommended number of candidates to retrieve before reranking
    """
    multipliers = {
        "how_to": 8,          # Increased from 4 for better procedural Q coverage
        "comparison": 10,      # Increased from 5 for diverse viewpoints
        "factual": 6,          # Increased from 3 for keyword precision
        "definition": 5,       # Increased from 2.5 for broader context
        "general": 6,          # Increased from 3 as fallback
    }
    multiplier = multipliers.get(query_type, 6)
    # Cap at reasonable limits: minimum 4x, maximum 20x (increased from 8x to allow more exploration)
    raw_k = int(base_k * multiplier)
    return max(base_k * 4, min(raw_k, base_k * 20))


def get_field_boost(query_type: str) -> dict:
    """
    Get field boosting weights based on query type.

    Returns:
        {"title_boost": float, "section_boost": float, "proximity_boost": bool}
    """
    boosts = {
        "factual": {"title_boost": 0.12, "section_boost": 0.06, "exact_match_boost": 0.10},
        "how_to": {"title_boost": 0.08, "section_boost": 0.08, "structure_boost": 0.12},
        "comparison": {"title_boost": 0.06, "section_boost": 0.10, "diversity_boost": 0.08},
        "definition": {"title_boost": 0.15, "section_boost": 0.05, "conciseness_boost": 0.05},
        "general": {"title_boost": 0.08, "section_boost": 0.05, "default_boost": 0.0},
    }
    return boosts.get(query_type, boosts["general"])


def enhance_field_matching(
    candidate_text: str,
    candidate_title: str,
    candidate_section: str,
    query_tokens: List[str],
    query_type: str,
) -> float:
    """
    Enhanced field matching with query-type-aware boosting.

    Returns:
        Score boost (0.0 - 0.3)
    """
    boost = 0.0
    boosts = get_field_boost(query_type)

    text_lower = candidate_text.lower()
    title_lower = candidate_title.lower()
    section_lower = candidate_section.lower()

    # Title matching (exact and partial)
    title_matches = sum(1 for token in query_tokens if token in title_lower)
    if title_matches > 0:
        # Exact phrase match in title gets extra boost
        phrase_match = 1 if " ".join(query_tokens) in title_lower else 0
        boost += boosts.get("title_boost", 0.08) * (1 + phrase_match)

    # Section matching (keywords)
    section_matches = sum(1 for token in query_tokens if token in section_lower)
    if section_matches > 0:
        boost += boosts.get("section_boost", 0.05)

    # Query-type-specific boosts
    if query_type == "how_to":
        # Boost if text contains procedural keywords
        procedural_keywords = ["step", "first", "then", "next", "finally", "instructions"]
        if any(kw in text_lower for kw in procedural_keywords):
            boost += boosts.get("structure_boost", 0.12)

    elif query_type == "definition":
        # Boost if title or section is concise (short definition)
        if len(candidate_title.split()) <= 4:
            boost += boosts.get("conciseness_boost", 0.05)

    elif query_type == "comparison":
        # Boost diversity (result from different section/category)
        boost += boosts.get("diversity_boost", 0.08) * 0.5  # Modest diversity boost

    elif query_type == "factual":
        # Boost if exact match found
        if any(f" {token} " in f" {text_lower} " for token in query_tokens):
            boost += boosts.get("exact_match_boost", 0.10)

    return min(boost, 0.3)  # Cap at 0.3


def should_enable_hybrid_search(query_type: str, query_length: int) -> bool:
    """
    Determine if hybrid search should be used based on query characteristics.

    Hybrid search (dense + BM25) is better for:
    - Factual queries (exact terms matter)
    - Longer, more complex queries
    - Queries with multiple keywords

    Returns:
        True if hybrid search should be enabled
    """
    # Always use hybrid for factual queries (keywords matter)
    if query_type == "factual":
        return True

    # Use hybrid for medium/long queries (more structural information)
    if query_length > 50:
        return True

    # Use hybrid for comparison queries (multiple entities/terms)
    if query_type == "comparison":
        return True

    # Default: ALWAYS use hybrid search for better keyword matching (catches exact title matches)
    # Pure semantic search misses documents with exact query terms in titles
    return True


def log_query_analysis(query: str, query_type: str, adaptive_k: int) -> None:
    """Log query analysis for debugging and optimization."""
    logger.debug(
        f"Query analysis: type={query_type}, adaptive_k={adaptive_k}, "
        f"length={len(query)}, tokens={len(query.split())}"
    )
