from __future__ import annotations

"""
Query expansion using domain glossary.

Expands queries with synonyms to improve recall during retrieval.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from loguru import logger

_glossary = None


def _load_glossary() -> dict:
    """Load glossary from data/domain/glossary.json."""
    global _glossary
    if _glossary is None:
        path = Path("data/domain/glossary.json")
        if path.exists():
            try:
                _glossary = json.loads(path.read_text(encoding="utf-8"))
                logger.info(f"Loaded glossary with {len(_glossary)} terms")
            except Exception as e:
                logger.warning(f"Failed to load glossary: {e}")
                _glossary = {}
        else:
            logger.debug("No glossary found at data/domain/glossary.json")
            _glossary = {}
    return _glossary


def expand(q: str, max_expansions: int = 8) -> List[str]:
    """
    Expand query with synonyms from glossary.

    Args:
        q: Original query
        max_expansions: Maximum number of expansions to add

    Returns:
        List of queries: [original, synonym1, synonym2, ...]
    """
    glossary = _load_glossary()
    q_lower = q.lower()
    
    expansions = []
    for term, synonyms in glossary.items():
        if term in q_lower:
            for syn in synonyms:
                syn = syn.strip()
                if syn and syn not in expansions and syn != term:
                    expansions.append(syn)
                if len(expansions) >= max_expansions:
                    break
        if len(expansions) >= max_expansions:
            break
    
    # Return original + unique expansions
    result = [q] + expansions[:max_expansions]
    return result


def expand_structured(q: str, max_expansions: int = 8, boost_terms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Expand query with synonyms and boost terms, returning structured variants with weights.

    Args:
        q: Original query
        max_expansions: Maximum number of glossary expansions to add
        boost_terms: Optional list of boost terms (weighted at 0.9) in addition to glossary

    Returns:
        List of dicts: [
            {text: "original query", weight: 1.0},
            {text: "synonym1", weight: 0.8},
            {text: "boost_term", weight: 0.9},
            ...
        ]
    """
    glossary = _load_glossary()
    q_lower = q.lower()

    variants = [{"text": q, "weight": 1.0}]  # Original at full weight

    # Add glossary-based expansions at lower weight
    glossary_expansions = []
    for term, synonyms in glossary.items():
        if term in q_lower:
            for syn in synonyms:
                syn = syn.strip()
                if syn and syn not in glossary_expansions and syn != term:
                    glossary_expansions.append(syn)
                if len(glossary_expansions) >= max_expansions:
                    break
        if len(glossary_expansions) >= max_expansions:
            break

    # Add glossary expansions with weight 0.8 (lower than original but higher than boost terms)
    for exp in glossary_expansions:
        variants.append({"text": exp, "weight": 0.8})

    # Add boost terms with weight 0.9 (high confidence from decomposition context)
    if boost_terms:
        for boost in boost_terms:
            boost = boost.strip()
            if boost and boost != q:
                variants.append({"text": boost, "weight": 0.9})

    return variants


if __name__ == "__main__":
    print("Testing query expansion...")
    queries = ["timesheet", "kiosk", "project budget", "what is sso"]
    for q in queries:
        expanded = expand(q)
        print(f"  '{q}' -> {expanded}")

    print("\nTesting structured expansion...")
    for q in queries:
        expanded = expand_structured(q)
        print(f"  '{q}' -> {expanded}")
