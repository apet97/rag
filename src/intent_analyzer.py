#!/usr/bin/env python3
"""
Intent Analyzer for RAG queries.

Consolidates intent detection and keyword extraction from existing modules:
- src/search_improvements.py (query type detection)
- src/query_optimizer.py (entity extraction)

Provides unified interface for analyzing user queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

# Import existing analysis functions
from src.search_improvements import detect_query_type

try:
    from src.query_optimizer import QueryOptimizer
    HAS_OPTIMIZER = True
except ImportError:
    logger.warning("QueryOptimizer not available, using fallback keyword extraction")
    HAS_OPTIMIZER = False


@dataclass
class IntentAnalysis:
    """Result of intent analysis."""

    intent: Literal["how_to", "comparison", "factual", "definition", "general"]
    keywords: List[str]
    confidence: float  # 0.0-1.0
    query_type: str  # Detailed classification
    entities: List[str]  # Named entities if available
    expansions: List[str]  # Query expansion suggestions


def analyze_intent(query: str, ticket_data: Optional[Any] = None) -> IntentAnalysis:
    """
    Analyze query intent and extract keywords.

    Combines multiple analysis approaches:
    - Intent detection (how-to, comparison, factual, etc.)
    - Keyword extraction (stop word removal)
    - Entity recognition (if QueryOptimizer available)
    - Query expansion suggestions

    Args:
        query: User's question or search query
        ticket_data: Optional TicketData if this is a parsed ticket

    Returns:
        IntentAnalysis with detected intent, keywords, and metadata
    """
    logger.debug(f"Analyzing intent for query: {query[:100]}...")

    # Detect primary intent
    intent = detect_query_type(query)

    # Extract keywords using multiple methods
    keywords = _extract_keywords(query)

    # Use QueryOptimizer if available for advanced analysis
    entities = []
    expansions = []
    confidence = 0.7  # Base confidence

    if HAS_OPTIMIZER:
        try:
            optimizer = QueryOptimizer()
            analysis = optimizer.analyze(query)

            entities = analysis.get("entities", [])
            expansions = analysis.get("expansion", [])

            # Merge entities into keywords (deduplicate)
            for entity in entities:
                if entity.lower() not in [k.lower() for k in keywords]:
                    keywords.append(entity)

            # Higher confidence if optimizer succeeded
            confidence = analysis.get("confidence", 0.7)

        except Exception as e:
            logger.warning(f"QueryOptimizer analysis failed: {e}")

    # If this is a ticket, incorporate ticket-specific keywords
    if ticket_data:
        ticket_keywords = _extract_ticket_keywords(ticket_data)
        for kw in ticket_keywords:
            if kw.lower() not in [k.lower() for k in keywords]:
                keywords.append(kw)

        # Tickets have higher confidence since they're structured
        confidence = min(confidence + 0.1, 1.0)

    # Determine query type (more detailed than intent)
    query_type = _classify_query_type(query, intent)

    # Adjust confidence based on intent clarity
    confidence = _adjust_confidence(query, intent, len(keywords), confidence)

    result = IntentAnalysis(
        intent=intent,
        keywords=keywords[:10],  # Limit to top 10
        confidence=confidence,
        query_type=query_type,
        entities=entities[:5],  # Top 5 entities
        expansions=expansions[:3]  # Top 3 expansions
    )

    logger.info(
        f"Intent analysis: intent={result.intent}, "
        f"keywords={result.keywords[:3]}..., "
        f"confidence={result.confidence:.2f}"
    )

    return result


def _extract_keywords(query: str) -> List[str]:
    """
    Extract keywords from query using stop word removal.

    Args:
        query: Query text

    Returns:
        List of extracted keywords (lowercase)
    """
    # Common stop words (English)
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else',
        'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours'
    }

    # Tokenize and filter
    tokens = re.findall(r'\b\w+\b', query.lower())

    keywords = [
        token for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    return unique_keywords


def _extract_ticket_keywords(ticket_data: Any) -> List[str]:
    """
    Extract keywords from parsed ticket data.

    Prioritizes error messages and issue description.

    Args:
        ticket_data: TicketData object

    Returns:
        List of ticket-specific keywords
    """
    keywords = []

    # Extract from error messages (high priority)
    for error in ticket_data.error_messages[:2]:  # First 2 errors
        error_keywords = _extract_keywords(error)
        keywords.extend(error_keywords[:3])  # Top 3 from each error

    # Extract from issue description
    issue_keywords = _extract_keywords(ticket_data.issue_description)
    keywords.extend(issue_keywords[:5])  # Top 5 from issue

    # Deduplicate
    unique_keywords = []
    seen = set()
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_keywords.append(kw)

    return unique_keywords


def _classify_query_type(
    query: str,
    intent: Literal["how_to", "comparison", "factual", "definition", "general"]
) -> str:
    """
    Classify query into detailed type based on intent and patterns.

    Args:
        query: Query text
        intent: Detected intent

    Returns:
        Detailed query type string
    """
    query_lower = query.lower()

    # Map intent to detailed types
    type_mapping = {
        "how_to": "procedural",
        "comparison": "comparative",
        "factual": "factual",
        "definition": "conceptual",
        "general": "exploratory"
    }

    base_type = type_mapping.get(intent, "general")

    # Add modifiers based on patterns
    modifiers = []

    if re.search(r'\b(troubleshoot|fix|error|problem|issue)\b', query_lower):
        modifiers.append("troubleshooting")

    if re.search(r'\b(best|optimal|recommend|should)\b', query_lower):
        modifiers.append("advisory")

    if re.search(r'\b(multiple|several|all|list)\b', query_lower):
        modifiers.append("comprehensive")

    if re.search(r'\b(specific|particular|exact)\b', query_lower):
        modifiers.append("targeted")

    # Combine
    if modifiers:
        return f"{base_type}_{modifiers[0]}"

    return base_type


def _adjust_confidence(
    query: str,
    intent: str,
    keyword_count: int,
    base_confidence: float
) -> float:
    """
    Adjust confidence score based on query characteristics.

    Args:
        query: Query text
        intent: Detected intent
        keyword_count: Number of extracted keywords
        base_confidence: Starting confidence from optimizer

    Returns:
        Adjusted confidence score (0.0-1.0)
    """
    confidence = base_confidence

    # Clear intent indicators boost confidence
    intent_indicators = {
        "how_to": [r'\bhow (do|to|can)\b', r'\bsteps?\b'],
        "comparison": [r'\bvs\b', r'\bdifference\b', r'\bcompare\b'],
        "factual": [r'\b(who|when|where|which)\b'],
        "definition": [r'\bwhat is\b', r'\bdefine\b', r'\bmeaning\b']
    }

    indicators = intent_indicators.get(intent, [])
    for pattern in indicators:
        if re.search(pattern, query.lower()):
            confidence = min(confidence + 0.05, 1.0)

    # More keywords = clearer intent
    if keyword_count >= 5:
        confidence = min(confidence + 0.1, 1.0)
    elif keyword_count >= 3:
        confidence = min(confidence + 0.05, 1.0)
    elif keyword_count < 2:
        confidence = max(confidence - 0.1, 0.3)

    # Very short or very long queries reduce confidence
    word_count = len(query.split())
    if word_count < 3:
        confidence = max(confidence - 0.15, 0.4)
    elif word_count > 30:
        confidence = max(confidence - 0.1, 0.5)

    # Question mark presence (clear question)
    if '?' in query:
        confidence = min(confidence + 0.05, 1.0)

    return min(max(confidence, 0.3), 1.0)  # Clamp to [0.3, 1.0]


def get_adaptive_article_count(
    intent: Literal["how_to", "comparison", "factual", "definition", "general"],
    config: Optional[Dict[str, int]] = None
) -> int:
    """
    Get adaptive article count based on query intent.

    Args:
        intent: Detected intent type
        config: Optional config dict with custom counts

    Returns:
        Number of articles to retrieve
    """
    # Default counts per intent
    default_counts = {
        "how_to": 3,
        "comparison": 5,
        "factual": 4,
        "definition": 3,
        "general": 4
    }

    if config:
        counts = {**default_counts, **config}
    else:
        counts = default_counts

    count = counts.get(intent, 4)

    logger.debug(f"Adaptive article count for intent='{intent}': {count}")

    return count
