#!/usr/bin/env python3
"""
Query optimization and understanding module.

Analyzes queries to extract intent, entities, and generates optimized search terms.
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types."""
    DEFINITION = "definition"  # "What is X?"
    HOWTO = "how-to"  # "How do I...?"
    COMPARISON = "comparison"  # "X vs Y"
    FACTUAL = "factual"  # "Does X have...?"
    GENERAL = "general"  # Everything else


class QueryOptimizer:
    """Optimize and analyze user queries for better retrieval."""

    # Query type detection patterns
    DEFINITION_PATTERNS = [
        r"^what\s+(?:is|are)\s+",
        r"^what's\s+",
        r"^define\s+",
        r"^explain\s+",
    ]

    HOWTO_PATTERNS = [
        r"^how\s+(?:do|can|to)\s+",
        r"^how\s+(?:do\s+)?i\s+",
        r"^show\s+me\s+",
        r"^help\s+",
    ]

    COMPARISON_PATTERNS = [
        r"\s+vs\s+",
        r"\s+versus\s+",
        r"difference\s+between",
        r"compare\s+",
    ]

    FACTUAL_PATTERNS = [
        r"^does\s+",
        r"^can\s+",
        r"^is\s+",
        r"^are\s+",
        r"^will\s+",
    ]

    # Stop words to remove from query
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "must", "shall", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they",
    }

    def __init__(self):
        """Initialize query optimizer."""
        self.query_history = []

    def analyze(self, query: str) -> Dict:
        """
        Analyze query and return optimization info.

        Returns:
            Dict with:
            - original: original query
            - cleaned: cleaned query (lowercase, normalized)
            - type: detected query type
            - entities: extracted entities/keywords
            - expansion: suggested term expansions
            - confidence: confidence 0-1
        """
        if not query or len(query) < 2:
            logger.warning(f"Query too short: {query}")
            return {
                "original": query,
                "cleaned": query.lower().strip(),
                "type": QueryType.GENERAL.value,
                "entities": [],
                "expansion": [],
                "confidence": 0.0,
                "error": "Query too short",
            }

        cleaned = self._clean(query)
        query_type = self._detect_type(cleaned)
        entities = self._extract_entities(cleaned)
        expansion = self._generate_expansion(entities, query_type)

        # Confidence: higher for specific queries with clear intent
        confidence = min(1.0, len(entities) * 0.2 + (0.5 if query_type != QueryType.GENERAL else 0.0))

        result = {
            "original": query,
            "cleaned": cleaned,
            "type": query_type.value,
            "entities": entities,
            "expansion": expansion,
            "confidence": confidence,
        }

        self.query_history.append(result)
        return result

    def _clean(self, query: str) -> str:
        """Clean and normalize query."""
        # Lowercase
        q = query.lower().strip()

        # Remove extra whitespace
        q = re.sub(r"\s+", " ", q)

        # Remove punctuation except spaces
        q = re.sub(r"[^\w\s]", "", q)

        return q.strip()

    def _detect_type(self, query: str) -> QueryType:
        """Detect query type (definition, how-to, comparison, etc.)."""
        query_lower = query.lower()

        # Check patterns in order of priority
        for pattern in self.DEFINITION_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.DEFINITION

        for pattern in self.HOWTO_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.HOWTO

        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON

        for pattern in self.FACTUAL_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.FACTUAL

        return QueryType.GENERAL

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities/keywords from query.

        Removes stop words and extracts meaningful terms.
        """
        words = query.lower().split()

        # Filter out stop words and short words
        entities = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                unique_entities.append(e)
                seen.add(e)

        return unique_entities[:5]  # Limit to 5 top entities

    def _generate_expansion(self, entities: List[str], query_type: QueryType) -> List[str]:
        """
        Generate term expansions for better retrieval.

        Suggests related terms based on entity type and query intent.
        """
        expansions = []

        # Map of entity patterns to expansion suggestions
        expansion_map = {
            "track": ["time", "entry", "timer", "tracking"],
            "project": ["create", "manage", "delete", "setup"],
            "user": ["member", "team", "permission", "role"],
            "time": ["track", "entry", "hours", "duration"],
            "report": ["generate", "export", "view", "analytics"],
            "team": ["user", "member", "group", "workspace"],
            "permission": ["access", "role", "grant", "allow"],
        }

        # Add query-type-specific expansions
        type_expansions = {
            QueryType.HOWTO: ["step", "process", "guide", "setup"],
            QueryType.DEFINITION: ["what", "explain", "meaning", "definition"],
            QueryType.COMPARISON: ["difference", "versus", "better", "alternative"],
        }

        # Apply entity-specific expansions
        for entity in entities:
            if entity in expansion_map:
                expansions.extend(expansion_map[entity])

        # Apply type-specific expansions
        if query_type in type_expansions:
            expansions.extend(type_expansions[query_type])

        # Remove duplicates and limit to 5
        unique_expansions = list(dict.fromkeys(expansions))[:5]

        return unique_expansions

    def get_search_query(self, analysis: Dict) -> str:
        """
        Generate optimized search query from analysis.

        Combines entities and expansions for better retrieval.
        """
        parts = []

        # Primary entities (always included)
        if analysis.get("entities"):
            parts.extend(analysis["entities"])

        # Include expansion terms
        if analysis.get("expansion"):
            # For high-confidence queries, add expansions
            if analysis.get("confidence", 0) > 0.3:
                parts.extend(analysis["expansion"][:2])  # Top 2 expansions

        # Join and deduplicate
        unique_parts = list(dict.fromkeys(parts))
        search_query = " ".join(unique_parts)

        return search_query if search_query else analysis.get("cleaned", "")

    def suggest_refinements(self, analysis: Dict, results_count: int = 0) -> List[str]:
        """
        Suggest query refinements if results are poor.

        Args:
            analysis: Query analysis result
            results_count: Number of results found

        Returns:
            List of suggested refined queries
        """
        suggestions = []

        # If no results, suggest broader searches
        if results_count == 0:
            entities = analysis.get("entities", [])
            if entities:
                # Try searching for individual entities
                suggestions = [f"Search for: {e}" for e in entities[:3]]
            else:
                suggestions.append("Try a different keyword or phrase")

        # If few results, suggest expansions
        elif results_count < 3:
            expansions = analysis.get("expansion", [])
            if expansions:
                suggestions.append(f"Also try: {expansions[0]}")

        return suggestions

    def get_stats(self) -> Dict:
        """Get optimizer statistics."""
        if not self.query_history:
            return {"queries_analyzed": 0}

        type_counts = {}
        for query in self.query_history:
            qtype = query.get("type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        avg_confidence = sum(q.get("confidence", 0) for q in self.query_history) / len(self.query_history)

        return {
            "queries_analyzed": len(self.query_history),
            "type_distribution": type_counts,
            "avg_confidence": round(avg_confidence, 2),
        }


# Global instance
_optimizer = None


def get_optimizer() -> QueryOptimizer:
    """Get or create global query optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer
