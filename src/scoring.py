#!/usr/bin/env python3
"""
Confidence scoring for search results.

Scores results based on semantic similarity, keyword matching, and metadata.
"""

import logging
from typing import Dict, List, Optional
import math

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Score result confidence based on multiple factors."""

    # Thresholds for confidence levels
    CONFIDENCE_THRESHOLD_HIGH = 75  # > 75 = High confidence (green)
    CONFIDENCE_THRESHOLD_MEDIUM = 50  # 50-75 = Medium confidence (yellow)
    # < 50 = Low confidence (red)

    def __init__(self):
        """Initialize scorer."""
        self.scoring_history = []

    def score(
        self,
        result: Dict,
        query: str,
        query_entities: List[str],
        query_type: str,
    ) -> Dict:
        """
        Score a search result for confidence.

        Args:
            result: Search result dict with title, content, relevance_score, etc.
            query: Original query string
            query_entities: Extracted entities from query
            query_type: Type of query (definition, how-to, etc.)

        Returns:
            Dict with:
            - confidence: 0-100 confidence score
            - level: "high", "medium", or "low"
            - factors: breakdown of scoring factors
            - recommendation: user-friendly confidence message
        """
        factors = {}

        # Factor 1: Semantic similarity (vector search score)
        # If available, use the model's relevance score (typically 0-1)
        semantic_score = result.get("relevance_score", result.get("score", 0.5))
        if isinstance(semantic_score, float) and semantic_score <= 1.0:
            # Convert 0-1 to 0-100
            factors["semantic"] = semantic_score * 100
        else:
            factors["semantic"] = min(100, semantic_score)  # Already 0-100

        # Factor 2: Keyword density (how many query entities appear in result)
        keyword_score = self._score_keywords(query, query_entities, result)
        factors["keywords"] = keyword_score

        # Factor 3: Content quality (article length, structure)
        quality_score = self._score_quality(result)
        factors["quality"] = quality_score

        # Factor 4: Query-type alignment
        alignment_score = self._score_alignment(result, query_type)
        factors["alignment"] = alignment_score

        # Factor 5: Source reliability (namespace, metadata)
        source_score = self._score_source(result)
        factors["source"] = source_score

        # Weighted average of all factors
        weights = {
            "semantic": 0.35,  # Semantic is most important
            "keywords": 0.20,
            "quality": 0.15,
            "alignment": 0.15,
            "source": 0.15,
        }

        confidence = sum(factors[key] * weights[key] for key in factors)
        confidence = max(0, min(100, confidence))  # Clamp 0-100

        # Determine confidence level
        if confidence >= self.CONFIDENCE_THRESHOLD_HIGH:
            level = "high"
            emoji = "🟢"
        elif confidence >= self.CONFIDENCE_THRESHOLD_MEDIUM:
            level = "medium"
            emoji = "🟡"
        else:
            level = "low"
            emoji = "🔴"

        # Generate recommendation message
        recommendation = self._generate_recommendation(confidence, level, factors)

        result_with_score = {
            "confidence": round(confidence, 1),
            "level": level,
            "emoji": emoji,
            "factors": {k: round(v, 1) for k, v in factors.items()},
            "recommendation": recommendation,
        }

        self.scoring_history.append(result_with_score)
        return result_with_score

    def _score_keywords(self, query: str, entities: List[str], result: Dict) -> float:
        """Score based on keyword matching."""
        if not entities:
            return 50.0  # Neutral if no entities

        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        combined = f"{title} {content}"

        # Count entity matches (weighted by position)
        matched = 0
        for entity in entities:
            if entity in title:
                matched += 2  # Title match worth more
            elif entity in combined:
                matched += 1

        # Calculate percentage match
        match_ratio = min(1.0, matched / (len(entities) * 2))
        return match_ratio * 100

    def _score_quality(self, result: Dict) -> float:
        """Score result quality based on content attributes."""
        score = 50.0  # Neutral baseline

        # Longer content = more thorough
        content_length = len(result.get("content", ""))
        if content_length > 500:
            score += 25
        elif content_length > 200:
            score += 15
        elif content_length > 50:
            score += 5

        # Article structure indicators
        title = result.get("title", "")
        if title and len(title) > 5:
            score += 10

        section = result.get("section", "")
        if section and len(section) > 2:
            score += 5

        # URL structure (help articles typically more reliable)
        url = result.get("url", "")
        if "help" in url.lower():
            score += 10
        elif "docs" in url.lower() or "guide" in url.lower():
            score += 5

        return min(100, score)

    def _score_alignment(self, result: Dict, query_type: str) -> float:
        """Score how well result aligns with query type."""
        score = 50.0  # Neutral baseline

        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        combined = f"{title} {content}"

        # Query type alignments
        type_keywords = {
            "definition": ["definition", "what is", "explanation", "meaning"],
            "how-to": ["how to", "steps", "process", "guide", "create", "setup"],
            "comparison": ["vs", "versus", "difference", "compare", "better"],
            "factual": ["does", "can", "is", "feature", "support"],
        }

        if query_type in type_keywords:
            keywords = type_keywords[query_type]
            for keyword in keywords:
                if keyword in combined:
                    score += 10
                    break  # One per category

        return min(100, score)

    def _score_source(self, result: Dict) -> float:
        """Score source reliability."""
        score = 50.0  # Neutral baseline

        # Namespace reliability (Clockify help typically most reliable)
        namespace = result.get("namespace", "")
        if "clockify" in namespace.lower():
            score += 20
        elif "help" in namespace.lower():
            score += 10

        # Recent updates if available
        if result.get("modified_at"):
            score += 10

        # Has URL (official source)
        if result.get("url"):
            score += 5

        return min(100, score)

    def _generate_recommendation(
        self, confidence: float, level: str, factors: Dict
    ) -> str:
        """Generate user-friendly recommendation message."""
        if level == "high":
            return "✓ Highly relevant result. Good confidence in this answer."
        elif level == "medium":
            if factors["keywords"] < 30:
                return "△ Decent match, but few keywords found. May need refinement."
            elif factors["semantic"] < 50:
                return "△ Moderate relevance. Consider alternative phrasing."
            else:
                return "△ Reasonable result. More refinement may help."
        else:  # low
            if confidence < 25:
                return "✗ Poor match. Try rephrasing your question."
            elif factors["keywords"] < 20:
                return "✗ Few matching keywords found. Search for specific terms."
            else:
                return "✗ Low confidence result. Consider a new search."

    def batch_score(
        self,
        results: List[Dict],
        query: str,
        query_entities: List[str],
        query_type: str,
    ) -> List[Dict]:
        """Score multiple results at once."""
        scored_results = []
        for result in results:
            scored = result.copy()
            score_info = self.score(result, query, query_entities, query_type)
            scored.update(score_info)
            scored_results.append(scored)

        # Sort by confidence (descending)
        scored_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return scored_results

    def get_stats(self) -> Dict:
        """Get scoring statistics."""
        if not self.scoring_history:
            return {"results_scored": 0}

        confidences = [r.get("confidence", 0) for r in self.scoring_history]
        levels = {}
        for r in self.scoring_history:
            level = r.get("level", "unknown")
            levels[level] = levels.get(level, 0) + 1

        return {
            "results_scored": len(self.scoring_history),
            "avg_confidence": round(sum(confidences) / len(confidences), 1),
            "level_distribution": levels,
            "min_confidence": round(min(confidences), 1),
            "max_confidence": round(max(confidences), 1),
        }


# Global instance
_scorer = None


def get_scorer() -> ConfidenceScorer:
    """Get or create global confidence scorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ConfidenceScorer()
    return _scorer
