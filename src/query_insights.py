#!/usr/bin/env python3
"""
TIER 2: Query Insights and Analysis Logging

Provides detailed logging and insights into query processing,
fusion strategy performance, and ranking decisions.

Key metrics tracked:
- Query characteristics (type, intent, complexity)
- Fusion strategy effectiveness (RRF vs weighted)
- Ranking quality and diversity metrics
- Reranker performance and decisions
"""

import time
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass
from collections import defaultdict
import threading


@dataclass
class QueryInsight:
    """Single query analysis record."""
    query_id: str
    query_text: str
    query_type: Optional[str] = None
    is_multi_intent: bool = False
    vector_results_count: int = 0
    bm25_results_count: int = 0
    fused_results_count: int = 0
    mmr_applied: bool = False
    time_decay_applied: bool = False
    cache_hit: bool = False
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class QueryInsightTracker:
    """Track and analyze query processing decisions."""

    def __init__(self, max_queries: int = 1000):
        """
        Initialize tracker.

        Args:
            max_queries: Maximum queries to keep in memory
        """
        self.max_queries = max_queries
        self._insights: List[QueryInsight] = []
        self._lock = threading.RLock()
        self._strategy_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "total": 0})

    def record_query(self, insight: QueryInsight) -> None:
        """Record a query analysis."""
        with self._lock:
            self._insights.append(insight)
            if len(self._insights) > self.max_queries:
                self._insights = self._insights[-self.max_queries :]

            # Log insight summary
            logger.debug(
                f"Query insight: {insight.query_type or 'unknown'} | "
                f"v:{insight.vector_results_count} bm25:{insight.bm25_results_count} "
                f"fused:{insight.fused_results_count} | "
                f"mmr={insight.mmr_applied} decay={insight.time_decay_applied} "
                f"cache_hit={insight.cache_hit}"
            )

    def track_fusion_strategy(self, strategy_name: str, succeeded: bool) -> None:
        """Track fusion strategy usage."""
        with self._lock:
            stats = self._strategy_stats[strategy_name]
            stats["total"] += 1
            if succeeded:
                stats["success"] += 1

    def get_fusion_effectiveness(self) -> Dict[str, Dict[str, Any]]:
        """Get fusion strategy performance stats."""
        with self._lock:
            result = {}
            for strategy, stats in self._strategy_stats.items():
                success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
                result[strategy] = {
                    "total_queries": stats["total"],
                    "successful": stats["success"],
                    "success_rate_pct": round(success_rate, 2),
                }
            return result

    def get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types."""
        with self._lock:
            distribution = defaultdict(int)
            for insight in self._insights:
                qtype = insight.query_type or "unknown"
                distribution[qtype] += 1
            return dict(distribution)

    def get_feature_adoption(self) -> Dict[str, Any]:
        """Get adoption rate of advanced features."""
        with self._lock:
            if not self._insights:
                return {}

            total = len(self._insights)
            mmr_count = sum(1 for i in self._insights if i.mmr_applied)
            decay_count = sum(1 for i in self._insights if i.time_decay_applied)
            multi_intent_count = sum(1 for i in self._insights if i.is_multi_intent)
            cache_hits = sum(1 for i in self._insights if i.cache_hit)

            return {
                "mmr_usage_pct": round(mmr_count / total * 100, 2),
                "time_decay_usage_pct": round(decay_count / total * 100, 2),
                "multi_intent_queries_pct": round(multi_intent_count / total * 100, 2),
                "cache_hit_rate_pct": round(cache_hits / total * 100, 2),
                "total_queries_analyzed": total,
            }

    def get_insights_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of query insights."""
        with self._lock:
            return {
                "fusion_effectiveness": self.get_fusion_effectiveness(),
                "query_type_distribution": self.get_query_type_distribution(),
                "feature_adoption": self.get_feature_adoption(),
                "total_insights_stored": len(self._insights),
            }


# Module-level singleton
_insight_tracker: Optional[QueryInsightTracker] = None
_tracker_lock = threading.Lock()


def get_query_insight_tracker() -> QueryInsightTracker:
    """Get or create module-level query insight tracker singleton."""
    global _insight_tracker

    if _insight_tracker is None:
        with _tracker_lock:
            if _insight_tracker is None:
                _insight_tracker = QueryInsightTracker()

    return _insight_tracker
