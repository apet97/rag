#!/usr/bin/env python3
"""
PHASE 5: Performance Tracker and Latency Metrics

Tracks end-to-end and stage-by-stage latency across the RAG pipeline.
Provides insights into performance bottlenecks and optimization opportunities.

Key metrics:
- Query embedding latency
- Retrieval latency (vector, BM25, hybrid, fusion)
- Reranking latency
- LLM generation latency
- Caching impact (hit/miss latency difference)
- End-to-end latency
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
from loguru import logger


class PipelineStage(str, Enum):
    """Pipeline stages for latency tracking."""
    QUERY_EMBEDDING = "query_embedding"
    VECTOR_SEARCH = "vector_search"
    BM25_SEARCH = "bm25_search"
    FUSION = "fusion"
    DIVERSITY_FILTER = "diversity_filter"
    RERANKING = "reranking"
    TIME_DECAY = "time_decay"
    LLM_GENERATION = "llm_generation"
    ANSWERABILITY_CHECK = "answerability_check"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_STORE = "cache_store"
    END_TO_END = "end_to_end"


@dataclass
class LatencySample:
    """Single latency measurement."""
    stage: PipelineStage
    duration_ms: float
    timestamp: float
    query_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageStats:
    """Statistics for a pipeline stage."""
    stage: PipelineStage
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    samples: List[float] = field(default_factory=list)

    def update(self, duration_ms: float) -> None:
        """Update stats with a new sample."""
        self.samples.append(duration_ms)
        self.count += 1
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

        if len(self.samples) >= 2:
            self.mean_ms = statistics.mean(self.samples)
            self.median_ms = statistics.median(self.samples)
            if len(self.samples) >= 20:
                self.p95_ms = statistics.quantiles(self.samples, n=20)[18]  # 95th percentile
                self.p99_ms = statistics.quantiles(self.samples, n=100)[98]  # 99th percentile

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage": self.stage.value,
            "count": self.count,
            "min_ms": round(self.min_ms, 2) if self.min_ms != float('inf') else 0,
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
        }


class PerformanceTracker:
    """
    Thread-safe performance tracker for RAG pipeline.

    Tracks latency across all stages and provides statistical analysis.
    """

    def __init__(self, max_samples_per_stage: int = 1000):
        """
        Initialize tracker.

        Args:
            max_samples_per_stage: Maximum samples to keep per stage (for memory efficiency)
        """
        self.max_samples = max_samples_per_stage
        self._stats: Dict[PipelineStage, StageStats] = {
            stage: StageStats(stage=stage) for stage in PipelineStage
        }
        self._samples: List[LatencySample] = []
        self._lock = threading.RLock()
        self._active_timers: Dict[str, float] = {}  # query_id -> start_time

        logger.info(f"Initialized PerformanceTracker (max_samples={max_samples_per_stage})")

    def start_timer(self, query_id: str) -> None:
        """Start end-to-end timer for a query."""
        with self._lock:
            self._active_timers[query_id] = time.time()

    def record(
        self,
        stage: PipelineStage,
        duration_ms: float,
        query_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a latency measurement.

        Args:
            stage: Pipeline stage
            duration_ms: Duration in milliseconds
            query_id: Query identifier (for correlation)
            metadata: Additional metadata (count, items, etc.)
        """
        with self._lock:
            # Update stage stats
            self._stats[stage].update(duration_ms)

            # Store sample (with limit)
            sample = LatencySample(
                stage=stage,
                duration_ms=duration_ms,
                timestamp=time.time(),
                query_id=query_id,
                metadata=metadata or {},
            )
            self._samples.append(sample)

            if len(self._samples) > self.max_samples * len(PipelineStage):
                # Trim oldest samples
                self._samples = self._samples[-(self.max_samples * len(PipelineStage)) :]

            # Log DEBUG info for slow operations
            if duration_ms > 1000:  # >1 second
                logger.warning(
                    f"Slow stage: {stage.value} took {duration_ms:.0f}ms"
                    f"{f' (query={query_id})' if query_id else ''}"
                )

    def end_timer(self, query_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End end-to-end timer and record duration."""
        with self._lock:
            if query_id not in self._active_timers:
                logger.warning(f"Timer not started for query {query_id}")
                return

            start_time = self._active_timers.pop(query_id)
            duration_ms = (time.time() - start_time) * 1000

            self.record(PipelineStage.END_TO_END, duration_ms, query_id, metadata)

    def get_stats(self, stage: Optional[PipelineStage] = None) -> Dict[str, Any]:
        """
        Get statistics for a stage or all stages.

        Args:
            stage: Specific stage, or None for all

        Returns:
            Statistics dictionary
        """
        with self._lock:
            if stage:
                return self._stats[stage].to_dict()

            return {
                "by_stage": {s.stage.value: s.to_dict() for s in self._stats.values()},
                "total_samples": len(self._samples),
            }

    def get_recent_samples(self, stage: Optional[PipelineStage] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent samples for analysis.

        Args:
            stage: Filter by stage, or None for all
            limit: Maximum samples to return

        Returns:
            List of sample dictionaries
        """
        with self._lock:
            samples = self._samples
            if stage:
                samples = [s for s in samples if s.stage == stage]

            return [
                {
                    "stage": s.stage.value,
                    "duration_ms": round(s.duration_ms, 2),
                    "query_id": s.query_id,
                    "metadata": s.metadata,
                    "timestamp": s.timestamp,
                }
                for s in samples[-limit:]
            ]

    def get_bottlenecks(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get slowest stages (potential bottlenecks).

        Args:
            top_n: Number of slowest stages to return

        Returns:
            List of stage stats sorted by mean latency
        """
        with self._lock:
            stages = sorted(
                [s for s in self._stats.values() if s.count > 0],
                key=lambda s: s.mean_ms,
                reverse=True,
            )
            return [s.to_dict() for s in stages[:top_n]]

    def clear(self) -> None:
        """Clear all statistics."""
        with self._lock:
            self._stats = {stage: StageStats(stage=stage) for stage in PipelineStage}
            self._samples.clear()
            self._active_timers.clear()
            logger.info("Performance tracker cleared")

    def report(self) -> str:
        """Generate a human-readable performance report."""
        with self._lock:
            lines = ["Performance Report", "=" * 60]

            # Overall stats
            end_to_end = self._stats[PipelineStage.END_TO_END]
            if end_to_end.count > 0:
                lines.append(f"\nEnd-to-End Latency (n={end_to_end.count})")
                lines.append(f"  Mean: {end_to_end.mean_ms:.0f}ms")
                lines.append(f"  Median: {end_to_end.median_ms:.0f}ms")
                lines.append(f"  P95: {end_to_end.p95_ms:.0f}ms")
                lines.append(f"  Min-Max: {end_to_end.min_ms:.0f}-{end_to_end.max_ms:.0f}ms")

            # Stage breakdown
            lines.append("\nStage Breakdown (sorted by mean latency)")
            lines.append("-" * 60)
            stages = sorted(
                [s for s in self._stats.values() if s.count > 0],
                key=lambda s: s.mean_ms,
                reverse=True,
            )
            for stage in stages:
                lines.append(
                    f"{stage.stage.value:25} {stage.mean_ms:7.0f}ms "
                    f"(n={stage.count:4d}, p95={stage.p95_ms:7.0f}ms)"
                )

            return "\n".join(lines)


# Module-level singleton instance
_performance_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_performance_tracker() -> PerformanceTracker:
    """Get or create module-level performance tracker singleton."""
    global _performance_tracker

    if _performance_tracker is None:
        with _tracker_lock:
            if _performance_tracker is None:
                _performance_tracker = PerformanceTracker()

    return _performance_tracker
