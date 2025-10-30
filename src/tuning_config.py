#!/usr/bin/env python3
"""
Centralized Tuning Constants for RAG System

This module consolidates all magic numbers and hyperparameters across the system.
Modifying values here affects global behavior without code changes.

PHASE 5 Gold-Standard RAG Tuning:
- RRF (Reciprocal Rank Fusion): Research-backed hybrid ranking
- MMR (Maximal Marginal Relevance): Diversity-aware result selection
- Time Decay: Freshness boosting for time-sensitive documents
- Answerability Scoring: Grounding validation (Jaccard overlap)
- Semantic Caching: LRU cache with embedding fingerprint keys
"""

# ============================================================================
# PHASE 5: Gold-Standard RAG Fusion & Diversity
# ============================================================================

# Reciprocal Rank Fusion (RRF)
# Formula: score = 1/(k+rank) for both vector and BM25 rankings
# k=60 is research-standard; lower k = higher weight to top results
RRF_K_CONSTANT: float = 60.0

# Maximal Marginal Relevance (MMR)
# Formula: MMR_score = λ*relevance - (1-λ)*max_similarity_to_selected
# λ=0.7 balances relevance (70%) vs diversity (30%)
MMR_LAMBDA: float = 0.7

# Time Decay for Freshness
# Formula: boosted_score = original_score * (decay_rate ^ months_old)
# 0.95 = 5% decay per month; documents decay over time
TIME_DECAY_RATE: float = 0.95
TIME_DECAY_MONTHS_DIVISOR: float = 30.0  # Days to months conversion

# ============================================================================
# Answerability & Grounding Validation
# ============================================================================

# Jaccard overlap threshold for answer grounding
# score >= 0.18 means >=18% token overlap between answer and context
# Below this threshold: answer may be hallucinated -> safe refusal instead
# Lowered from 0.25 to 0.18 to allow more paraphrased answers while still catching hallucinations
ANSWERABILITY_THRESHOLD: float = 0.18

# ============================================================================
# Semantic Answer Caching
# ============================================================================

# Maximum number of cached entries (LRU eviction after this)
SEMANTIC_CACHE_MAX_SIZE: int = 10000

# Time-to-live for cached entries (seconds = 1 hour)
SEMANTIC_CACHE_TTL_SECONDS: int = 3600

# ============================================================================
# Hybrid Search Configuration
# ============================================================================

# Hybrid ranking weight: α*vector + (1-α)*bm25
# DEPRECATED: PHASE 5 replaced with RRF, but kept for legacy support
HYBRID_ALPHA: float = 0.7

# Diversity penalty weight in old diversity filtering (deprecated)
# PHASE 5: Replaced with MMR (Maximal Marginal Relevance)
DIVERSITY_PENALTY_WEIGHT: float = 0.15

# ============================================================================
# BM25 Text Ranking
# ============================================================================

# BM25 length normalization parameter (typical: 0.5-0.75)
# 0.75 = moderate length normalization (default tuning)
BM25_B: float = 0.75

# ============================================================================
# LLM Generation
# ============================================================================

# LLM temperature for deterministic/consistent responses
# 0.0 = deterministic (always same answer for same query)
# Higher values = more creative/varied responses
LLM_TEMPERATURE_DEFAULT: float = 0.0
LLM_TEMPERATURE_MIN: float = 0.0
LLM_TEMPERATURE_MAX: float = 2.0

# LLM backoff multiplier for retry delays
# Used in exponential backoff: delay = base * (backoff ^ attempt)
LLM_BACKOFF: float = 0.75

# LLM response timeout (seconds)
# Default max tokens for generation
LLM_MAX_TOKENS_DEFAULT: int = 800

# ============================================================================
# Rate Limiting & Circuit Breaker
# ============================================================================

# Minimum interval between requests from same IP (seconds)
# 0.1 = 100ms minimum, allows ~10 requests/sec per IP
RATE_LIMIT_INTERVAL: float = 0.1

# Circuit breaker recovery timeout (seconds)
# Time to wait in OPEN state before trying HALF_OPEN
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 60.0

# ============================================================================
# Search Improvements & Boosting (PHASE 10a)
# ============================================================================

# Query-specific boost factors for semantic relevance
# Structured as: query_type -> field -> boost_factor
QUERY_BOOST_FACTORS: dict = {
    "factual": {
        "title_boost": 0.12,
        "section_boost": 0.06,
        "exact_match_boost": 0.10,
    },
    "how_to": {
        "title_boost": 0.08,
        "section_boost": 0.08,
        "structure_boost": 0.12,
    },
    "comparison": {
        "title_boost": 0.06,
        "section_boost": 0.10,
        "diversity_boost": 0.08,
    },
    "definition": {
        "title_boost": 0.15,
        "section_boost": 0.05,
        "conciseness_boost": 0.05,
    },
    "general": {
        "title_boost": 0.08,
        "section_boost": 0.05,
        "default_boost": 0.0,
    },
}

# Maximum boost cap (prevents scores >1.0)
BOOST_MAX_CAP: float = 0.3

# Phrase match boost multiplier
PHRASE_MATCH_MULTIPLIER: float = 1.0

# Diversity boost reduction factor
DIVERSITY_BOOST_FACTOR: float = 0.5

# ============================================================================
# Query Expansion & Decomposition
# ============================================================================

# Synonym/glossary expansion weight
EXPANSION_SYNONYM_WEIGHT: float = 0.8

# Boost term weight (from decomposition)
EXPANSION_BOOST_WEIGHT: float = 0.9

# Query decomposition timeout (seconds)
QUERY_DECOMPOSE_TIMEOUT: float = 0.75

# LLM fallback timeout for decomposition (seconds)
QUERY_DECOMPOSE_LLM_FALLBACK: float = 0.5

# Maximum number of decomposed subtasks
QUERY_DECOMPOSE_MAX_SUBTASKS: int = 3

# ============================================================================
# Retrieval Configuration
# ============================================================================

# Default retrieval timeout (seconds)
RETRIEVAL_TIMEOUT_SECONDS: float = 30.0

# Confidence threshold for query analysis
ANALYSIS_CONFIDENCE_THRESHOLD: float = 0.3

# Per-entity confidence weight
ENTITY_CONFIDENCE_WEIGHT: float = 0.2

# General query confidence baseline
GENERAL_CONFIDENCE_BOOST: float = 0.5

# ============================================================================
# Helper Functions
# ============================================================================


def get_boost_factors(query_type: str) -> dict:
    """
    Get boost factors for a query type.

    Args:
        query_type: One of "factual", "how_to", "comparison", "definition", "general"

    Returns:
        Dictionary of boost factors for that query type
    """
    return QUERY_BOOST_FACTORS.get(query_type, QUERY_BOOST_FACTORS["general"])


def get_rrf_score(vector_rank: int, bm25_rank: int) -> float:
    """
    Calculate RRF score for a result at given ranks.

    Args:
        vector_rank: Rank in vector search results (0-indexed)
        bm25_rank: Rank in BM25 results (0-indexed)

    Returns:
        Combined RRF score
    """
    k = RRF_K_CONSTANT
    vector_score = 1.0 / (k + vector_rank)
    bm25_score = 1.0 / (k + bm25_rank)
    return vector_score + bm25_score


def get_mmr_score(
    relevance: float,
    max_similarity: float,
    lambda_param: float = MMR_LAMBDA,
) -> float:
    """
    Calculate MMR score balancing relevance and diversity.

    Args:
        relevance: Original relevance score (0-1)
        max_similarity: Maximum similarity to already-selected results (0-1)
        lambda_param: Balance parameter (default: MMR_LAMBDA)

    Returns:
        MMR score (can be negative if diversity penalty is high)
    """
    return lambda_param * relevance - (1 - lambda_param) * max_similarity


def get_time_decay_factor(days_old: float) -> float:
    """
    Calculate time decay multiplier for a document.

    Args:
        days_old: Age of document in days

    Returns:
        Decay factor (0-1) to multiply with score
    """
    months_old = days_old / TIME_DECAY_MONTHS_DIVISOR
    return TIME_DECAY_RATE ** months_old
