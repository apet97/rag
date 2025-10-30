"""
Pydantic Data Models for RAG System

Provides structured, type-safe data definitions for:
- Search requests and responses
- Query analysis results
- Retrieval results
- Chunk metadata
- Configuration validation
"""

from __future__ import annotations

from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# ============================================================================
# Enums
# ============================================================================


class QueryType(str, Enum):
    """Types of queries for adaptive processing."""
    DEFINITION = "definition"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    FACTUAL = "factual"
    GENERAL = "general"


class ConfidenceLevel(str, Enum):
    """Confidence level of search results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Request Models
# ============================================================================


class SearchRequest(BaseModel):
    """Request for semantic search operation."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How do I track time on a project?",
            "namespace": "clockify",
            "k": 5,
            "temperature": 0.0,
            "expand_query": True,
            "hybrid": True,
            "clustering": True,
        }
    })

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    namespace: str = Field(default="clockify", description="Namespace to search in")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    expand_query: bool = Field(default=True, description="Whether to expand query with synonyms")
    hybrid: bool = Field(default=True, description="Whether to use hybrid search")
    clustering: bool = Field(default=True, description="Whether to cluster similar results")


class ChatRequest(BaseModel):
    """Request for RAG chat with LLM generation."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How do I start tracking time?",
            "namespace": "clockify",
            "k": 5,
            "temperature": 0.0,
        }
    })

    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    namespace: str = Field(default="clockify", description="Namespace to search")
    k: int = Field(default=5, ge=1, le=20, description="Number of context docs")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")


# ============================================================================
# Metadata & Chunk Models
# ============================================================================


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    model_config = ConfigDict(extra="allow")  # Allow additional metadata fields

    source_url: str = Field(description="Original document URL")
    title: str = Field(description="Document title")
    namespace: str = Field(description="Document namespace")
    chunk_index: int = Field(ge=0, description="Index of this chunk in document")
    total_chunks: int = Field(ge=1, description="Total chunks in document")
    embedding_model: str = Field(description="Model used to create embedding")


class Chunk(BaseModel):
    """Document chunk with content and metadata."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "clockify_doc1_chunk0",
            "text": "Time tracking helps you manage projects...",
            "metadata": {
                "source_url": "https://docs.clockify.me/article/1",
                "title": "Getting Started",
                "namespace": "clockify",
                "chunk_index": 0,
                "total_chunks": 5,
                "embedding_model": "nomic-embed-text:latest",
            }
        }
    })

    id: str = Field(description="Unique chunk ID")
    text: str = Field(description="Chunk content")
    metadata: ChunkMetadata = Field(description="Chunk metadata")


# ============================================================================
# Search Result Models
# ============================================================================


class ResultScoreBreakdown(BaseModel):
    """Detailed score breakdown for a result."""

    semantic_similarity: float = Field(ge=0.0, le=1.0, description="Vector similarity score")
    keyword_match: float = Field(ge=0.0, le=1.0, description="BM25 keyword matching score")
    entity_alignment: float = Field(ge=0.0, le=1.0, description="Entity mention score")
    diversity_bonus: float = Field(ge=0.0, le=1.0, description="Diversity score (1.0 = unique)")


class SearchResult(BaseModel):
    """Individual search result with rich metadata for UI consumption."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "clockify_doc1_chunk0",
            "title": "Getting Started with Time Tracking",
            "content": "Time tracking helps you manage your time more effectively...",
            "url": "https://docs.clockify.me/article/1",
            "namespace": "clockify",
            "confidence": 92,
            "level": "high",
            "score": 0.92,
            "semantic_score": 0.92,
            "factors": {
                "semantic_similarity": 0.92,
                "keyword_match": 0.85,
                "entity_alignment": 0.80,
                "diversity_bonus": 0.95,
            },
            "explanation": "Ranked high due to strong semantic match and keyword match",
        }
    })

    id: str = Field(description="Unique result ID")
    title: str = Field(description="Document title")
    content: str = Field(description="Chunk content (truncated to 300 chars)")
    url: str = Field(description="Source URL")
    namespace: str = Field(description="Document namespace")
    confidence: int = Field(ge=0, le=100, description="Confidence percentage")
    level: ConfidenceLevel = Field(description="Confidence level")
    score: float = Field(ge=0.0, le=1.0, description="Raw score (0-1)")
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Semantic similarity")
    factors: Optional[ResultScoreBreakdown] = Field(default=None, description="Score breakdown")
    explanation: Optional[str] = Field(default=None, description="Why this result ranked high")
    cluster_id: Optional[int] = Field(default=None, description="Cluster ID if clustering enabled")
    cluster_size: Optional[int] = Field(default=None, description="Size of cluster")
    # Rich metadata for UI rendering
    breadcrumb: Optional[List[str]] = Field(default=None, description="Navigation breadcrumb path (e.g., ['Clockify Help Center', 'Administration', 'User Roles'])")
    title_path: Optional[List[str]] = Field(default=None, description="Document section path (e.g., ['Admin', 'Permissions', 'Role Assignment'])")
    anchor: Optional[str] = Field(default=None, description="Section anchor ID for deep linking within page")
    section: Optional[str] = Field(default=None, description="Main section title of this chunk")


class QueryAnalysis(BaseModel):
    """Analysis of query for adaptive processing."""

    query_type: QueryType = Field(description="Detected query type")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of detection")
    variants: List[str] = Field(default_factory=list, description="Query variants for search")
    typo_detected: bool = Field(default=False, description="Whether typo was detected")
    primary_search_query: str = Field(description="Query to use for actual search")


class DecompositionMetadata(BaseModel):
    """Metadata about query decomposition for multi-intent queries."""

    strategy: str = Field(description="Decomposition strategy: none, heuristic, or llm")
    subtask_count: int = Field(ge=1, description="Number of subtasks generated")
    subtasks: List[str] = Field(description="List of subtask texts")
    llm_used: bool = Field(description="Whether LLM fallback was used")
    fused_docs: int = Field(ge=0, description="Number of unique documents after fusion")
    multi_hit_docs: int = Field(ge=0, description="Documents matching multiple subtasks")


class ResponseMetadata(BaseModel):
    """Metadata about the response generation."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "decomposition": {
                "strategy": "heuristic",
                "subtask_count": 2,
                "subtasks": ["What is kiosk?", "What is timer?"],
                "llm_used": False,
                "fused_docs": 5,
                "multi_hit_docs": 2,
            },
            "cache_hit": False,
            "index_normalized": True,
        }
    })

    decomposition: Optional[DecompositionMetadata] = Field(default=None, description="Decomposition details if multi-intent")
    latency_breakdown_ms: Optional[Dict[str, float]] = Field(default=None, description="Latency breakdown by component")
    cache_hit: Optional[bool] = Field(default=None, description="Whether result was from cache")
    index_normalized: Optional[bool] = Field(default=None, description="Whether indexes are L2-normalized")


class QueryAnalysisConfig:
    """Config for QueryAnalysis."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query_type": "how_to",
            "entities": ["time tracking", "project"],
            "confidence": 0.95,
            "variants": [
                "how to track time on a project",
                "time tracking for projects",
                "project time tracking",
            ],
            "typo_detected": False,
            "primary_search_query": "how to track time on a project",
        }
    })


# Update QueryAnalysis with new ConfigDict pattern
QueryAnalysis.model_config = ConfigDict(json_schema_extra=QueryAnalysisConfig.model_config.get("json_schema_extra", {}))


# ============================================================================
# Response Models
# ============================================================================


class SearchResponse(BaseModel):
    """Response to search request."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "query": "How do I track time?",
            "query_analysis": {
                "query_type": "how_to",
                "entities": ["time tracking"],
                "confidence": 0.95,
                "variants": ["how to track time", "time tracking"],
                "typo_detected": False,
                "primary_search_query": "how to track time",
            },
            "results": [],
            "total_results": 0,
            "latency_ms": 125.5,
        }
    })

    success: bool = Field(description="Whether search succeeded")
    query: str = Field(description="Original query")
    query_analysis: Optional[QueryAnalysis] = Field(default=None, description="Query analysis")
    results: List[SearchResult] = Field(description="Search results")
    total_results: int = Field(ge=0, description="Total number of results")
    latency_ms: float = Field(ge=0, description="Search latency in milliseconds")
    metadata: Optional[ResponseMetadata] = Field(default=None, description="Response generation metadata")
    query_decomposition: Optional[Dict[str, Any]] = Field(default=None, description="Query decomposition metadata if multi-intent query (deprecated: use metadata.decomposition)")


class ChatResponse(BaseModel):
    """Response to chat request with LLM-generated answer."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "query": "How do I start tracking time?",
            "answer": "To start tracking time in Clockify, you need to create a project...",
            "context_docs": [],
            "latency_ms": 350.2,
            "model": "gpt-oss:20b",
        }
    })

    success: bool = Field(description="Whether request succeeded")
    query: str = Field(description="Original query")
    answer: str = Field(description="LLM-generated answer")
    context_docs: List[SearchResult] = Field(description="Context documents used")
    latency_ms: float = Field(ge=0, description="Total latency in milliseconds")
    model: str = Field(description="LLM model used")
    metadata: Optional[ResponseMetadata] = Field(default=None, description="Response generation metadata")


class ErrorResponse(BaseModel):
    """Error response."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": False,
            "error": "Invalid query: too long",
            "error_code": "VALIDATION_ERROR",
            "error_type": "ValidationError",
            "details": {"max_length": 2000, "actual_length": 3000},
            "request_id": "req_123456",
        }
    })

    success: bool = Field(default=False, description="Always False for errors")
    error: str = Field(description="Error message")
    error_code: str = Field(description="Machine-readable error code")
    error_type: str = Field(description="Type of error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request ID for debugging")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "components": {
                "faiss_index": "healthy",
                "embedding_model": "healthy",
                "llm_client": "healthy",
                "cache": "healthy",
            },
            "version": "1.0.0",
            "uptime_seconds": 3600.0,
        }
    })

    status: Literal["healthy", "degraded", "unhealthy"] = Field(description="System health status")
    components: Dict[str, str] = Field(description="Health of individual components")
    version: str = Field(description="System version")
    uptime_seconds: float = Field(ge=0, description="System uptime in seconds")


# ============================================================================
# Internal Processing Models
# ============================================================================


class RetrievalResult(BaseModel):
    """Raw retrieval result before formatting."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    similarity_score: float = Field(ge=0.0, le=1.0)
    bm25_score: Optional[float] = Field(default=None, ge=0.0)
    final_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""

    text: str = Field(description="Text that was embedded")
    embedding: List[float] = Field(description="768-dimensional embedding vector")
    model: str = Field(description="Model used for embedding")
    latency_ms: float = Field(ge=0, description="Embedding latency")


class RerankingResult(BaseModel):
    """Result after reranking operation."""

    original_rank: int = Field(ge=0, description="Original rank")
    new_rank: int = Field(ge=0, description="Rank after reranking")
    original_score: float = Field(ge=0.0, le=1.0)
    new_score: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = Field(default=None, description="Why reranking occurred")


# ============================================================================
# Configuration Models
# ============================================================================


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    model: str = Field(default="nomic-embed-text:latest")
    dimension: int = Field(default=768, ge=1)
    batch_size: int = Field(default=32, ge=1, le=256)
    cache_size: int = Field(default=512, ge=1)
    timeout_seconds: int = Field(default=30, ge=10)


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    k: int = Field(default=5, ge=1, le=100)
    oversampling_factor: float = Field(default=2.0, ge=1.0)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for semantic vs keyword")
    clustering_enabled: bool = Field(default=True)
    clustering_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """LLM configuration."""

    base_url: str = Field(default="http://10.127.0.192:11434")
    model: str = Field(default="gpt-oss:20b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=10)
    max_retries: int = Field(default=3, ge=0)
    backoff_seconds: float = Field(default=0.75, ge=0.0)


class RAGSystemConfig(BaseModel):
    """Complete RAG system configuration."""

    model_config = ConfigDict(validate_assignment=True)

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api_token: str = Field(default="change-me", description="API authentication token")
    namespaces: List[str] = Field(default_factory=lambda: ["clockify", "langchain"])
    environment: Literal["dev", "staging", "prod"] = Field(default="dev")
    mock_llm: bool = Field(default=False, description="Use mock LLM instead of real")
