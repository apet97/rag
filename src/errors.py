"""
Structured Error Handling for RAG System

Provides a hierarchy of exceptions for different failure scenarios
with clear semantics for retry behavior and error handling.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RAGError(Exception):
    """
    Base exception for RAG system.

    All RAG-specific errors should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RAG_ERROR",
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize RAG error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
            severity: Error severity level
            context: Additional context for debugging
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "context": self.context,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.error_code}: {self.message})"


class RetriableError(RAGError):
    """
    Error that should be retried with backoff.

    Typically temporary issues like timeouts, rate limits, transient failures.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "RETRIABLE_ERROR",
        max_retries: int = 3,
        backoff_seconds: int = 1,
        **kwargs,
    ):
        super().__init__(message, error_code, severity=ErrorSeverity.MEDIUM, **kwargs)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds


class NonRetriableError(RAGError):
    """
    Error that should NOT be retried.

    Typically permanent issues like invalid API key, bad configuration, malformed input.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "NON_RETRIABLE_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, severity=ErrorSeverity.HIGH, **kwargs)


# ============================================================================
# Specific Error Types
# ============================================================================


class ConfigurationError(NonRetriableError):
    """Invalid or missing configuration"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class ValidationError(NonRetriableError):
    """Input validation failed"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field


class IndexError(RAGError):
    """Index-related errors (loading, searching, building)"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_code="INDEX_ERROR", severity=ErrorSeverity.CRITICAL, **kwargs
        )


class RetrievalError(RetriableError):
    """Vector or lexical search failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RETRIEVAL_ERROR", **kwargs)


class RetrievalTimeoutError(RetrievalError):
    """Retrieval operation timed out"""

    def __init__(self, message: str, timeout_seconds: float = 30, **kwargs):
        super().__init__(message, error_code="RETRIEVAL_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class RankingError(RetriableError):
    """Reranking or scoring failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RANKING_ERROR", **kwargs)


class QueryOptimizationError(RetriableError):
    """Query analysis or expansion failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="QUERY_OPTIMIZATION_ERROR", **kwargs)


class EmbeddingError(RetriableError):
    """Embedding model error"""

    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="EMBEDDING_ERROR", **kwargs)
        self.model = model


class LLMError(RetriableError):
    """LLM API error"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)
        self.status_code = status_code
        self.model = model


class LLMConnectionError(RetriableError):
    """Cannot connect to LLM service"""

    def __init__(
        self,
        message: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_CONNECTION_ERROR", **kwargs)
        self.base_url = base_url


class LLMTimeoutError(LLMError):
    """LLM request timed out"""

    def __init__(self, message: str, timeout_seconds: float = 30, **kwargs):
        super().__init__(message, error_code="LLM_TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded"""

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="LLM_RATE_LIMIT", **kwargs)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthenticationError(LLMError):
    """LLM authentication failed (invalid API key, etc)"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="LLM_AUTH_ERROR", **kwargs)


class CacheError(RAGError):
    """Cache operation failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CACHE_ERROR", severity=ErrorSeverity.LOW, **kwargs)


class CircuitOpenError(NonRetriableError):
    """Circuit breaker is open - service temporarily unavailable"""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CIRCUIT_OPEN", **kwargs)
        self.service = service


class DependencyError(RetriableError):
    """External dependency is unavailable"""

    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        self.dependency = dependency


# ============================================================================
# Error Utilities
# ============================================================================


def is_retryable(error: Exception) -> bool:
    """Check if error should be retried"""
    return isinstance(error, RetriableError)


def is_fatal(error: Exception) -> bool:
    """Check if error is fatal and cannot be recovered"""
    return isinstance(error, NonRetriableError)


def get_error_severity(error: Exception) -> ErrorSeverity:
    """Get severity level of error"""
    if isinstance(error, RAGError):
        return error.severity
    return ErrorSeverity.HIGH  # Default for non-RAG exceptions


def format_error_for_logging(error: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format error for structured logging"""
    result: Dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": str(error),
    }

    if request_id:
        result["request_id"] = request_id

    if isinstance(error, RAGError):
        result.update(error.to_dict())

    if error.__cause__:
        result["caused_by"] = {
            "type": type(error.__cause__).__name__,
            "message": str(error.__cause__),
        }

    return result
