from __future__ import annotations

"""
Centralized logging configuration for RAG system.

Provides unified logging across all modules using loguru.
Supports both console and file output with structured logging.
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
from src.config import CONFIG, redact_secrets


def setup_logging() -> None:
    """
    Configure unified logging for the entire RAG system.

    This should be called once at application startup.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format=(
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=CONFIG.LOG_LEVEL,
        colorize=True,
    )

    # File handler if LOG_FILE is set
    if CONFIG.LOG_FILE:
        logger.add(
            CONFIG.LOG_FILE,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            ),
            level=CONFIG.LOG_LEVEL,
            rotation="500 MB",
            retention="7 days",
            compression="zip",
        )

    logger.info(f"Logging configured: level={CONFIG.LOG_LEVEL}")


def log_structured(event_type: str, data: Dict[str, Any], level: str = "info") -> None:
    """
    Log structured data as JSON.

    Args:
        event_type: Type of event (e.g., 'search_completed', 'error_occurred')
        data: Dictionary of data to log
        level: Log level (debug, info, warning, error, critical)
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        **data,
    }

    # Redact sensitive information
    if "error" in log_entry and log_entry["error"]:
        log_entry["error"] = redact_secrets(log_entry["error"])

    log_func = getattr(logger, level.lower(), logger.info)
    log_func(json.dumps(log_entry))


def log_search(
    request_id: str,
    query: str,
    latency_ms: int,
    results_count: int,
    namespace: Optional[str] = None,
    k: Optional[int] = None,
) -> None:
    """Log a search operation."""
    log_structured(
        "search_completed",
        {
            "request_id": request_id,
            "query": query[:100],  # Truncate long queries
            "latency_ms": latency_ms,
            "results_count": results_count,
            "namespace": namespace,
            "k": k,
        },
    )


def log_embedding(query: str, latency_ms: int, cache_hit: bool = False) -> None:
    """Log an embedding operation."""
    log_structured(
        "embedding_generated",
        {
            "query": query[:100],
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
        },
        level="debug",
    )


def log_llm_call(
    request_id: str,
    prompt_tokens: int,
    response_tokens: int,
    latency_ms: int,
    model: Optional[str] = None,
) -> None:
    """Log an LLM API call."""
    log_structured(
        "llm_call_completed",
        {
            "request_id": request_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "latency_ms": latency_ms,
        },
    )


def log_error(error_type: str, message: str, request_id: Optional[str] = None, **kwargs: Any) -> None:
    """Log an error with context."""
    log_structured(
        "error_occurred",
        {
            "error_type": error_type,
            "message": message,
            "request_id": request_id,
            **kwargs,
        },
        level="error",
    )
