#!/usr/bin/env python3
"""
Full-Article Context Builder for RAG.

Transforms chunk-based search results into full-article context:
1. Groups chunks by URL to reconstruct complete articles
2. Loads FULL text from metadata (not truncated 1600 chars)
3. Applies adaptive article count based on query intent
4. Builds enriched context for LLM with full articles

This is a key component of Phase 2: moving from chunks to full articles.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from loguru import logger


@dataclass
class Article:
    """Reconstructed article from chunks."""

    url: str
    title: str
    full_text: str
    chunks: List[Dict[str, Any]]  # Original chunks
    score: float  # Aggregated relevance score
    breadcrumb: str  # Navigation path
    word_count: int
    char_count: int


@dataclass
class ArticleContext:
    """Full-article context for LLM."""

    articles: List[Article]
    full_text: str  # Concatenated article text with citations
    urls: List[str]
    count: int
    total_chars: int
    truncated: bool  # Whether any article was truncated due to length limits


def build_full_article_context(
    search_results: List[Dict[str, Any]],
    intent: Literal["how_to", "comparison", "factual", "definition", "general"],
    adaptive_count: bool = True,
    max_article_length: Optional[int] = None,
    article_count_config: Optional[Dict[str, int]] = None
) -> ArticleContext:
    """
    Build full-article context from search results.

    Args:
        search_results: List of chunk results from hybrid search
        intent: Query intent type
        adaptive_count: Whether to adapt article count based on intent
        max_article_length: Max characters per article (safety limit)
        article_count_config: Custom article counts per intent

    Returns:
        ArticleContext with full articles
    """
    logger.info(
        f"Building full-article context: intent={intent}, "
        f"adaptive={adaptive_count}, results={len(search_results)}"
    )

    # Get max article length from env or param
    if max_article_length is None:
        max_article_length = int(os.getenv("MAX_ARTICLE_LENGTH", "50000"))

    # Determine article count
    if adaptive_count:
        from src.intent_analyzer import get_adaptive_article_count
        target_count = get_adaptive_article_count(intent, article_count_config)
    else:
        target_count = int(os.getenv("ARTICLE_COUNT_GENERAL", "4"))

    logger.debug(f"Target article count: {target_count}")

    # Group chunks by URL
    articles_by_url = _group_chunks_by_url(search_results)

    logger.debug(f"Grouped into {len(articles_by_url)} unique articles")

    # Reconstruct articles
    articles = []
    for url, chunks in articles_by_url.items():
        article = _reconstruct_article(url, chunks, max_article_length)
        articles.append(article)

    # Sort by aggregated score (descending)
    articles.sort(key=lambda a: a.score, reverse=True)

    # Take top N articles based on intent
    top_articles = articles[:target_count]

    logger.info(
        f"Selected {len(top_articles)} articles "
        f"(avg score: {sum(a.score for a in top_articles) / len(top_articles):.3f})"
    )

    # Build concatenated context text
    full_text, truncated = _build_concatenated_context(top_articles)

    # Extract URLs
    urls = [article.url for article in top_articles]

    # Calculate total chars
    total_chars = sum(article.char_count for article in top_articles)

    context = ArticleContext(
        articles=top_articles,
        full_text=full_text,
        urls=urls,
        count=len(top_articles),
        total_chars=total_chars,
        truncated=truncated
    )

    logger.info(
        f"Article context built: {context.count} articles, "
        f"{context.total_chars:,} chars, truncated={context.truncated}"
    )

    return context


def _group_chunks_by_url(
    search_results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group search result chunks by URL.

    Args:
        search_results: List of chunk results

    Returns:
        Dict mapping URL to list of chunks
    """
    grouped = defaultdict(list)

    for result in search_results:
        url = result.get("url", "")
        if url:
            grouped[url].append(result)

    return dict(grouped)


def _reconstruct_article(
    url: str,
    chunks: List[Dict[str, Any]],
    max_length: int
) -> Article:
    """
    Reconstruct full article from chunks.

    Args:
        url: Article URL
        chunks: List of chunks for this URL
        max_length: Max character length (safety limit)

    Returns:
        Reconstructed Article object
    """
    # Extract metadata from first chunk
    first_chunk = chunks[0]
    title = first_chunk.get("title", first_chunk.get("h1", "Untitled"))
    title_path = first_chunk.get("title_path", [])
    breadcrumb = " > ".join(title_path) if title_path else title

    # Aggregate relevance scores (average of chunk scores)
    scores = [chunk.get("score", 0.0) for chunk in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Concatenate FULL TEXT from all chunks (not truncated!)
    # Sort chunks by order (if available) or keep search result order
    chunks_sorted = sorted(
        chunks,
        key=lambda c: c.get("chunk_order", c.get("tokens", 0))
    )

    text_parts = []
    total_chars = 0

    for chunk in chunks_sorted:
        chunk_text = chunk.get("text", "")

        # Check length limit
        if total_chars + len(chunk_text) > max_length:
            logger.warning(
                f"Article {url} exceeds max length ({max_length} chars), truncating"
            )
            remaining = max_length - total_chars
            if remaining > 100:  # Only add if meaningful amount remains
                text_parts.append(chunk_text[:remaining] + "...")
            break

        text_parts.append(chunk_text)
        total_chars += len(chunk_text)

    full_text = "\n\n".join(text_parts)

    # Count words
    word_count = len(full_text.split())

    article = Article(
        url=url,
        title=title,
        full_text=full_text,
        chunks=chunks,
        score=avg_score,
        breadcrumb=breadcrumb,
        word_count=word_count,
        char_count=len(full_text)
    )

    logger.debug(
        f"Reconstructed article: {title[:50]}... "
        f"({article.word_count} words, {article.char_count} chars, "
        f"score={article.score:.3f})"
    )

    return article


def _build_concatenated_context(articles: List[Article]) -> tuple[str, bool]:
    """
    Build concatenated context text with citation markers.

    Args:
        articles: List of Article objects

    Returns:
        Tuple of (context_text, truncated_flag)
    """
    context_parts = []
    truncated = False

    for i, article in enumerate(articles, 1):
        # Citation header
        header = f"[{i}] {article.breadcrumb}\nURL: {article.url}\n"

        # Check if article was truncated during reconstruction
        if article.full_text.endswith("..."):
            truncated = True

        # Combine
        context_parts.append(f"{header}\n{article.full_text}")

    full_text = "\n\n---\n\n".join(context_parts)

    return full_text, truncated


def format_articles_for_display(articles: List[Article]) -> List[Dict[str, Any]]:
    """
    Format articles for API response (sources field).

    Args:
        articles: List of Article objects

    Returns:
        List of source dicts for ChatResponse
    """
    sources = []

    for article in articles:
        source = {
            "url": article.url,
            "title": article.title,
            "breadcrumb": article.breadcrumb,
            "score": article.score,
            "word_count": article.word_count,
            "char_count": article.char_count,
        }
        sources.append(source)

    return sources


def get_article_statistics(context: ArticleContext) -> Dict[str, Any]:
    """
    Get statistics about article context.

    Args:
        context: ArticleContext object

    Returns:
        Statistics dict
    """
    if not context.articles:
        return {
            "count": 0,
            "total_chars": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "truncated": False
        }

    char_counts = [a.char_count for a in context.articles]

    return {
        "count": context.count,
        "total_chars": context.total_chars,
        "avg_chars": sum(char_counts) // len(char_counts),
        "min_chars": min(char_counts),
        "max_chars": max(char_counts),
        "truncated": context.truncated
    }


def preview_context(context: ArticleContext, max_preview_chars: int = 200) -> str:
    """
    Generate preview of context for logging/debugging.

    Args:
        context: ArticleContext object
        max_preview_chars: Max chars per article preview

    Returns:
        Preview string
    """
    lines = [
        f"Article Context ({context.count} articles, {context.total_chars:,} chars):",
        ""
    ]

    for i, article in enumerate(context.articles, 1):
        preview = article.full_text[:max_preview_chars]
        if len(article.full_text) > max_preview_chars:
            preview += "..."

        lines.append(f"[{i}] {article.title}")
        lines.append(f"    URL: {article.url}")
        lines.append(f"    Score: {article.score:.3f}")
        lines.append(f"    Length: {article.char_count:,} chars")
        lines.append(f"    Preview: {preview}")
        lines.append("")

    return "\n".join(lines)