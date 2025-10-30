"""
Citation Validator for RAG Responses

Validates that LLM responses properly cite their sources:
- Checks that all citations [1], [2], etc. have corresponding sources
- Detects missing or invalid citation numbers
- Validates citation format and sequencing
- Provides detailed validation reports for debugging

This improves response quality by ensuring proper source attribution.
"""

from __future__ import annotations

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    is_valid: bool
    cited_indices: Set[int]  # Citation numbers found in response
    available_indices: Set[int]  # Source indices available
    missing_citations: Set[int]  # Citations without sources
    unused_sources: Set[int]  # Sources not cited
    invalid_citations: List[str]  # Malformed citations
    warnings: List[str]  # Non-critical issues
    total_citations: int  # Total citation occurrences


def extract_citation_numbers(text: str) -> List[int]:
    """
    Extract all citation numbers from text like [1], [2], [3].

    Args:
        text: Response text containing citations

    Returns:
        List of citation numbers found (may contain duplicates)

    Examples:
        >>> extract_citation_numbers("According to [1], the answer is [2].")
        [1, 2]
        >>> extract_citation_numbers("See [1] and [2] for details. Also [1].")
        [1, 2, 1]
    """
    # Match [N] where N is 1-3 digits (supports up to 999 sources)
    pattern = r'\[(\d{1,3})\]'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


def extract_inline_citations(text: str) -> Set[int]:
    """
    Extract unique citation numbers from inline citations in the response body.

    Excludes citations in the "Sources:" section to avoid double-counting.

    Args:
        text: Full response text

    Returns:
        Set of unique citation numbers from inline citations only
    """
    # Split at "Sources:" or similar markers to isolate response body
    # Common patterns: "Sources:", "References:", "Citations:"
    split_markers = [
        "\nSources:",
        "\n\nSources:",
        "\nReferences:",
        "\n\nReferences:",
        "\nCitations:",
        "\n\nCitations:"
    ]

    body = text
    for marker in split_markers:
        if marker in text:
            body = text.split(marker)[0]
            break

    citations = extract_citation_numbers(body)
    return set(citations)


def extract_source_section(text: str) -> Optional[str]:
    """
    Extract the "Sources:" section from the response if it exists.

    Args:
        text: Full response text

    Returns:
        Source section text or None if not found
    """
    # Look for "Sources:" section
    patterns = [
        r'\n\s*Sources:\s*\n(.*)',
        r'\n\s*References:\s*\n(.*)',
        r'\n\s*Citations:\s*\n(.*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(0)

    return None


def validate_citations(response_text: str, num_sources: int, strict: bool = False) -> CitationValidationResult:
    """
    Validate that response citations match available sources.

    Args:
        response_text: LLM response text with citations
        num_sources: Number of sources provided to the LLM
        strict: If True, require all sources to be cited (default: False)

    Returns:
        CitationValidationResult with detailed validation info

    Examples:
        >>> result = validate_citations("Answer from [1] and [2].", num_sources=3)
        >>> result.is_valid
        True
        >>> result.unused_sources
        {3}

        >>> result = validate_citations("Answer from [5].", num_sources=3)
        >>> result.is_valid
        False
        >>> result.missing_citations
        {5}
    """
    # Extract all citation numbers from the full response
    all_citations = extract_citation_numbers(response_text)
    cited_indices = set(all_citations)
    total_citations = len(all_citations)

    # Available source indices (1-indexed)
    available_indices = set(range(1, num_sources + 1)) if num_sources > 0 else set()

    # Find issues
    missing_citations = cited_indices - available_indices  # Citations without sources
    unused_sources = available_indices - cited_indices  # Sources not cited

    # Check for malformed citations (e.g., [0], [abc], empty brackets)
    invalid_citations = []
    invalid_pattern = r'\[(?:0|\D+|)\]'
    invalid_matches = re.findall(invalid_pattern, response_text)
    if invalid_matches:
        invalid_citations.extend(invalid_matches)

    # Collect warnings
    warnings = []

    # Warn if citations are not sequential
    if cited_indices and max(cited_indices) > num_sources:
        warnings.append(
            f"Citation number {max(cited_indices)} exceeds available sources ({num_sources})"
        )

    # Warn if citation numbers skip (e.g., [1], [3] but no [2])
    if cited_indices and num_sources > 0:
        expected_range = set(range(1, max(cited_indices) + 1))
        skipped = expected_range - cited_indices
        if skipped and max(cited_indices) <= num_sources:
            warnings.append(f"Citation sequence has gaps: missing {sorted(skipped)}")

    # Warn if no citations found
    if total_citations == 0 and num_sources > 0:
        warnings.append("Response contains no citations despite having sources")

    # Strict mode: warn about unused sources
    if strict and unused_sources:
        warnings.append(f"Not all sources were cited: unused {sorted(unused_sources)}")

    # Determine overall validity
    # Invalid if: citations reference non-existent sources OR has malformed citations
    is_valid = len(missing_citations) == 0 and len(invalid_citations) == 0

    # Log validation results
    if not is_valid:
        logger.warning(
            f"Citation validation failed: "
            f"missing={missing_citations}, invalid={invalid_citations}"
        )
    elif warnings:
        logger.debug(f"Citation validation warnings: {warnings}")

    return CitationValidationResult(
        is_valid=is_valid,
        cited_indices=cited_indices,
        available_indices=available_indices,
        missing_citations=missing_citations,
        unused_sources=unused_sources,
        invalid_citations=invalid_citations,
        warnings=warnings,
        total_citations=total_citations,
    )


def validate_response_with_sources(
    response_text: str,
    sources: List[Dict[str, str]],
    strict: bool = False
) -> CitationValidationResult:
    """
    Convenience function to validate response against actual source list.

    Args:
        response_text: LLM response with citations
        sources: List of source dictionaries (with 'text' or 'content' keys)
        strict: Enable strict validation

    Returns:
        CitationValidationResult
    """
    return validate_citations(response_text, len(sources), strict=strict)


def format_validation_report(result: CitationValidationResult) -> str:
    """
    Format validation result as human-readable report.

    Args:
        result: CitationValidationResult to format

    Returns:
        Formatted string report
    """
    lines = []
    lines.append(f"✓ Valid: {result.is_valid}")
    lines.append(f"  Citations found: {sorted(result.cited_indices) if result.cited_indices else 'none'}")
    lines.append(f"  Sources available: {sorted(result.available_indices) if result.available_indices else 'none'}")
    lines.append(f"  Total citation occurrences: {result.total_citations}")

    if result.missing_citations:
        lines.append(f"  ❌ Missing sources for citations: {sorted(result.missing_citations)}")

    if result.invalid_citations:
        lines.append(f"  ❌ Invalid citation format: {result.invalid_citations}")

    if result.unused_sources:
        lines.append(f"  ⚠️  Unused sources: {sorted(result.unused_sources)}")

    if result.warnings:
        for warning in result.warnings:
            lines.append(f"  ⚠️  {warning}")

    return "\n".join(lines)


# =============================================================================
# Automatic Validation Integration
# =============================================================================

def auto_validate_response(response_text: str, num_sources: int) -> Tuple[str, bool]:
    """
    Automatically validate response and optionally append validation status.

    Args:
        response_text: LLM response to validate
        num_sources: Number of sources provided

    Returns:
        Tuple of (response_text, is_valid)
    """
    result = validate_citations(response_text, num_sources, strict=False)

    # Only log validation failures, not every validation
    if not result.is_valid:
        report = format_validation_report(result)
        logger.warning(f"Response citation validation failed:\n{report}")

    return response_text, result.is_valid
