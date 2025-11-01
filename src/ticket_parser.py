#!/usr/bin/env python3
"""
Support Ticket Parser for RAG system.

Detects and parses support ticket format to extract structured information:
- Issue description
- Steps to reproduce
- Error messages
- Environment details (browser, OS, version)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class TicketData:
    """Structured data extracted from support ticket."""

    issue_description: str
    steps_to_reproduce: List[str]
    error_messages: List[str]
    environment: Dict[str, str]
    raw_text: str
    confidence: float  # 0.0-1.0, how confident we are this is a ticket


def is_ticket(text: str) -> bool:
    """
    Quick check if text looks like a support ticket.

    Args:
        text: User input text

    Returns:
        True if text has ticket-like patterns
    """
    if not text or len(text) < 50:
        return False

    ticket_indicators = [
        # Multi-line with numbered/bulleted steps
        r'^\d+\.',  # "1. Step one"
        r'^[\*\-]\s',  # "* Step" or "- Step"

        # Error patterns
        r'error:',
        r'exception:',
        r'failed to',
        r'cannot',
        r'not working',

        # Issue reporting patterns
        r'issue:',
        r'problem:',
        r'bug:',
        r'expected:',
        r'actual:',

        # Environment patterns
        r'browser:',
        r'os:',
        r'version:',
        r'chrome|firefox|safari|edge',
        r'windows|mac|linux',

        # Stack trace patterns
        r'at \w+\.\w+',  # "at Object.method"
        r'line \d+',
        r'\.js:\d+',
    ]

    # Count pattern matches
    matches = sum(1 for pattern in ticket_indicators if re.search(pattern, text, re.IGNORECASE | re.MULTILINE))

    # Multi-line structure
    lines = text.split('\n')
    has_multiple_paragraphs = len([l for l in lines if l.strip()]) >= 3

    # Heuristic: 2+ patterns + multi-line = likely a ticket
    return matches >= 2 and has_multiple_paragraphs


def parse_ticket(text: str) -> Optional[TicketData]:
    """
    Parse support ticket text into structured data.

    Args:
        text: Raw ticket text

    Returns:
        TicketData if parsing succeeds, None if not a ticket
    """
    if not is_ticket(text):
        logger.debug("Text does not match ticket patterns")
        return None

    logger.info("Parsing text as support ticket")

    # Initialize containers
    issue_lines = []
    steps = []
    errors = []
    environment = {}

    lines = text.split('\n')
    current_section = "issue"  # Start assuming issue description

    # Section detection patterns
    steps_pattern = re.compile(r'^(steps to reproduce|reproduce|steps):', re.IGNORECASE)
    error_pattern = re.compile(r'^(error|exception|stack trace|logs?):', re.IGNORECASE)
    env_pattern = re.compile(r'^(environment|setup|browser|os|version):', re.IGNORECASE)

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Detect section headers
        if steps_pattern.search(stripped):
            current_section = "steps"
            continue
        elif error_pattern.search(stripped):
            current_section = "errors"
            continue
        elif env_pattern.search(stripped):
            current_section = "environment"
            continue

        # Parse content based on current section
        if current_section == "issue":
            # Extract issue description
            issue_lines.append(stripped)

        elif current_section == "steps":
            # Extract numbered/bulleted steps
            step_match = re.match(r'^[\d\*\-\.]+\s*(.+)$', stripped)
            if step_match:
                steps.append(step_match.group(1))
            else:
                # Continuation of previous step or standalone line
                if steps:
                    steps[-1] += f" {stripped}"
                else:
                    steps.append(stripped)

        elif current_section == "errors":
            # Extract error messages and stack traces
            errors.append(stripped)

        elif current_section == "environment":
            # Extract key-value pairs
            env_match = re.match(r'^([^:]+):\s*(.+)$', stripped)
            if env_match:
                key = env_match.group(1).strip().lower()
                value = env_match.group(2).strip()
                environment[key] = value

    # Heuristic extraction if no explicit sections found
    if not steps:
        steps = _extract_implicit_steps(text)

    if not errors:
        errors = _extract_error_messages(text)

    if not environment:
        environment = _extract_environment(text)

    # Build issue description
    if not issue_lines:
        # Take first paragraph as issue if no explicit section
        for line in lines:
            if line.strip():
                issue_lines.append(line.strip())
            if len(issue_lines) >= 3:  # First few lines
                break

    issue_description = ' '.join(issue_lines)

    # Calculate confidence
    confidence = _calculate_confidence(
        has_steps=bool(steps),
        has_errors=bool(errors),
        has_environment=bool(environment),
        text_length=len(text)
    )

    ticket_data = TicketData(
        issue_description=issue_description,
        steps_to_reproduce=steps,
        error_messages=errors,
        environment=environment,
        raw_text=text,
        confidence=confidence
    )

    logger.info(
        f"Ticket parsed: confidence={confidence:.2f}, "
        f"steps={len(steps)}, errors={len(errors)}, env_keys={len(environment)}"
    )

    return ticket_data


def _extract_implicit_steps(text: str) -> List[str]:
    """
    Extract steps from text without explicit section.

    Looks for numbered lists, bulleted lists, or sequential patterns.
    """
    steps = []

    # Pattern: "1. Step", "2. Step", etc.
    numbered = re.findall(r'^\d+\.\s+(.+)$', text, re.MULTILINE)
    if numbered:
        steps.extend(numbered)

    # Pattern: "- Step", "* Step"
    if not steps:
        bulleted = re.findall(r'^[\*\-]\s+(.+)$', text, re.MULTILINE)
        if bulleted:
            steps.extend(bulleted)

    # Pattern: "First...", "Then...", "Next...", "Finally..."
    if not steps:
        sequential_pattern = re.compile(
            r'(first|then|next|after that|finally)[,:]?\s+(.+)',
            re.IGNORECASE
        )
        sequential = sequential_pattern.findall(text)
        if sequential:
            steps.extend([match[1] for match in sequential])

    return steps


def _extract_error_messages(text: str) -> List[str]:
    """
    Extract error messages from text.

    Looks for common error patterns.
    """
    errors = []

    # Pattern: "Error: message"
    error_pattern = re.compile(
        r'(error|exception|failed|cannot|unable to)[:\s]+(.+)',
        re.IGNORECASE
    )

    for match in error_pattern.finditer(text):
        error_msg = match.group(0).strip()
        if error_msg and len(error_msg) < 500:  # Skip very long lines
            errors.append(error_msg)

    # Pattern: Stack trace lines
    stack_pattern = re.compile(r'^\s+at\s+\S+', re.MULTILINE)
    stack_lines = stack_pattern.findall(text)
    if stack_lines:
        errors.append(f"Stack trace: {len(stack_lines)} frames")

    return errors


def _extract_environment(text: str) -> Dict[str, str]:
    """
    Extract environment details from text.

    Looks for browser, OS, version mentions.
    """
    environment = {}

    # Browser detection
    browser_pattern = re.compile(
        r'\b(chrome|firefox|safari|edge|brave|opera)[\s/]*([\d\.]+)?',
        re.IGNORECASE
    )
    browser_match = browser_pattern.search(text)
    if browser_match:
        browser = browser_match.group(1)
        version = browser_match.group(2) or "unknown"
        environment['browser'] = f"{browser} {version}"

    # OS detection
    os_pattern = re.compile(
        r'\b(windows|mac\s?os|linux|ubuntu|android|ios)[\s]*(\d+)?',
        re.IGNORECASE
    )
    os_match = os_pattern.search(text)
    if os_match:
        os_name = os_match.group(1).strip()
        os_version = os_match.group(2) or ""
        environment['os'] = f"{os_name} {os_version}".strip()

    # Version pattern: "Version: X.Y.Z"
    version_pattern = re.compile(r'version[:\s]+([v\d\.]+)', re.IGNORECASE)
    version_match = version_pattern.search(text)
    if version_match:
        environment['version'] = version_match.group(1)

    return environment


def _calculate_confidence(
    has_steps: bool,
    has_errors: bool,
    has_environment: bool,
    text_length: int
) -> float:
    """
    Calculate confidence that text is a support ticket.

    Args:
        has_steps: Whether steps were extracted
        has_errors: Whether errors were found
        has_environment: Whether environment details found
        text_length: Length of text in characters

    Returns:
        Confidence score 0.0-1.0
    """
    confidence = 0.4  # Base confidence (since is_ticket() already matched patterns)

    if has_steps:
        confidence += 0.2
    if has_errors:
        confidence += 0.2
    if has_environment:
        confidence += 0.1

    # Length bonus: tickets are typically 100-2000 chars
    if 100 <= text_length <= 2000:
        confidence += 0.1
    elif text_length > 2000:
        confidence += 0.05  # Very long might be a ticket

    return min(confidence, 1.0)


def format_ticket_for_query(ticket: TicketData) -> str:
    """
    Format parsed ticket into a query string for RAG retrieval.

    Prioritizes issue description and error messages.

    Args:
        ticket: Parsed ticket data

    Returns:
        Formatted query string
    """
    parts = [ticket.issue_description]

    # Add first error (most relevant)
    if ticket.error_messages:
        parts.append(ticket.error_messages[0])

    # Add first step (context)
    if ticket.steps_to_reproduce:
        parts.append(ticket.steps_to_reproduce[0])

    query = ' '.join(parts)

    # Limit length (avoid very long queries)
    if len(query) > 500:
        query = query[:500] + "..."

    logger.debug(f"Formatted ticket query: {query[:100]}...")

    return query
