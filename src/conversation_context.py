#!/usr/bin/env python3
"""
Conversation Context Builder for multi-turn RAG.

Builds LLM prompts that include conversation history, enabling context-aware responses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from loguru import logger


def build_conversation_prompt(
    current_question: str,
    conversation_history: List[Dict[str, Any]],
    retrieved_context: str,
    sources: List[Dict[str, Any]],
    namespace: str = "clockify",
    max_history_turns: int = 3
) -> Tuple[List[Dict[str, str]], str]:
    """
    Build multi-turn conversation messages for LLM.

    Includes conversation history to maintain context across turns.
    Prevents context window explosion by limiting history to recent turns.

    Args:
        current_question: User's current question
        conversation_history: List of previous turns (from session)
        retrieved_context: Context retrieved for current question
        sources: List of source documents with metadata
        namespace: Knowledge base namespace
        max_history_turns: Maximum number of previous turns to include (default: 3)

    Returns:
        Tuple of (messages, developer_instructions)
        - messages: List of OpenAI-style message dicts for LLM
        - developer_instructions: System prompt for developer visibility

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": "How to create project?"},
            {"role": "assistant", "content": "To create a project..."},
            {"role": "user", "content": "How do I add members?"}  # Remembers project context
        ]
    """

    # Build system prompt
    system_prompt = _build_system_prompt(namespace)

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Include recent conversation history (limit to prevent context explosion)
    history_to_include = conversation_history[-max_history_turns:] if conversation_history else []

    for turn in history_to_include:
        # Add user question from previous turn
        messages.append({
            "role": "user",
            "content": turn["user_question"]
        })

        # Add assistant answer from previous turn
        messages.append({
            "role": "assistant",
            "content": turn["llm_answer"]
        })

    logger.debug(
        f"Including {len(history_to_include)} previous turns in conversation context "
        f"(max={max_history_turns})"
    )

    # Build current turn prompt with retrieved context
    current_turn_content = _build_current_turn_prompt(
        question=current_question,
        context=retrieved_context,
        sources=sources,
        is_followup=len(history_to_include) > 0
    )

    messages.append({
        "role": "user",
        "content": current_turn_content
    })

    # Developer instructions (for logging/debugging)
    developer_instructions = _build_developer_instructions(
        namespace=namespace,
        has_history=len(history_to_include) > 0,
        turn_number=len(conversation_history) + 1
    )

    return messages, developer_instructions


def _build_system_prompt(namespace: str) -> str:
    """
    Build system prompt for conversational RAG.

    Args:
        namespace: Knowledge base namespace

    Returns:
        System prompt string
    """
    return f"""You are a helpful assistant for {namespace} documentation.

Your role:
1. Answer questions based on the provided documentation context
2. If this is a follow-up question, refer to the previous conversation
3. Use numbered citations [1], [2] to reference sources
4. If the documentation doesn't contain enough information, say so clearly
5. Maintain a helpful, professional tone

Remember:
- Base answers on provided documentation only
- Use citations for all factual claims
- If context is missing, don't make assumptions
- For follow-ups, reference previous answers when relevant
"""


def _build_current_turn_prompt(
    question: str,
    context: str,
    sources: List[Dict[str, Any]],
    is_followup: bool = False
) -> str:
    """
    Build prompt for the current conversation turn.

    Args:
        question: User's current question
        context: Retrieved documentation context
        sources: List of source documents
        is_followup: Whether this is a follow-up question

    Returns:
        Formatted prompt string
    """
    # Build sources list for citations
    sources_text = _format_sources(sources)

    if is_followup:
        prompt = f"""This is a follow-up question. Please consider the previous conversation when answering.

Question: {question}

Relevant documentation:
{context}

Sources:
{sources_text}

Answer the question based on the documentation, using [1], [2], etc. citations.
If this question refers to the previous answer, incorporate that context naturally."""
    else:
        prompt = f"""Question: {question}

Relevant documentation:
{context}

Sources:
{sources_text}

Answer the question based on the documentation, using [1], [2], etc. citations."""

    return prompt


def _format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list for inclusion in prompt.

    Args:
        sources: List of source documents with metadata

    Returns:
        Formatted sources string
    """
    if not sources:
        return "(No sources available)"

    lines = []
    for i, source in enumerate(sources, 1):
        title = source.get("title", source.get("h1", "Untitled"))
        url = source.get("url", "")
        lines.append(f"[{i}] {title}\n    {url}")

    return "\n".join(lines)


def _build_developer_instructions(
    namespace: str,
    has_history: bool,
    turn_number: int
) -> str:
    """
    Build developer-facing instructions for logging/debugging.

    Args:
        namespace: Knowledge base namespace
        has_history: Whether conversation has previous turns
        turn_number: Current turn number in conversation

    Returns:
        Developer instructions string
    """
    mode = "Multi-turn conversation" if has_history else "Single-turn Q&A"

    return f"""RAG Mode: {mode}
Namespace: {namespace}
Turn: {turn_number}
Context: {"Conversation history included" if has_history else "Fresh conversation"}
"""


def summarize_conversation_for_retrieval(
    current_question: str,
    conversation_history: List[Dict[str, Any]],
    max_turns_to_consider: int = 2
) -> str:
    """
    Summarize conversation history to improve retrieval for follow-up questions.

    For example:
    - Turn 1: "How to create project?"
    - Turn 2: "How do I add members?"
    - Summary: "How to add members to a project?"

    Args:
        current_question: User's current question
        conversation_history: Previous conversation turns
        max_turns_to_consider: How many recent turns to consider

    Returns:
        Expanded query that includes conversation context
    """
    if not conversation_history:
        return current_question

    # Get recent turns
    recent_turns = conversation_history[-max_turns_to_consider:]

    # Extract keywords from previous questions
    previous_keywords = []
    for turn in recent_turns:
        question = turn.get("user_question", "")
        # Simple keyword extraction (can be enhanced with NLP)
        keywords = turn.get("keywords", [])
        if keywords:
            previous_keywords.extend(keywords)

    # If current question is short and vague, expand with context
    if len(current_question.split()) <= 5 and previous_keywords:
        # Question likely references previous context
        logger.debug(
            f"Expanding vague follow-up question with context: '{current_question}' "
            f"+ keywords={previous_keywords[:3]}"
        )
        expanded = f"{current_question} {' '.join(previous_keywords[:3])}"
        return expanded

    return current_question
