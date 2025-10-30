#!/usr/bin/env python3
"""RAG prompt templates with inline citations.

Implements the Clockify RAG Standard v1 with strict context adherence,
parameter quoting, and consistent citation format across channels.
"""

from typing import List, Dict, Any, Optional
import re

from src.config import CONFIG


class RAGPrompt:
    """Build RAG prompts with citations and strict adherence to system instructions."""

    # System prompt enforces: strict context use, "Not in docs" fallback, inline citations
    SYSTEM_PROMPT_CLOCKIFY = """You are an expert technical assistant for Clockify Help documentation.

INSTRUCTIONS:
1. Answer ONLY from the provided context. Do NOT use prior knowledge.
2. Be accurate, concise, and direct.
3. Use inline citations [1], [2] tied to the Sources section.
4. If asked about something not in the docs, respond: "Not in docs" and suggest related topics.
5. When mentioning API parameters, functions, or settings, use QUOTES: "parameter_name"
6. Format: Answer first, then numbered Sources.
7. Respect breadcrumbs: cite section titles for better navigation (e.g., [1] Administration > User Roles).
"""

    # Developer role (Harmony format): RAG-specific enforcement rules for grounding and citation
    DEVELOPER_INSTRUCTIONS_CLOCKIFY = """You are grounding RAG responses to Clockify documentation.

ENFORCEMENT:
1. CITE ONLY: Every factual claim must reference a numbered source [1], [2], etc.
2. GROUND STRICTLY: Never use knowledge outside the provided context blocks.
3. OUT OF SCOPE: If information is not in the docs, respond with "Not in docs" before suggesting alternatives.
4. PARAMETER QUOTING: Quote all API parameters, field names, and settings: "parameter_name"
5. NO SYNTHESIS: Do not combine information across sources to infer new facts. Present each source independently.
6. REASONING: Use low reasoning effort to minimize response latency; prioritize speed over depth."""

    SYSTEM_PROMPT_LANGCHAIN = """You are an expert technical assistant for LangChain documentation.

INSTRUCTIONS:
1. Answer ONLY from the provided context. Do NOT use prior knowledge.
2. Be accurate, concise, and direct.
3. Use inline citations [1], [2] tied to the Sources section.
4. If asked about something not in the docs, respond: "Not in docs" and suggest related topics.
5. When mentioning functions, modules, or API elements, use QUOTES: "function_name"
6. Format: Answer first, then numbered Sources.
"""

    # Developer role (Harmony format): RAG-specific enforcement rules for LangChain
    DEVELOPER_INSTRUCTIONS_LANGCHAIN = """You are grounding RAG responses to LangChain documentation.

ENFORCEMENT:
1. CITE ONLY: Every factual claim must reference a numbered source [1], [2], etc.
2. GROUND STRICTLY: Never use knowledge outside the provided context blocks.
3. OUT OF SCOPE: If information is not in the docs, respond with "Not in docs" before suggesting alternatives.
4. CODE EXAMPLES: Quote all function names, modules, and class names: "function_name"
5. NO SYNTHESIS: Do not combine information across sources to infer new facts. Present each source independently.
6. REASONING: Use low reasoning effort to minimize response latency; prioritize speed over depth."""

    # P2: Query decomposition prompts for breaking down complex questions
    DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant. Your task is to break down complex questions \
into focused sub-questions that can be searched independently. Return ONLY a JSON array \
of strings, one per line, with no markdown formatting or explanation. Example:
["What is kiosk?", "What is timer?", "How do they compare?"]"""

    @staticmethod
    def get_decomposition_prompts(query: str, max_subtasks: int = 3) -> tuple[str, str]:
        """Get system and user prompts for query decomposition.

        Args:
            query: The complex question to decompose
            max_subtasks: Maximum number of sub-questions to generate

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = RAGPrompt.DECOMPOSITION_SYSTEM_PROMPT
        user_prompt = (
            f"Break down this complex question into at most {max_subtasks} focused sub-questions "
            f"that can be searched independently. Return only the JSON array, no explanation.\n\n"
            f"Question: {query}"
        )
        return system_prompt, user_prompt

    @staticmethod
    def get_system_prompt(namespace: str = "clockify") -> str:
        """Get system prompt for a specific namespace."""
        if namespace == "langchain":
            return RAGPrompt.SYSTEM_PROMPT_LANGCHAIN
        return RAGPrompt.SYSTEM_PROMPT_CLOCKIFY

    @staticmethod
    def get_developer_instructions(namespace: str = "clockify") -> str:
        """Get developer role instructions for Harmony format (RAG-specific enforcement).

        Args:
            namespace: Documentation namespace (clockify, langchain, etc.)

        Returns:
            Developer role instructions for RAG grounding and citation enforcement
        """
        if namespace == "langchain":
            return RAGPrompt.DEVELOPER_INSTRUCTIONS_LANGCHAIN
        return RAGPrompt.DEVELOPER_INSTRUCTIONS_CLOCKIFY

    @staticmethod
    def build_context_block(chunks: List[Dict[str, Any]], max_chunks: int = None) -> tuple[str, List[Dict]]:
        """Format chunks as numbered context blocks with breadcrumb titles.

        Args:
            chunks: List of chunk dictionaries from retrieval
            max_chunks: Maximum number of chunks to include (default from CONFIG.MAX_CONTEXT_CHUNKS)

        Returns:
            Tuple of (formatted_context_string, sources_list)
        """
        # Use config default if not specified
        if max_chunks is None:
            max_chunks = CONFIG.MAX_CONTEXT_CHUNKS

        sources = []
        context_lines = []

        # Limit to max_chunks
        chunks_to_use = chunks[:max_chunks]

        for idx, chunk in enumerate(chunks_to_use, 1):
            url = chunk.get("url", "")
            title = chunk.get("title", "Untitled")
            namespace = chunk.get("namespace", "")
            text = chunk.get("text", "")
            breadcrumb = chunk.get("breadcrumb", [])
            section = chunk.get("section", "")

            # Build breadcrumb title for better navigation context
            breadcrumb_title = " > ".join(breadcrumb) if breadcrumb else title

            # Truncate text (configurable via CONTEXT_CHAR_LIMIT) for better coverage while managing context window
            text_excerpt = text[:CONFIG.CONTEXT_CHAR_LIMIT] if text else ""

            source = {
                "number": idx,
                "title": title,
                "breadcrumb": breadcrumb,
                "breadcrumb_title": breadcrumb_title,
                "url": url,
                "namespace": namespace,
                "section": section,
            }
            sources.append(source)

            # Format context with breadcrumb for better navigation
            context_lines.append(
                f"[{idx}] {breadcrumb_title}\n"
                f"URL: {url}\n\n"
                f"{text_excerpt}\n"
            )

        context = "\n---\n".join(context_lines)
        return context, sources

    @staticmethod
    def build_user_prompt(question: str, context: str, max_chunks: int = 4) -> str:
        """Build final user prompt with context limit enforcement.

        Args:
            question: The user's question
            context: Formatted context blocks (already limited by build_context_block)
            max_chunks: Context limit for instructions (for user awareness)

        Returns:
            Formatted user prompt
        """
        return f"""Based on the following {max_chunks} context blocks, answer the user's question.
Use inline citations [1], [2], etc., when referencing context.
If the answer is not in these docs, respond: "Not in docs" and suggest related topics.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, accurate answer with inline citations to the source numbers above."""

    @staticmethod
    def build_messages(
        question: str,
        chunks: List[Dict[str, Any]],
        namespace: str = "clockify",
        max_chunks: int = None,
        reasoning_effort: str = "low",
    ) -> tuple[List[Dict[str, str]], List[Dict], Optional[str]]:
        """Build messages for LLM + track sources + developer instructions for Harmony.

        Args:
            question: The user's question
            chunks: Retrieved chunks from search/retrieval
            namespace: Documentation namespace (clockify, langchain, etc.)
            max_chunks: Maximum context chunks to include (default from CONFIG.MAX_CONTEXT_CHUNKS)
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG latency optimization)

        Returns:
            Tuple of (messages_list, sources_list, developer_instructions)
            - messages_list: Standard OpenAI format messages (system + user)
            - sources_list: Metadata for each context chunk
            - developer_instructions: RAG enforcement rules for Harmony Developer role
        """
        # Use config default if not specified
        if max_chunks is None:
            max_chunks = CONFIG.MAX_CONTEXT_CHUNKS

        context, sources = RAGPrompt.build_context_block(chunks, max_chunks=max_chunks)
        system_msg = RAGPrompt.get_system_prompt(namespace)
        user_msg = RAGPrompt.build_user_prompt(question, context, max_chunks=max_chunks)
        developer_instructions = RAGPrompt.get_developer_instructions(namespace)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages, sources, developer_instructions

    @staticmethod
    def suggest_refinements(question: str, tried_terms: Optional[List[str]] = None) -> List[str]:
        """
        P2: Suggest related search terms when answer is not found in docs.

        Generates alternative queries based on the original question
        to help users refine their search.

        Args:
            question: Original question that returned "Not in docs"
            tried_terms: Terms already tried (to avoid repeating)

        Returns:
            List of suggested refinement questions
        """
        tried = set(tried_terms or [])
        suggestions = []

        # Extract key nouns/concepts from the question
        # Simple heuristic: look for common Clockify terms and related concepts
        question_lower = question.lower()

        # Define related term mappings for common Clockify queries
        refinement_map = {
            "timesheet": ["time entry", "time tracking", "work log", "track time"],
            "timer": ["kiosk mode", "stopwatch", "active tracking", "real-time clock"],
            "project": ["task", "assignment", "work item", "billable work"],
            "team": ["workspace", "members", "organization", "company"],
            "report": ["analytics", "time summary", "billing report", "time report"],
            "approval": ["timesheet approval", "review", "sign-off", "manager approval"],
            "integration": ["webhook", "API", "third-party", "connected app"],
            "permission": ["access", "role", "authorization", "admin"],
            "rate": ["billable rate", "cost", "pricing", "hourly rate"],
            "clock": ["check in", "start timer", "punch clock", "time clock"],
        }

        # Suggest alternatives based on detected keywords
        for keyword, alternatives in refinement_map.items():
            if keyword in question_lower:
                for alt in alternatives:
                    if alt not in tried:
                        # Create suggestion by replacing keyword with alternative
                        suggested = question.replace(keyword, alt, 1)
                        if suggested not in suggestions and suggested != question:
                            suggestions.append(suggested)
                            if len(suggestions) >= 3:  # Limit to 3 suggestions
                                return suggestions

        # Fallback: suggest broad topics if no keyword matches
        fallback_suggestions = [
            "How do I get started with Clockify?",
            "What are the main features of Clockify?",
            "How do I set up my account?",
        ]
        for fallback in fallback_suggestions:
            if fallback not in tried and fallback != question:
                suggestions.append(fallback)
                if len(suggestions) >= 3:
                    break

        return suggestions

    @staticmethod
    def format_response(answer: str, sources: List[Dict], use_citations: bool = True) -> Dict[str, Any]:
        """Format final response with sources list.

        Args:
            answer: The LLM-generated answer text
            sources: List of source metadata from build_context_block
            use_citations: Whether to include sources section

        Returns:
            Dictionary with formatted answer and sources metadata
        """
        response_text = answer

        if use_citations and sources:
            # Add Sources section with breadcrumb titles for navigation
            sources_section = "\n\n## Sources\n\n"
            for src in sources:
                sources_section += f"[{src['number']}] **{src['breadcrumb_title']}** ({src['namespace']})\n"
                sources_section += f"    URL: {src['url']}\n"
                if src.get("section"):
                    sources_section += f"    Section: {src['section']}\n"
                sources_section += "\n"

            response_text += sources_section

        return {
            "answer": response_text,
            "sources": sources,
            "sources_count": len(sources),
            "chunk_limit": 4,  # Enforced in build_context_block
        }
