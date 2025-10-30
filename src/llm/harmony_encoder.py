#!/usr/bin/env python3
"""Harmony Chat Format Support for gpt-oss:20b.

Implements proper Harmony encoding for optimal gpt-oss:20b performance.
Handles message encoding, stop tokens, and fallback to standard format.

References:
- https://github.com/openai/openai-harmony
- gpt-oss:20b expects Harmony format; standard OpenAI format causes degradation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Attempt to import Harmony support
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
        UserContent,
        AssistantContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning(
        "openai-harmony not installed. Install with: pip install openai-harmony. "
        "gpt-oss:20b performance will be degraded without Harmony format."
    )


class HarmonyEncoder:
    """Encode messages in Harmony format for gpt-oss:20b."""

    def __init__(self, use_harmony: bool = True):
        """Initialize Harmony encoder.

        Args:
            use_harmony: Whether to use Harmony format (requires openai-harmony)
        """
        self.use_harmony = use_harmony and HARMONY_AVAILABLE

        if self.use_harmony:
            try:
                self.encoding = load_harmony_encoding(
                    HarmonyEncodingName.HARMONY_GPT_OSS
                )
                logger.info("✓ Harmony encoder initialized for gpt-oss:20b")
            except Exception as e:
                logger.warning(f"Failed to initialize Harmony encoder: {e}. Falling back to standard format.")
                self.use_harmony = False
                self.encoding = None
        else:
            self.encoding = None
            if not HARMONY_AVAILABLE:
                logger.debug(
                    "Harmony format disabled: openai-harmony not installed. "
                    "Install for optimal gpt-oss:20b performance."
                )

    def render_messages(
        self,
        system_prompt: str,
        developer_instructions: Optional[str] = None,
        user_message: str = "",
        reasoning_effort: str = "low",
    ) -> Tuple[List[int], Optional[List[int]]]:
        """Render messages in Harmony format.

        Args:
            system_prompt: Base system prompt
            developer_instructions: RAG-specific instructions (moves to Developer role)
            user_message: User's question/prompt
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG)

        Returns:
            Tuple of (prefill_token_ids, stop_token_ids)
            If Harmony unavailable, returns ([], None) and caller should use standard format
        """
        if not self.use_harmony or self.encoding is None:
            return [], None

        try:
            # Build conversation with Harmony roles
            messages = []

            # System role: Base instructions
            if system_prompt:
                messages.append(
                    Message.from_role_and_content(
                        Role.SYSTEM,
                        SystemContent.new().with_content(system_prompt),
                    )
                )

            # Developer role: RAG-specific instructions (enforces grounding, citations)
            if developer_instructions:
                dev_content = DeveloperContent.new().with_instructions(
                    developer_instructions
                )
                # Add reasoning effort control
                if reasoning_effort == "low":
                    dev_content = dev_content.with_instructions(
                        "Use low reasoning effort to minimize latency."
                    )
                elif reasoning_effort == "high":
                    dev_content = dev_content.with_instructions(
                        "Use high reasoning effort for complex analysis."
                    )

                messages.append(
                    Message.from_role_and_content(Role.DEVELOPER, dev_content)
                )

            # User role: The actual question with context
            if user_message:
                messages.append(
                    Message.from_role_and_content(
                        Role.USER,
                        UserContent.from_content(user_message),
                    )
                )

            convo = Conversation(messages)

            # Render conversation for completion (generates prefill)
            prefill_ids = self.encoding.render_conversation_for_completion(
                convo, Role.ASSISTANT
            )

            # Get stop tokens for assistant actions (prevents leakage)
            stop_ids = self.encoding.stop_tokens_for_assistant_actions()

            logger.debug(
                f"Rendered Harmony messages: prefill_len={len(prefill_ids)}, "
                f"stop_tokens={len(stop_ids) if stop_ids else 0}"
            )

            return prefill_ids, stop_ids

        except Exception as e:
            logger.warning(
                f"Failed to render Harmony messages: {e}. Falling back to standard format."
            )
            return [], None

    def build_messages_standard(
        self,
        system_prompt: str,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """Build messages in standard OpenAI format (fallback).

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            List of message dicts with role and content
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_message:
            messages.append({"role": "user", "content": user_message})
        return messages


def get_harmony_encoder(model_name: str = "") -> HarmonyEncoder:
    """Get Harmony encoder instance.

    Args:
        model_name: Model name (used to determine if Harmony is applicable)

    Returns:
        HarmonyEncoder instance (with Harmony enabled for gpt-oss models)
    """
    # Enable Harmony for gpt-oss and oss models
    use_harmony = any(
        model_name.lower().startswith(prefix)
        for prefix in ["gpt-oss", "oss", "oss20b", "oss13b", "oss7b"]
    )

    # Check environment override
    harmony_env = os.getenv("LLM_USE_HARMONY", "auto").lower()
    if harmony_env == "true":
        use_harmony = True
    elif harmony_env == "false":
        use_harmony = False

    return HarmonyEncoder(use_harmony=use_harmony)
