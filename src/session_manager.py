#!/usr/bin/env python3
"""
Session Manager for multi-turn conversational RAG.

Manages chat sessions with conversation history and TTL-based cleanup.
Each session tracks a conversation across multiple turns, enabling context-aware responses.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


class SessionManager:
    """
    Manage chat sessions with conversation history.

    Each session represents a conversation with multiple turns.
    Sessions expire after TTL (default: 1 hour) of inactivity.
    """

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000):
        """
        Initialize session manager.

        Args:
            ttl_seconds: Time-to-live for inactive sessions (default: 1 hour)
            max_sessions: Maximum number of concurrent sessions (default: 1000)
        """
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        logger.info(f"SessionManager initialized: ttl={ttl_seconds}s, max_sessions={max_sessions}")

    def create_session(self, namespace: str = "clockify") -> str:
        """
        Create a new chat session.

        Args:
            namespace: Knowledge base namespace (default: "clockify")

        Returns:
            Session ID (UUID)
        """
        # Cleanup expired sessions before creating new one
        self.cleanup_expired()

        # Enforce max sessions limit
        if len(self._sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_id = min(
                self._sessions.keys(),
                key=lambda sid: self._sessions[sid]["last_accessed"]
            )
            logger.warning(f"Max sessions reached ({self.max_sessions}), removing oldest: {oldest_id}")
            del self._sessions[oldest_id]

        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        self._sessions[session_id] = {
            "session_id": session_id,
            "namespace": namespace,
            "created_at": now,
            "last_accessed": now,
            "conversation": [],
            "ttl_seconds": self.ttl_seconds
        }

        logger.info(f"Created session: {session_id} (namespace={namespace})")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.

        Updates last_accessed timestamp if session exists.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session data dict, or None if not found/expired
        """
        session = self._sessions.get(session_id)

        if session:
            # Check if expired
            now = datetime.now(timezone.utc)
            ttl = timedelta(seconds=session["ttl_seconds"])
            if now - session["last_accessed"] > ttl:
                logger.info(f"Session {session_id} expired, removing")
                del self._sessions[session_id]
                return None

            # Update access time
            session["last_accessed"] = now
            logger.debug(f"Retrieved session: {session_id} (turns={len(session['conversation'])})")

        return session

    def add_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """
        Add a new conversation turn to a session.

        Args:
            session_id: Session ID
            turn_data: Turn data containing:
                - user_question: User's question
                - llm_answer: LLM's answer
                - retrieved_urls: List of URLs used for context
                - keywords: Optional extracted keywords
                - intent: Optional detected intent

        Returns:
            True if added successfully, False if session not found
        """
        session = self.get_session(session_id)

        if not session:
            logger.warning(f"Cannot add turn: session {session_id} not found")
            return False

        # Add turn metadata
        turn_data["turn"] = len(session["conversation"]) + 1
        turn_data["timestamp"] = datetime.now(timezone.utc)

        session["conversation"].append(turn_data)

        logger.info(
            f"Added turn {turn_data['turn']} to session {session_id}: "
            f"q='{turn_data['user_question'][:50]}...'"
        )

        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            turns = len(self._sessions[session_id]["conversation"])
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id} ({turns} turns)")
            return True

        logger.warning(f"Cannot delete: session {session_id} not found")
        return False

    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())

    def cleanup_expired(self) -> int:
        """
        Remove expired sessions.

        Called automatically on create_session() and can be called periodically.

        Returns:
            Number of sessions removed
        """
        now = datetime.now(timezone.utc)
        expired = []

        for session_id, session in self._sessions.items():
            ttl = timedelta(seconds=session["ttl_seconds"])
            if now - session["last_accessed"] > ttl:
                expired.append(session_id)

        for session_id in expired:
            del self._sessions[session_id]

        if expired:
            logger.info(f"Cleanup: removed {len(expired)} expired sessions")

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.

        Returns:
            Stats dict with counts and metrics
        """
        total_turns = sum(len(s["conversation"]) for s in self._sessions.values())

        return {
            "total_sessions": len(self._sessions),
            "total_turns": total_turns,
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl_seconds,
            "avg_turns_per_session": total_turns / len(self._sessions) if self._sessions else 0
        }
