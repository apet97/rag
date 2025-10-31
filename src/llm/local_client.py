#!/usr/bin/env python3
"""Local LLM client with mock mode and Harmony chat format support."""

import httpx
import json
import time
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.llm.harmony_encoder import get_harmony_encoder

logger = logging.getLogger(__name__)


# P5: Normalize model names to accept multiple formats (oss20b, gpt-oss:20b)
def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to support multiple naming conventions.

    Supports:
    - "oss20b" -> "oss20b" (native Ollama format)
    - "gpt-oss:20b" -> "oss20b" (GPT-style format alias)
    - Other formats -> returned as-is

    Args:
        model_name: Model name in any supported format

    Returns:
        Normalized model name
    """
    if not model_name:
        return "oss20b"

    model_name = str(model_name).strip()

    # Map aliases to canonical names
    model_aliases = {
        "gpt-oss:20b": "oss20b",
        "gpt-oss:13b": "oss13b",
        "gpt-oss:7b": "oss7b",
        "gpt-4": "gpt-4",  # Support GPT-4 if using OpenAI
    }

    return model_aliases.get(model_name, model_name)

class LocalLLMClient:
    """Client for Ollama or OpenAI-compatible LLM endpoint with optional mock mode."""

    # Declare attributes for type checkers
    base_url: str
    model_name: str
    timeout: int
    max_retries: int
    api_type: str
    endpoint: str
    mock_mode: bool

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "oss20b",
        timeout: int = 60,
        max_retries: int = 3,
        mock_mode: Optional[bool] = None,
        api_type: str = "ollama",  # "ollama" or "openai"
    ):
        """Initialize LLM client.

        Args:
            base_url: LLM server endpoint (uses env var if None)
            model_name: Model name to use
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            mock_mode: Mock mode flag (None = auto-detect)
            api_type: "ollama" or "openai" - endpoint format
        """
        # Use environment variable if base_url not provided
        if base_url is None:
            base_url = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")

        self.base_url = base_url
        # P5: Normalize model name to support multiple formats
        self.model_name = normalize_model_name(model_name)
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_type = api_type

        # Harmony chat format support for gpt-oss:20b optimal performance
        self.harmony_encoder = get_harmony_encoder(self.model_name)

        # Set endpoint based on API type
        if api_type == "ollama":
            # Ollama API: /api/chat
            self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        else:
            # OpenAI-compatible: /v1/chat/completions
            self.endpoint = f"{base_url.rstrip('/')}/chat/completions"

        # Determine mock mode
        if mock_mode is None:
            # Auto-detect: check if LLM is running
            self.mock_mode = not self._check_connection_silent()
        else:
            self.mock_mode = mock_mode

        mode_str = "MOCK_MODE" if self.mock_mode else "PRODUCTION_MODE"
        api_str = f"({api_type})"
        harmony_str = " [Harmony enabled]" if self.harmony_encoder.use_harmony else ""
        logger.info(f"🚀 LLM Client initialized in {mode_str} {api_str}{harmony_str}")
        logger.info(f"   Endpoint: {self.endpoint}")
        logger.info(f"   Model: {self.model_name}")

    def _check_connection_silent(self) -> bool:
        """Silently check if LLM is available."""
        try:
            response = httpx.post(
                self.endpoint,
                json={"model": self.model_name, "messages": [{"role": "user", "content": "OK"}], "max_tokens": 5},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def test_connection(self) -> bool:
        """Test connection to LLM server (or mock mode).

        Returns:
            True if connected/mocked, False otherwise
        """
        if self.mock_mode:
            logger.info("✅ Mock LLM ready")
            return True

        try:
            logger.info(f"Testing connection to {self.base_url}")
            response = httpx.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "Say 'OK'"}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                logger.info("✅ LLM connection successful")
                return True
            else:
                logger.error(f"❌ LLM returned status {response.status_code}")
                return False
        except httpx.ConnectError:
            logger.error(f"❌ Cannot connect to {self.base_url}")
            logger.error("   Make sure LLM is running (e.g., ollama serve)")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
            return False

    def _generate_mock_response(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_context: str = "",
    ) -> str:
        """Generate a mock response for testing.

        Args:
            system_prompt: System prompt
            user_prompt: User question
            retrieved_context: Context from retrieval

        Returns:
            Mock response text
        """
        # Template-based mock responses
        if "create" in user_prompt.lower() and "project" in user_prompt.lower():
            return """Based on the Clockify documentation:

To create a project in Clockify:
1. Log in to your Clockify account
2. Click on the "Projects" tab
3. Click the "Create new project" button
4. Enter a project name
5. (Optional) Add a client name
6. (Optional) Set billable rates and budget
7. Click "Save" to create the project

Once created, you can assign team members to the project and start tracking time against it.

[Source: Clockify Project Management Guide]"""

        elif "report" in user_prompt.lower() and "timesheet" in user_prompt.lower():
            return """Based on the Clockify documentation:

To generate a timesheet report in Clockify:
1. Navigate to the "Reports" section
2. Select "Timesheet" from the report type dropdown
3. Choose the date range you want to report on
4. Select team members or leave blank for all
5. Click "Generate Report"
6. Export to Excel or PDF using the export buttons

The timesheet report shows total hours worked, breaks, billable time, and project allocation for the selected period.

[Source: Clockify Reporting Guide]"""

        elif "integration" in user_prompt.lower():
            return """Based on the Clockify documentation:

Clockify integrates with many popular tools:
- **Project Management:** Jira, Asana, Monday.com, ClickUp
- **Communication:** Slack, Microsoft Teams
- **Calendar:** Google Calendar, Outlook
- **Development:** GitHub, GitLab, Bitbucket
- **Productivity:** Google Sheets, Zapier

You can find and configure integrations in your account settings under "Integrations". Each integration can be customized with authentication tokens and mapping preferences.

[Source: Clockify Integrations Documentation]"""

        else:
            # Generic response template
            return f"""Based on the Clockify documentation:

{user_prompt.strip().rstrip('?')}:

Clockify provides a comprehensive time tracking solution. The feature you're asking about is available through our web interface, desktop app, and mobile applications. For detailed instructions, please refer to our help documentation or contact our support team.

Configuration and customization options are available in your account settings.

[Source: Clockify Support Documentation]"""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        top_p: float = 0.9,
        retrieved_context: str = "",
        developer_instructions: Optional[str] = None,
        reasoning_effort: str = "low",
    ) -> Optional[str]:
        """Generate response from LLM or mock, with Harmony format support.

        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            retrieved_context: Context from retrieval (for mock)
            developer_instructions: RAG-specific instructions (Harmony Developer role)
            reasoning_effort: "low", "medium", or "high" (default: "low" for RAG)

        Returns:
            Generated text or None if failed
        """
        if self.mock_mode:
            logger.debug("📝 Generating mock response")
            return self._generate_mock_response(system_prompt, user_prompt, retrieved_context)

        # Try Harmony format first if available
        prefill_ids, stop_ids = self.harmony_encoder.render_messages(
            system_prompt=system_prompt,
            developer_instructions=developer_instructions,
            user_message=user_prompt,
            reasoning_effort=reasoning_effort,
        )

        # Use Harmony prefill if available, otherwise fall back to standard messages
        if prefill_ids:
            # For Harmony prefill: pass token IDs directly (Ollama/vLLM compatible)
            # This is experimental; may need adjustment based on server support
            messages = None  # Harmony uses prefill_ids instead
            logger.debug("Using Harmony chat format")
        else:
            # Standard OpenAI format fallback
            messages = self.harmony_encoder.encoding.build_messages_standard(
                system_prompt=system_prompt,
                user_message=user_prompt,
            ) if hasattr(self.harmony_encoder.encoding, 'build_messages_standard') else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            logger.debug("Using standard chat format")

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling LLM (attempt {attempt + 1}/{self.max_retries})")

                # Build request body based on API type
                if self.api_type == "ollama":
                    # Ollama API format
                    payload = {
                        "model": self.model_name,
                        "stream": False,
                        "temperature": temperature,
                    }
                    if messages is not None:
                        payload["messages"] = messages
                    elif prefill_ids:
                        # For Harmony: pass prefill tokens if supported
                        # Fallback: use standard messages if server doesn't support prefill
                        payload["messages"] = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                    # Note: Ollama might not support max_tokens, but we add it
                else:
                    # OpenAI-compatible format
                    payload = {
                        "model": self.model_name,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stream": False,
                    }
                    if messages is not None:
                        payload["messages"] = messages
                    # Add Harmony stop tokens if available (Harmony-aware servers will respect them)
                    if stop_ids:
                        payload["stop"] = stop_ids

                response = httpx.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                    verify=False,  # For self-signed certs from company
                )

                if response.status_code == 200:
                    data = response.json()

                    # Parse response based on API type
                    if self.api_type == "ollama":
                        # Ollama response: {"message": {"role": "assistant", "content": "..."}}
                        if "message" in data:
                            answer = data["message"].get("content", "")
                        elif "choices" in data:
                            # Some Ollama versions use choices format
                            answer = data["choices"][0].get("message", {}).get("content", "")
                        else:
                            logger.warning("Unexpected Ollama response format")
                            return None
                    else:
                        # OpenAI-compatible response: {"choices": [{"message": {"content": "..."}}]}
                        if "choices" in data and len(data["choices"]) > 0:
                            answer = data["choices"][0].get("message", {}).get("content", "")
                        else:
                            logger.warning("Unexpected response format from LLM")
                            return None

                    logger.debug(f"LLM response: {answer[:100]}...")
                    return answer
                else:
                    logger.warning(f"LLM returned status {response.status_code}")
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    continue

            except httpx.TimeoutException:
                logger.warning(f"LLM request timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            except httpx.ConnectError:
                logger.error("Cannot connect to LLM server")
                return None
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue

        logger.error(f"Failed to get LLM response after {self.max_retries} attempts")
        return None

    def generate_streaming(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """Generate response from LLM with streaming.

        Yields:
            Chunks of generated text
        """
        if self.mock_mode:
            # For mock mode, yield the response in chunks
            mock_text = self._generate_mock_response(system_prompt, user_prompt)
            # Yield in ~50 character chunks to simulate streaming
            for i in range(0, len(mock_text), 50):
                yield mock_text[i:i + 50]
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            with httpx.stream(
                "POST",
                self.endpoint,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": True,
                },
                timeout=self.timeout,
            ) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str and data_str != "[DONE]":
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    continue
                else:
                    logger.error(f"LLM streaming returned status {response.status_code}")
                    yield f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"LLM streaming failed: {str(e)}")
            yield f"Error: {str(e)}"


# Singleton instance
_client: Optional[LocalLLMClient] = None


def get_llm_client(
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    mock_mode: Optional[bool] = None,
    api_type: Optional[str] = None,
) -> LocalLLMClient:
    """Get or create singleton LLM client.

    Args:
        base_url: LLM server endpoint (uses env var if None)
        model_name: Model name to use (uses env var if None)
        mock_mode: Mock mode flag (uses env var if None)
        api_type: "ollama" or "openai" (uses env var if None)

    Returns:
        LocalLLMClient instance
    """
    global _client
    if _client is None:
        # Use environment variables as defaults
        if base_url is None:
            base_url = os.getenv("LLM_ENDPOINT", "http://localhost:8080/v1")
        if model_name is None:
            model_name = os.getenv("LLM_MODEL", "oss20b")
        if api_type is None:
            api_type = os.getenv("LLM_API_TYPE", "ollama")
        if mock_mode is None:
            mock_env = os.getenv("MOCK_LLM", "auto").lower()
            if mock_env == "auto":
                mock_mode = None  # Auto-detect
            else:
                mock_mode = mock_env in ("true", "1", "yes")

        _client = LocalLLMClient(
            base_url=base_url,
            model_name=model_name,
            mock_mode=mock_mode,
            api_type=api_type,
        )
    return _client


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Test in mock mode
    print("\n" + "=" * 80)
    print("Testing Mock LLM Client")
    print("=" * 80 + "\n")

    client = LocalLLMClient(mock_mode=True)

    if client.test_connection():
        print("✅ Mock LLM client ready for use\n")
    else:
        print("❌ Mock LLM client failed\n")
        exit(1)
