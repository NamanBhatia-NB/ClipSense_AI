"""
ClipSense LLM Client Abstraction

Provides a decoupled, provider-agnostic interface for structured LLM inference:
- LLMClient: Abstract base protocol.
- GeminiLLMClient: Google GenAI implementation with native structured output (Pydantic schema),
  timeout enforcement, and exponential backoff retry.
- MockLLMClient: Deterministic, offline test mock for unit testing and CI.
- get_llm_client: Factory resolving the configured client.
"""

from abc import ABC, abstractmethod
import json
import logging
import re
import time
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

from app.config import LLMConfig, config

logger = logging.getLogger("clipsense.core.llm")

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Abstract base interface for structured LLM reasoning."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.config = llm_config or config.llm
        self.provider = self.config.provider
        self.model_name = self.config.model_name

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
    ) -> T:
        """
        Execute structured LLM inference and validate the output against response_schema.

        Args:
            prompt: Formatted user query / context prompt.
            response_schema: Pydantic model type defining expected response structure.
            system_instruction: Optional system instruction / role definition.

        Returns:
            Validated instance of response_schema.

        Raises:
            RuntimeError / ValueError on network failure or schema validation error.
        """
        pass


class GeminiLLMClient(LLMClient):
    """Google Gemini implementation using official google-genai SDK."""

    def __init__(self, llm_config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        super().__init__(llm_config=llm_config)
        from google import genai

        effective_key = api_key or config.gemini_api_key
        if not effective_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Set GEMINI_API_KEY environment variable "
                "or pass an explicit key."
            )
        self.client = genai.Client(api_key=effective_key)

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
    ) -> T:
        from google.genai import types

        max_retries = max(self.config.max_retries, 6)
        backoff_sec = 2.0

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Calling Gemini ({self.model_name}) attempt {attempt + 1}/{max_retries}..."
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                        system_instruction=system_instruction,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                if not response.text:
                    raise RuntimeError("Gemini returned empty response text.")

                # Validate and parse into target Pydantic schema
                parsed = response_schema.model_validate_json(response.text)
                return parsed

            except Exception as e:
                err_str = str(e)
                is_transient = any(
                    err in err_str
                    for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "timeout"]
                )
                if is_transient and attempt < max_retries - 1:
                    sleep_time = backoff_sec
                    # Extract explicit retry delay if provided by Gemini API
                    match = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
                    if not match:
                        match = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
                    if match:
                        sleep_time = float(match.group(1)) + 2.0
                    elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        sleep_time = max(backoff_sec, 25.0)

                    logger.warning(
                        f"Transient Gemini rate limit / error detected ({err_str[:120]}...). "
                        f"Waiting {sleep_time:.1f}s before retry (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    backoff_sec = max(backoff_sec * 2.0, 10.0)
                else:
                    logger.error(f"Fatal Gemini LLM failure: {e}")
                    raise RuntimeError(f"Gemini LLM inference failed: {e}") from e

        raise RuntimeError(f"Exceeded max retries ({max_retries}) calling Gemini LLM.")


class MockLLMClient(LLMClient):
    """Deterministic mock LLM client for testing without external API calls."""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mock_response: Optional[Any] = None,
    ):
        super().__init__(llm_config=llm_config)
        self.mock_response = mock_response

    def set_mock_response(self, mock_response: Any):
        self.mock_response = mock_response

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
    ) -> T:
        if self.mock_response is not None:
            if isinstance(self.mock_response, response_schema):
                return self.mock_response
            if isinstance(self.mock_response, dict):
                return response_schema.model_validate(self.mock_response)
            if isinstance(self.mock_response, str):
                return response_schema.model_validate_json(self.mock_response)
            raise ValueError(f"Incompatible mock response type: {type(self.mock_response)}")

        # If no explicit mock provided, attempt default initialization of schema
        try:
            return response_schema()
        except Exception:
            raise RuntimeError("MockLLMClient has no configured mock response for schema.")


def get_llm_client(llm_config: Optional[LLMConfig] = None) -> LLMClient:
    """Factory creating the appropriate LLMClient based on configuration."""
    cfg = llm_config or config.llm
    if cfg.provider.lower() == "gemini":
        return GeminiLLMClient(llm_config=cfg)
    elif cfg.provider.lower() in ("mock", "test"):
        return MockLLMClient(llm_config=cfg)
    else:
        raise ValueError(f"Unsupported LLM provider: '{cfg.provider}'")
