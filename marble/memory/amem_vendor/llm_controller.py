"""Vendored and adapted LLM controller utilities from A-MEM."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Literal, Optional

from litellm import completion


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        """Get completion from an LLM backend."""


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI package not found. Install dependencies with `poetry install --with amem`."
            ) from exc

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return content or "{}"


class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama3"):
        self.model = model

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        response = completion(
            model=f"ollama_chat/{self.model}",
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or "{}"


class LLMController:
    """LLM wrapper used by the vendored A-MEM evolution pipeline."""

    def __init__(
        self,
        backend: Literal["openai", "ollama"] = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama'")

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
