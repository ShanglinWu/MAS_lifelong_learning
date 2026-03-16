"""Vendored and adapted LLM controller utilities from A-MEM."""

from __future__ import annotations

import json
import os
import re
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


class BedrockController(BaseLLMController):
    def __init__(self, model: str = "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0"):
        self.model = model

    def _coerce_json_str(self, content: str) -> str:
        text = content.strip()
        if not text:
            return "{}"

        # Fast path: already valid JSON.
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Common path for markdown-wrapped responses.
        match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        # Last-resort: first JSON object substring.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        return "{}"

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        # Try strict JSON mode first (if provider/model supports it), then
        # gracefully fall back to plain text JSON.
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with a valid JSON object only."},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
                temperature=temperature,
            )
            content = response.choices[0].message.content or "{}"
            return self._coerce_json_str(content)
        except Exception:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with a valid JSON object only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content or "{}"
            return self._coerce_json_str(content)


class LLMController:
    """LLM wrapper used by the vendored A-MEM evolution pipeline."""

    def __init__(
        self,
        backend: Literal["openai", "ollama", "bedrock"] = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "bedrock":
            self.llm = BedrockController(model)
        else:
            raise ValueError("Backend must be one of: 'openai', 'ollama', 'bedrock'")

    def get_completion(
        self,
        prompt: str,
        response_format: dict,
        temperature: float = 0.7,
    ) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
