from __future__ import annotations

from typing import Any, Mapping

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMFactory:
    @staticmethod
    def create(model: str | None, temperature: float | None, extras: Mapping[str, Any]) -> Any:
        if not model:
            raise ValueError("Agent config must define a model")

        parts = model.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid model format '{model}'. Expected 'provider/model-name' "
                f"(e.g. 'openai/gpt-4o' or 'ollama/llama3')."
            )
        provider, name = parts

        temp = temperature if temperature is not None else 0.0

        if provider == "ollama":
            return ChatOllama(
                model=name,
                temperature=temp,
                **extras,
            )

        if provider == "openai":
            return ChatOpenAI(
                model=name,
                temperature=temp,
                **extras,
            )

        raise ValueError(f"Unsupported provider: {provider}")
