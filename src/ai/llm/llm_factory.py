from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

class LLMFactory:
    @staticmethod
    def create(model: str | None, temperature: float | None) -> Any:
        if not model:
            raise ValueError("Agent config must define a model")

        provider, name = model.split("/", 1)

        if provider == "ollama":
            return ChatOllama(
                model=name,
                temperature=temperature or 0,
            )

        if provider == "openai":
            return ChatOpenAI(
                model=name,
                temperature=temperature or 0,
            )

        raise ValueError(f"Unsupported provider: {provider}")
