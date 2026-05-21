"""Adapters expose third-party LLM APIs behind a single small interface.

Each adapter implements `LLMAdapter`. Add new providers (OpenAI,
Anthropic, vLLM, llama-cpp) by dropping a file in this package and
registering it in `agentic_bench.adapters.REGISTRY`.
"""

from agentic_bench.adapters.base import LLMAdapter, ToolCall, ChatResult
from agentic_bench.adapters.ollama_adapter import OllamaAdapter

REGISTRY: dict[str, type[LLMAdapter]] = {
    "ollama": OllamaAdapter,
}

__all__ = ["LLMAdapter", "OllamaAdapter", "ToolCall", "ChatResult", "REGISTRY"]
