"""Adapter interface every backend has to implement.

We keep this deliberately tiny so a new provider is ~50 lines of code.
The runner only ever calls `chat()` (for text + tool routing) and
`chat_with_image()` (for multimodal). Streaming, batching, and embeddings
are explicitly out of scope -- benchmarks should be reproducible and
boring, not feature-rich.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model wants to make."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    """What every adapter returns from a chat turn.

    `text` is the assistant's natural-language reply (may be empty if
    the model only emitted tool calls). `tool_calls` is a possibly-empty
    list of ToolCall objects. `latency_s` is wall-clock latency for the
    turn, measured by the adapter so we can compare apples to apples
    across providers.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    raw: Any = None


class LLMAdapter(Protocol):
    """Minimal interface a backend has to implement.

    Adapters MUST be deterministic at temperature=0; the benchmark sets
    temperature=0 for every run so two runs of the same task on the same
    model should give the same score modulo floating-point noise.
    """

    def name(self) -> str: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult: ...

    def chat_with_image(
        self,
        messages: list[dict[str, Any]],
        image_path: str,
        temperature: float = 0.0,
    ) -> ChatResult: ...
