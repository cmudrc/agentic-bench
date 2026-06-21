"""HybridAdapter: route text/tool tasks to one model, multimodal to another.

This adapter mirrors the production setup in
``cmudrc/agent-mcp/hybrid_agent.py``. The default is **all-Gemma**:
``gemma4:e4b`` for both the planner and the multimodal seeker. Earlier
benchmarks (2026-05) used ``qwen2.5:7b`` as the planner; those numbers
are preserved in ``reports/`` for transparency but are no longer the
recommended default. Neither model dominated every category, so the
split exists to send each task to the model that does its kind best
(text/tool vs. image-grounded).

API matches LLMAdapter exactly: ``chat()`` goes to the planner model,
``chat_with_image()`` goes to the seeker model. Callers (the agentic-
bench runner) don't need to know.
"""

from __future__ import annotations

from typing import Any

from agentic_bench.adapters.base import ChatResult
from agentic_bench.adapters.ollama_adapter import OllamaAdapter


class HybridAdapter:
    """Compose two OllamaAdapter instances behind one LLMAdapter face."""

    def __init__(
        self,
        model: str,                          # planner/text model
        seeker_model: str = "gemma4:e4b",    # multimodal model
        host: str | None = None,
        num_ctx: int = 8192,
    ):
        self.planner_model = model
        self.seeker_model = seeker_model
        self._planner = OllamaAdapter(model=model, host=host, num_ctx=num_ctx)
        self._seeker = OllamaAdapter(model=seeker_model, host=host, num_ctx=num_ctx)

    def name(self) -> str:
        return f"hybrid:planner={self.planner_model}+seeker={self.seeker_model}"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        return self._planner.chat(messages, tools=tools, temperature=temperature)

    def chat_with_image(
        self,
        messages: list[dict[str, Any]],
        image_path: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        return self._seeker.chat_with_image(messages, image_path, temperature=temperature)
