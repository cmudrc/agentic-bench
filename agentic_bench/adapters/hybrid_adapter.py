"""HybridAdapter: route text/tool tasks to one model, multimodal to another.

Motivation: on our 2026-05-21 aircraft-design suite Qwen 2.5 7B beat
Gemma 4 E4B by 55 % on aggregate loss on text/tool tasks. On the
multimodal sub-suite (post 2026-05-28 image-bug fix) both models
score 2/3 — but Qwen has no vision and just defaults to "acceptable"
on every image, so the tie is coincidence; only Gemma's verdict is
grounded in the actual picture. Neither model dominates everywhere,
so this adapter sends each task to the model that does its kind best.

API matches LLMAdapter exactly: `chat()` goes to the planner model,
`chat_with_image()` goes to the seeker model. Callers (the agentic-
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
