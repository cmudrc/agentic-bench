"""Ollama-backed LLMAdapter. Default and reference implementation."""

from __future__ import annotations

import time
from typing import Any

import ollama

from agentic_bench.adapters.base import ChatResult, LLMAdapter, ToolCall


class OllamaAdapter:
    """Thin shim over the ollama Python client.

    Implements `LLMAdapter`. Pass the model tag (e.g. `gemma4:e4b`,
    `qwen2.5:7b`) at construction time. We always set `keep_alive` to a
    long value so the model stays loaded across benchmark items --
    otherwise the warm-up cost dominates short tasks and you measure
    disk I/O instead of model quality.
    """

    def __init__(self, model: str, host: str | None = None, num_ctx: int = 8192):
        self.model = model
        self._client = ollama.Client(host=host) if host else ollama
        self._num_ctx = num_ctx

    def name(self) -> str:
        return f"ollama:{self.model}"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        t0 = time.time()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": self._num_ctx},
            "keep_alive": "10m",
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat(**kwargs)
        msg = resp["message"]
        tool_calls = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
            name = fn["name"] if isinstance(fn, dict) else fn.name
            args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=name, arguments=args or {}))
        return ChatResult(
            text=msg.get("content", "") or "",
            tool_calls=tool_calls,
            latency_s=time.time() - t0,
            raw=resp,
        )

    def chat_with_image(
        self,
        messages: list[dict[str, Any]],
        image_path: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        # Ollama attaches images per-message via the `images` key.
        if not messages:
            raise ValueError("messages must be non-empty for chat_with_image")
        last = dict(messages[-1])
        last.setdefault("images", []).append(image_path)
        patched = messages[:-1] + [last]
        return self.chat(patched, tools=None, temperature=temperature)
