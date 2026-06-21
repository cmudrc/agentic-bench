"""Structured-output (ReAct) adapter for models without native tool calling.

Some strong open-weight models — notably the Gemma 3 family
(`gemma3:27b`, `gemma3:12b`) — do not expose Ollama's native
function-calling API and return HTTP 400 ("does not support tools")
when `tools=` is passed. They DO, however, support Ollama's
*structured-output* feature (`format=<json_schema>`), which forces the
model to emit a JSON object conforming to a schema.

This adapter routes tool tasks through that path: when the runner asks
for a tool call, we describe the tool catalogue in the prompt and force
the model to emit `{"tool": "<name>", "arguments": {...}}` via constrained
decoding, then parse that into a `ToolCall`. Text-only tasks (numerical,
planning) and multimodal image tasks fall back to a plain chat, exactly
like `OllamaAdapter`.

This is a real ReAct/structured-output harness — the same mechanism the
production `gemma_agent_v2.py` planner uses — not a stub.
"""

from __future__ import annotations

import json
import time
from typing import Any

import ollama

from agentic_bench.adapters.base import ChatResult, ToolCall


class OllamaReactAdapter:
    """Ollama adapter that emulates tool calling via structured output.

    Use for models that lack native function calling (e.g. gemma3:27b).
    For models WITH native tool calling, prefer `OllamaAdapter`.
    """

    def __init__(self, model: str, host: str | None = None, num_ctx: int = 8192):
        self.model = model
        self._client = ollama.Client(host=host) if host else ollama
        self._num_ctx = num_ctx

    def name(self) -> str:
        return f"ollama-react:{self.model}"

    @staticmethod
    def _catalogue(tools: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for t in tools:
            fn = t.get("function", t)
            name = fn["name"]
            desc = fn.get("description", "")
            params = (fn.get("parameters") or {}).get("properties", {})
            required = set((fn.get("parameters") or {}).get("required") or [])
            arg_lines = []
            for arg, spec in params.items():
                mark = "*" if arg in required else " "
                adesc = spec.get("description") or spec.get("type", "")
                arg_lines.append(f"      {mark} {arg}: {adesc}")
            block = f"  - {name}: {desc}"
            if arg_lines:
                block += "\n" + "\n".join(arg_lines)
            lines.append(block)
        return "\n".join(lines)

    @staticmethod
    def _turn_schema(tools: list[dict[str, Any]]) -> dict[str, Any]:
        names = [t.get("function", t)["name"] for t in tools]
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": names},
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        t0 = time.time()

        if not tools:
            resp = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_ctx": self._num_ctx},
                keep_alive="10m",
            )
            msg = resp["message"]
            return ChatResult(
                text=msg.get("content", "") or "",
                tool_calls=[],
                latency_s=time.time() - t0,
                raw=resp,
            )

        schema = self._turn_schema(tools)
        guidance = (
            "You must select exactly ONE tool and fill its arguments.\n"
            "Reply with a single JSON object: "
            '{"tool": "<one of the names below>", "arguments": {<key>: <value>}}.\n'
            "Fill arguments using the exact numeric/string values in the user's "
            "request. Use the documented argument names verbatim.\n\n"
            f"AVAILABLE TOOLS:\n{self._catalogue(tools)}"
        )
        patched = [{"role": "system", "content": guidance}] + list(messages)
        resp = self._client.chat(
            model=self.model,
            messages=patched,
            format=schema,
            options={"temperature": temperature, "num_ctx": self._num_ctx},
            keep_alive="10m",
        )
        raw_text = resp["message"].get("content", "") or ""
        tool_calls: list[ToolCall] = []
        try:
            obj = json.loads(raw_text)
            name = obj.get("tool", "")
            args = obj.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if name:
                tool_calls.append(ToolCall(name=name, arguments=args))
        except json.JSONDecodeError:
            pass
        return ChatResult(
            text=raw_text,
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
        if not messages:
            raise ValueError("messages must be non-empty for chat_with_image")
        last = dict(messages[-1])
        last.setdefault("images", []).append(image_path)
        patched = messages[:-1] + [last]
        return self.chat(patched, tools=None, temperature=temperature)
