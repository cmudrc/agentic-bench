"""Benchmark runner: load a YAML suite, dispatch each item, accumulate scores."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from agentic_bench import scoring
from agentic_bench.adapters.base import LLMAdapter


def load_suite(path: str | Path) -> dict[str, Any]:
    """Load a YAML benchmark suite. See examples/aircraft_design.yaml."""
    p = Path(path)
    with p.open() as f:
        return yaml.safe_load(f)


def _run_numerical(adapter: LLMAdapter, item: dict) -> scoring.ScoreItem:
    prompt = item["prompt"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an aerospace-engineering knowledge assistant. "
                "Answer with a single decimal number followed by SI or "
                "English units. Do not write any other commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    r = adapter.chat(messages, tools=None, temperature=0.0)
    s = scoring.score_numerical(
        expected=float(item["expected"]),
        got_text=r.text,
        tolerance_pct=float(item.get("tolerance_pct", 10.0)),
    )
    return scoring.ScoreItem(
        item["id"], "numerical", s.score, s.expected, r.text[:120], s.note
    )


def _run_routing(
    adapter: LLMAdapter, item: dict, tool_specs: list[dict]
) -> scoring.ScoreItem:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an aerospace-analysis agent. The user describes "
                "an analysis they want done. You MUST respond by calling "
                "exactly one tool from the provided set. Do not produce "
                "free-form text."
            ),
        },
        {"role": "user", "content": item["prompt"]},
    ]
    r = adapter.chat(messages, tools=tool_specs, temperature=0.0)
    s = scoring.score_tool_routing(item["expected_tool"], r.tool_calls)
    got = [tc.name for tc in r.tool_calls] or "(none)"
    return scoring.ScoreItem(item["id"], "routing", s.score, s.expected, got, s.note)


def _run_args(
    adapter: LLMAdapter, item: dict, tool_specs: list[dict]
) -> scoring.ScoreItem:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an aerospace-analysis agent. Pick the right "
                "tool AND fill in the numeric/string arguments from the "
                "user's request. Be precise."
            ),
        },
        {"role": "user", "content": item["prompt"]},
    ]
    r = adapter.chat(messages, tools=tool_specs, temperature=0.0)
    s = scoring.score_arg_extraction(
        item["expected_args"],
        r.tool_calls,
        arg_tolerance_pct=float(item.get("arg_tolerance_pct", 5.0)),
    )
    return scoring.ScoreItem(item["id"], "args", s.score, s.expected, s.got, s.note)


def _run_planning(
    adapter: LLMAdapter, item: dict, tool_specs: list[dict]
) -> scoring.ScoreItem:
    messages = [
        {
            "role": "system",
            "content": (
                "You are planning a multi-step aerospace analysis. "
                'Respond with a JSON object {"plan": ["tool1", '
                '"tool2", ...]} listing the tools you would call in '
                "order. Use only tool names from the catalog. No prose."
            ),
        },
        {
            "role": "user",
            "content": f"Tools: {[t['function']['name'] for t in tool_specs]}\nRequest: {item['prompt']}",
        },
    ]
    r = adapter.chat(messages, tools=None, temperature=0.0)
    got_seq: list[str] = []
    try:
        text = r.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        obj = json.loads(text)
        got_seq = [str(x) for x in obj.get("plan", [])]
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: pull tool-name-shaped tokens out of the text.
        names = {t["function"]["name"] for t in tool_specs}
        for tok in r.text.replace(",", " ").split():
            tok = tok.strip("\"'[]() \t\n")
            if tok in names:
                got_seq.append(tok)
    s = scoring.score_planning(item["expected_plan"], got_seq)
    return scoring.ScoreItem(item["id"], "planning", s.score, s.expected, got_seq)


def _run_multimodal(
    adapter: LLMAdapter, item: dict, suite_dir: Path
) -> scoring.ScoreItem:
    img_path = (suite_dir / item["image"]).resolve()
    if not img_path.exists():
        return scoring.ScoreItem(
            item["id"],
            "multimodal",
            0.0,
            item["expected_label"],
            f"image missing: {img_path}",
        )
    messages = [
        {
            "role": "user",
            "content": item["prompt"],
        }
    ]
    r = adapter.chat_with_image(messages, str(img_path), temperature=0.0)
    s = scoring.score_multimodal(item["expected_label"], r.text)
    return scoring.ScoreItem(
        item["id"], "multimodal", s.score, s.expected, r.text[:120], s.note
    )


def run_suite(adapter: LLMAdapter, suite_path: str | Path) -> dict[str, Any]:
    """Run every item in a suite, return a structured report."""
    suite_path = Path(suite_path)
    suite = load_suite(suite_path)
    suite_dir = suite_path.parent
    tool_specs = suite.get("tools", [])
    items = suite.get("items", [])

    results: list[scoring.ScoreItem] = []
    t_total0 = time.time()
    for it in items:
        kind = it["kind"]
        try:
            if kind == "numerical":
                results.append(_run_numerical(adapter, it))
            elif kind == "routing":
                results.append(_run_routing(adapter, it, tool_specs))
            elif kind == "args":
                results.append(_run_args(adapter, it, tool_specs))
            elif kind == "planning":
                results.append(_run_planning(adapter, it, tool_specs))
            elif kind == "multimodal":
                results.append(_run_multimodal(adapter, it, suite_dir))
            else:
                results.append(
                    scoring.ScoreItem(
                        it["id"], kind, 0.0, None, None, f"unknown kind: {kind}"
                    )
                )
        except Exception as e:
            results.append(
                scoring.ScoreItem(
                    it["id"], kind, 0.0, None, None, f"error: {type(e).__name__}: {e}"
                )
            )
    t_total = time.time() - t_total0

    agg = scoring.aggregate(results, weights=suite.get("weights"))
    return {
        "adapter": adapter.name(),
        "suite": suite.get("name", suite_path.stem),
        "n_items": len(results),
        "wall_time_s": round(t_total, 2),
        "aggregate": agg,
        "items": [asdict(r) for r in results],
    }


def save_report(report: dict, path: str | Path) -> None:
    """Write a report as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(report, f, indent=2, default=str)
