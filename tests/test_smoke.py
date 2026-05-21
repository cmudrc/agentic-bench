"""Adapter-free smoke tests. Verify the scoring math and that the
default YAML suite loads without errors. Anything that requires a
live LLM is intentionally not tested here -- run `make bench` for that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_bench import scoring
from agentic_bench.adapters.base import ToolCall
from agentic_bench.runner import load_suite

SUITE = Path(__file__).resolve().parent.parent / "agentic_bench" / "tasks" / "aircraft_design.yaml"


def test_suite_loads():
    suite = load_suite(SUITE)
    assert suite["name"] == "aircraft_design_v1"
    kinds = {it["kind"] for it in suite["items"]}
    assert kinds == {"numerical", "routing", "args", "planning"}
    assert len(suite["tools"]) >= 5


def test_numerical_perfect():
    s = scoring.score_numerical(expected=0.55, got_text="0.55")
    assert s.score == pytest.approx(1.0)


def test_numerical_within_tolerance():
    s = scoring.score_numerical(expected=0.55, got_text="The answer is 0.60.", tolerance_pct=15)
    assert s.score == pytest.approx(1.0)


def test_numerical_garbage():
    s = scoring.score_numerical(expected=0.55, got_text="I don't know.")
    assert s.score == 0.0


def test_routing_first_hit():
    calls = [ToolCall(name="su2_run_aero"), ToolCall(name="tigl_export_step")]
    s = scoring.score_tool_routing("su2_run_aero", calls)
    assert s.score == 1.0


def test_routing_late_hit():
    calls = [ToolCall(name="tigl_export_step"), ToolCall(name="su2_run_aero")]
    s = scoring.score_tool_routing("su2_run_aero", calls)
    assert s.score == 0.5


def test_routing_miss():
    calls = [ToolCall(name="nseg_run_segments")]
    s = scoring.score_tool_routing("su2_run_aero", calls)
    assert s.score == 0.0


def test_args_exact():
    calls = [ToolCall(name="su2_run_aero", arguments={"mach": 0.78, "aoa_deg": 2.5, "preset": "workstation"})]
    s = scoring.score_arg_extraction({"mach": 0.78, "aoa_deg": 2.5, "preset": "workstation"}, calls)
    assert s.score == pytest.approx(1.0)


def test_args_partial():
    calls = [ToolCall(name="su2_run_aero", arguments={"mach": 0.78})]
    s = scoring.score_arg_extraction({"mach": 0.78, "preset": "workstation"}, calls)
    assert 0.0 < s.score < 1.0


def test_planning_perfect():
    s = scoring.score_planning(["a", "b", "c"], ["a", "b", "c"])
    assert s.score == 1.0


def test_planning_one_off():
    s = scoring.score_planning(["a", "b", "c"], ["a", "b"])
    assert s.score == pytest.approx(2 / 3, rel=1e-3)


def test_aggregate_renormalises_weights():
    items = [
        scoring.ScoreItem("a", "numerical", 0.8, 0, 0),
        scoring.ScoreItem("b", "routing", 0.5, 0, 0),
    ]
    out = scoring.aggregate(items)
    assert pytest.approx(sum(out["weights_used"].values())) == 1.0
    assert 0.0 <= out["loss"] <= 1.0
