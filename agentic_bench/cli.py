"""agentic-bench CLI.

Usage:
    agentic-bench run --suite path/to/suite.yaml --model gemma4:e4b
    agentic-bench run --suite path/to/suite.yaml --model qwen2.5:7b \\
                     --backend ollama --report reports/qwen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_bench.adapters import REGISTRY
from agentic_bench.runner import run_suite, save_report


def _build_adapter(backend: str, model: str, host: str | None, seeker_model: str | None):
    if backend not in REGISTRY:
        raise SystemExit(f"unknown backend: {backend}. known: {sorted(REGISTRY)}")
    cls = REGISTRY[backend]
    kwargs: dict[str, str] = {"model": model}
    if host:
        kwargs["host"] = host
    if backend == "hybrid" and seeker_model:
        kwargs["seeker_model"] = seeker_model
    return cls(**kwargs)


def cmd_run(args: argparse.Namespace) -> int:
    adapter = _build_adapter(args.backend, args.model, args.host, args.seeker_model)
    print(f"[agentic-bench] adapter = {adapter.name()}", file=sys.stderr)
    print(f"[agentic-bench] suite   = {args.suite}", file=sys.stderr)

    report = run_suite(adapter, args.suite)
    if args.report:
        save_report(report, args.report)
        print(f"[agentic-bench] report -> {args.report}", file=sys.stderr)

    print("\n=== AGENTIC-BENCH REPORT ===")
    print(f"  adapter  : {report['adapter']}")
    print(f"  suite    : {report['suite']}")
    print(f"  items    : {report['n_items']}")
    print(f"  wall (s) : {report['wall_time_s']}")
    print(f"  loss     : {report['aggregate']['loss']:.4f}")
    print("  per-category scores:")
    for cat, score in sorted(report["aggregate"]["per_category"].items()):
        print(f"    {cat:<11} {score:.3f}")
    print("\n  per-item:")
    for it in report["items"]:
        marker = "[OK]" if it["score"] >= 0.95 else "[--]" if it["score"] >= 0.5 else "[XX]"
        print(f"   {marker} {it['task_id']:<30} ({it['category']:<10}) score={it['score']:.3f}  note={it['note']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Pretty-print a saved report."""
    p = Path(args.path)
    with p.open() as f:
        report = json.load(f)
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a benchmark suite")
    run_p.add_argument("--suite", required=True, help="path to YAML suite")
    run_p.add_argument("--backend", default="ollama", help="adapter name (default: ollama)")
    run_p.add_argument("--model", required=True, help="model tag (e.g. gemma4:e4b)")
    run_p.add_argument("--host", default=None, help="adapter host (e.g. http://localhost:11434)")
    run_p.add_argument("--seeker-model", default="gemma4:e4b",
                       help="Multimodal model for the hybrid adapter (default: gemma4:e4b). Ignored by non-hybrid backends.")
    run_p.add_argument("--report", default=None, help="write JSON report to this path")
    run_p.set_defaults(fn=cmd_run)

    show_p = sub.add_parser("show", help="pretty-print a saved JSON report")
    show_p.add_argument("path", help="path to JSON report")
    show_p.set_defaults(fn=cmd_show)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
