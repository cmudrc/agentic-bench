"""Per-task scoring + aggregate loss for agentic-bench.

Each task type has its own deterministic scorer. All scorers return a
float in [0, 1] where 1 == perfect, 0 == terrible. The aggregate loss
combines them as:

    L = w_num * (1 - num_score)
      + w_route * (1 - route_score)
      + w_arg * (1 - arg_score)
      + w_plan * (1 - plan_score)
      + w_mm * (1 - mm_score)

Default weights weight all five categories equally, but the YAML suite
can override.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

DEFAULT_WEIGHTS = {
    "numerical": 0.25,
    "routing": 0.25,
    "args": 0.20,
    "planning": 0.20,
    "multimodal": 0.10,
}


_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ScoreItem:
    task_id: str
    category: str
    score: float
    expected: object
    got: object
    note: str = ""


def extract_number(text: str) -> float | None:
    """Pull the *first* signed decimal number out of free-form text.

    We deliberately use the first match, not the largest, because models
    that start with "The answer is X" should be rewarded; models that
    open with a long preamble of conversational filler containing
    spurious numbers are penalised.
    """
    if not text:
        return None
    m = _NUMBER_RE.search(text)
    return float(m.group(0)) if m else None


def score_numerical(
    expected: float,
    got_text: str,
    tolerance_pct: float = 10.0,
) -> ScoreItem:
    """Score a numeric-answer task.

    We use a smooth, bounded measure rather than hard pass/fail: the
    score is 1.0 when the answer is within `tolerance_pct` of expected,
    decays exponentially as the relative error grows, and is 0.0 when
    the model refuses to emit a number.
    """
    got = extract_number(got_text)
    if got is None:
        return ScoreItem("", "numerical", 0.0, expected, got_text, "no number")
    if expected == 0.0:
        # Avoid div-by-zero; treat as absolute tolerance of tolerance_pct/100.
        err = abs(got - expected) / max(abs(got), 1e-9)
    else:
        err = abs(got - expected) / abs(expected)
    if err <= tolerance_pct / 100.0:
        return ScoreItem("", "numerical", 1.0, expected, got)
    # exp decay: 50% mass within 2x tolerance, ~0 by 5x tolerance
    decay = math.exp(-(err - tolerance_pct / 100.0) * 5.0)
    return ScoreItem("", "numerical", float(max(0.0, decay)), expected, got)


def score_tool_routing(
    expected_tool: str,
    got_tool_calls: list[object],
) -> ScoreItem:
    """Score a tool-routing task.

    Strict match on the *first* tool call; partial credit if the right
    tool appears later in the same turn. 0.0 if the model produced no
    tool calls at all.
    """
    names = [tc.name for tc in got_tool_calls]
    if not names:
        return ScoreItem("", "routing", 0.0, expected_tool, "(no tool calls)")
    if names[0] == expected_tool:
        return ScoreItem("", "routing", 1.0, expected_tool, names[0])
    if expected_tool in names:
        return ScoreItem("", "routing", 0.5, expected_tool, names, note="late hit")
    return ScoreItem("", "routing", 0.0, expected_tool, names)


def score_arg_extraction(
    expected_args: dict,
    got_tool_calls: list[object],
    arg_tolerance_pct: float = 5.0,
) -> ScoreItem:
    """Score how well the model fills numeric/string args.

    For numeric args we use the same exp-decay rule as score_numerical.
    For string/enum args we use exact match. Final score is the mean
    over expected keys.
    """
    if not got_tool_calls:
        return ScoreItem("", "args", 0.0, expected_args, None, "no tool call")
    got_args = got_tool_calls[0].arguments
    sub_scores: list[float] = []
    breakdown: dict[str, tuple] = {}
    for k, exp_v in expected_args.items():
        if k not in got_args:
            sub_scores.append(0.0)
            breakdown[k] = (exp_v, None)
            continue
        got_v = got_args[k]
        if isinstance(exp_v, (int, float)) and not isinstance(exp_v, bool):
            try:
                got_f = float(got_v)
            except (TypeError, ValueError):
                sub_scores.append(0.0)
                breakdown[k] = (exp_v, got_v)
                continue
            if exp_v == 0.0:
                err = abs(got_f - exp_v)
            else:
                err = abs(got_f - exp_v) / abs(exp_v)
            if err <= arg_tolerance_pct / 100.0:
                sub_scores.append(1.0)
            else:
                sub_scores.append(
                    float(max(0.0, math.exp(-(err - arg_tolerance_pct / 100.0) * 5.0)))
                )
            breakdown[k] = (exp_v, got_f)
        else:
            sub_scores.append(1.0 if str(got_v) == str(exp_v) else 0.0)
            breakdown[k] = (exp_v, got_v)
    score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
    return ScoreItem("", "args", score, expected_args, breakdown)


def score_planning(
    expected_sequence: list[str],
    got_sequence: list[str],
) -> ScoreItem:
    """Score a planning task by normalised edit distance over tool sequences.

    Returns 1 - (Levenshtein(expected, got) / max(len_expected, len_got)).
    Empty plans get 0.0; perfect matches get 1.0.
    """
    if not got_sequence:
        return ScoreItem("", "planning", 0.0, expected_sequence, got_sequence)
    n, m = len(expected_sequence), len(got_sequence)
    if n == 0 and m == 0:
        return ScoreItem("", "planning", 1.0, expected_sequence, got_sequence)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if expected_sequence[i - 1] == got_sequence[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    dist = dp[n][m]
    score = 1.0 - dist / max(n, m)
    return ScoreItem(
        "", "planning", float(max(0.0, score)), expected_sequence, got_sequence
    )


def score_multimodal(expected_label: str, got_text: str) -> ScoreItem:
    """Score a multimodal verdict by case-insensitive substring match."""
    if not got_text:
        return ScoreItem("", "multimodal", 0.0, expected_label, "")
    hit = expected_label.lower() in got_text.lower()
    return ScoreItem(
        "", "multimodal", 1.0 if hit else 0.0, expected_label, got_text[:120]
    )


def aggregate(
    scores: list[ScoreItem],
    weights: dict[str, float] | None = None,
) -> dict:
    """Return per-category mean, weighted aggregate loss, and item count.

    Loss = sum_c w_c * (1 - mean_score_c) over categories that actually
    appeared in `scores`. Weights are renormalised to the present
    categories so a suite that omits e.g. multimodal still produces a
    well-defined loss.
    """
    weights = weights or DEFAULT_WEIGHTS
    by_cat: dict[str, list[float]] = {}
    for s in scores:
        by_cat.setdefault(s.category, []).append(s.score)
    per_cat = {c: sum(v) / len(v) for c, v in by_cat.items()}
    active_weights = {c: weights.get(c, 0.0) for c in per_cat}
    wsum = sum(active_weights.values()) or 1.0
    active_weights = {c: w / wsum for c, w in active_weights.items()}
    loss = sum(active_weights[c] * (1.0 - per_cat[c]) for c in per_cat)
    return {
        "per_category": per_cat,
        "weights_used": active_weights,
        "loss": loss,
        "n_items": len(scores),
    }
