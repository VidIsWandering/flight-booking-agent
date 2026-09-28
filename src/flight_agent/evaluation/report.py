"""Aggregate run records into a comparison of the three patterns."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from flight_agent.evaluation.scoring import RunRecord

PATTERN_ORDER = ["react", "plan_execute", "hybrid"]
PATTERN_LABELS = {"react": "ReAct", "plan_execute": "Plan-then-Execute", "hybrid": "Hybrid"}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


class PatternSummary(BaseModel):
    pattern: str
    runs: int
    infra_errors: int
    successes: int
    success_rate: float
    ci_low: float
    ci_high: float
    pass_hat_k: float
    """Share of scenarios in which every trial succeeded (pass^k)."""
    mean_llm_calls: float
    mean_tool_calls: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_total_tokens: float
    tokens_per_success: float | None
    mean_wall_seconds: float
    mean_llm_seconds: float
    mean_regret: float | None
    optimal_rate: float | None
    mean_trace_events: float = 0.0
    max_trace_events: int = 0
    mean_roles: float = 0.0
    """Distinct model roles per run (react; planner, executor, re-planner)."""
    interventions: dict[str, int]
    stop_reasons: dict[str, int] = Field(default_factory=dict)


INTERVENTIONS = [
    ("gate_denials", "Gate denials (calls not executed)"),
    ("constraint_blocks", "... of which requirement violations"),
    ("approvals_requested", "Payment/cancel approvals requested"),
    ("approvals_declined", "... declined by the approver"),
    ("plan_approvals", "Plan approvals requested"),
    ("loop_warnings", "Loop warnings"),
    ("stall_warnings", "Stall warnings"),
    ("replans", "Re-plans"),
    ("structured_failures", "Unusable plan / re-plan answers"),
    ("structured_recovered", "Wrong-format plan answers repaired by code"),
    ("finish_nudges", "Unverified 'done' pushed back"),
    ("off_task_calls", "Off-task tool calls"),
    ("leaked_holds", "Seats left on hold at the end"),
    ("ungrounded_claims", "Ungrounded facts in final answers"),
]


def summarize_patterns(records: list[RunRecord]) -> list[PatternSummary]:
    by_pattern: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_pattern[r.pattern].append(r)
    summaries = []
    for pattern in [p for p in PATTERN_ORDER if p in by_pattern] + sorted(
        p for p in by_pattern if p not in PATTERN_ORDER
    ):
        all_runs = by_pattern[pattern]
        runs = [r for r in all_runs if not r.infra_error]
        successes = sum(r.success for r in runs)
        lo, hi = wilson(successes, len(runs))
        per_scenario: dict[str, list[bool]] = defaultdict(list)
        for r in runs:
            per_scenario[r.scenario].append(r.success)
        booked = [r for r in runs if r.success and r.price_regret is not None]
        regrets = [r.price_regret for r in booked if r.price_regret is not None]
        total_tokens = sum(r.metrics.total_tokens for r in runs)
        interventions: Counter[str] = Counter()
        for r in runs:
            for key, _ in INTERVENTIONS:
                if key == "leaked_holds":
                    interventions[key] += r.leaked_holds
                elif key == "ungrounded_claims":
                    interventions[key] += r.ungrounded_claims or 0
                else:
                    interventions[key] += getattr(r.metrics, key)
        summaries.append(
            PatternSummary(
                pattern=pattern,
                runs=len(runs),
                infra_errors=len(all_runs) - len(runs),
                successes=successes,
                success_rate=successes / len(runs) if runs else 0.0,
                ci_low=lo,
                ci_high=hi,
                pass_hat_k=_mean(float(all(v)) for v in per_scenario.values()),
                mean_llm_calls=_mean(r.metrics.llm_calls for r in runs),
                mean_tool_calls=_mean(r.metrics.tool_calls for r in runs),
                mean_input_tokens=_mean(r.metrics.input_tokens for r in runs),
                mean_output_tokens=_mean(r.metrics.output_tokens for r in runs),
                mean_total_tokens=_mean(r.metrics.total_tokens for r in runs),
                tokens_per_success=total_tokens / successes if successes else None,
                mean_wall_seconds=_mean(r.metrics.wall_seconds for r in runs),
                mean_llm_seconds=_mean(r.metrics.llm_seconds for r in runs),
                mean_regret=_mean(regrets) if regrets else None,
                optimal_rate=sum(r == 0 for r in regrets) / len(regrets) if regrets else None,
                mean_trace_events=_mean(r.metrics.trace_events for r in runs),
                max_trace_events=max((r.metrics.trace_events for r in runs), default=0),
                mean_roles=_mean(len(r.metrics.llm_calls_by_role) for r in runs),
                interventions=dict(interventions),
                stop_reasons=dict(Counter(r.stop_reason for r in runs)),
            )
        )
    return summaries


def scenario_matrix(records: list[RunRecord]) -> dict[str, dict[str, dict[str, float]]]:
    cells: dict[str, dict[str, list[RunRecord]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.infra_error:
            cells[r.scenario][r.pattern].append(r)
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for scenario, by_pattern in cells.items():
        matrix[scenario] = {}
        for pattern, runs in by_pattern.items():
            matrix[scenario][pattern] = {
                "successes": sum(r.success for r in runs),
                "runs": len(runs),
                "mean_tokens": _mean(r.metrics.total_tokens for r in runs),
                "mean_llm_calls": _mean(r.metrics.llm_calls for r in runs),
            }
    return matrix


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _header(meta: dict[str, Any]) -> list[str]:
    out = [f"# {meta.get('title', 'Evaluation results')}", ""]
    bullets = [f"- **{key}**: {value}" for key, value in meta.items() if key != "title"]
    return out + ([*bullets, ""] if bullets else [])


def render_markdown(records: list[RunRecord], meta: dict[str, Any]) -> str:
    """Report of one evaluation (one model)."""
    return "\n".join(_header(meta) + render_sections(records))


def render_pooled(by_source: dict[str, list[RunRecord]], meta: dict[str, Any]) -> str:
    """One report over several evaluations of the same code, one per serving model.

    Every trial is run entirely by one model, so within a trial all patterns face the
    same model. The consistency table shows each pattern's success with each model; the
    remaining sections pool every run, and pass^k then means "succeeded in every trial,
    whichever model served it".
    """
    pooled = [r for records in by_source.values() for r in records]
    patterns = [p for p in PATTERN_ORDER if any(r.pattern == p for r in pooled)]
    out = _header(meta)
    out += [
        "## Consistency across serving models",
        "",
        "Successful runs per pattern, for each model that served the agent.",
        "",
        "| Model | " + " | ".join(PATTERN_LABELS.get(p, p) for p in patterns) + " |",
        "|---|" + "---|" * len(patterns),
    ]
    for source, records in by_source.items():
        by_pattern = {s.pattern: s for s in summarize_patterns(records)}
        cells = [
            f"{by_pattern[p].successes}/{by_pattern[p].runs}" if p in by_pattern else "-"
            for p in patterns
        ]
        out.append(f"| {source} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out + render_sections(pooled))


def render_sections(records: list[RunRecord]) -> list[str]:
    summaries = summarize_patterns(records)
    matrix = scenario_matrix(records)
    patterns = [s.pattern for s in summaries]
    labels = [PATTERN_LABELS.get(p, p) for p in patterns]
    scenario_order = list(dict.fromkeys(r.scenario for r in records))

    out: list[str] = []
    out += [
        "## Overall",
        "",
        "| Pattern | Success (95% CI) | pass^k | LLM calls | Tool calls | Tokens "
        "| Tokens per success | Wall time | Price regret | Cheapest option |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        regret = f"{s.mean_regret:,.0f} VND" if s.mean_regret is not None else "-"
        optimal = _pct(s.optimal_rate) if s.optimal_rate is not None else "-"
        per_success = f"{s.tokens_per_success:,.0f}" if s.tokens_per_success else "-"
        out.append(
            f"| {PATTERN_LABELS.get(s.pattern, s.pattern)} | {s.successes}/{s.runs} = "
            f"{_pct(s.success_rate)} ({_pct(s.ci_low)}-{_pct(s.ci_high)}) | {_pct(s.pass_hat_k)} | "
            f"{s.mean_llm_calls:.1f} | {s.mean_tool_calls:.1f} | {s.mean_total_tokens:,.0f} | "
            f"{per_success} | {s.mean_wall_seconds:.0f}s | {regret} | {optimal} |"
        )
    out.append("")

    out += ["## Success by scenario", "", "| Scenario | " + " | ".join(labels) + " |"]
    out.append("|---|" + "---|" * len(labels))
    for scenario in scenario_order:
        row = [scenario]
        for p in patterns:
            cell = matrix.get(scenario, {}).get(p)
            row.append(f"{cell['successes']:.0f}/{cell['runs']:.0f}" if cell else "-")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    out += ["## Mean tokens per run by scenario", "", "| Scenario | " + " | ".join(labels) + " |"]
    out.append("|---|" + "---|" * len(labels))
    for scenario in scenario_order:
        row = [scenario]
        for p in patterns:
            cell = matrix.get(scenario, {}).get(p)
            row.append(
                f"{cell['mean_tokens']:,.0f} ({cell['mean_llm_calls']:.1f} calls)" if cell else "-"
            )
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    reasons = sorted({reason for s in summaries for reason in s.stop_reasons})
    out += ["## How runs ended", "", "| Stop reason | " + " | ".join(labels) + " |"]
    out.append("|---|" + "---|" * len(labels))
    for reason in reasons:
        out.append(
            f"| {reason} | "
            + " | ".join(str(s.stop_reasons.get(reason, 0)) for s in summaries)
            + " |"
        )
    out.append("")

    out += [
        "## Debuggability",
        "",
        "How much a person must read to reconstruct a run: trace events (model calls, tool "
        "calls, gate decisions, approvals, plans, deviations...) and the number of model "
        "roles whose decisions interleave.",
        "",
        "| Measure | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
        "| Trace events per run (mean) | "
        + " | ".join(f"{s.mean_trace_events:.1f}" for s in summaries)
        + " |",
        "| Trace events per run (max) | "
        + " | ".join(str(s.max_trace_events) for s in summaries)
        + " |",
        "| Model roles per run | " + " | ".join(f"{s.mean_roles:.1f}" for s in summaries) + " |",
        "| Re-plans per run | "
        + " | ".join(f"{s.interventions.get('replans', 0) / max(s.runs, 1):.2f}" for s in summaries)
        + " |",
        "",
    ]

    out += ["## Harness interventions (totals)", "", "| Signal | " + " | ".join(labels) + " |"]
    out.append("|---|" + "---|" * len(labels))
    for key, title in INTERVENTIONS:
        out.append(
            f"| {title} | " + " | ".join(str(s.interventions.get(key, 0)) for s in summaries) + " |"
        )
    out.append("")

    infra = sum(s.infra_errors for s in summaries)
    if infra:
        out.append(
            f"_{infra} run(s) ended with provider errors after retries and are excluded above._"
        )
        out.append("")
    return out
