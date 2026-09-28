"""Command-line interface.

flight-agent scenarios                         list the evaluation scenarios
flight-agent run -p react -s baseline          run one pattern on one scenario (live trace)
flight-agent replay -p hybrid -s stale_fare    replay a recorded run (no model, no API key)
flight-agent eval --trials 5                   run the full comparison and write a report
flight-agent report results/<dir>              rebuild the report from recorded runs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

from flight_agent.agents import PATTERNS
from flight_agent.domain import format_vnd
from flight_agent.evaluation.oracle import derive_expectation
from flight_agent.evaluation.report import (
    render_markdown,
    render_pooled,
    scenario_matrix,
    summarize_patterns,
)
from flight_agent.evaluation.runner import (
    ProviderUnavailable,
    load_records,
    run_evaluation,
    run_scenario,
)
from flight_agent.evaluation.scenarios import get_scenario, load_scenarios
from flight_agent.harness.approval import AutoApprover, ConsoleApprover
from flight_agent.harness.runtime import HarnessConfig
from flight_agent.harness.trace import ConsolePrinter, TraceEvent, Tracer
from flight_agent.llm import ModelSettings, build_models


def _cmd_scenarios(args: argparse.Namespace) -> int:
    for scenario in load_scenarios(args.file):
        expectation = derive_expectation(scenario)
        if expectation.outcome == "booked":
            best = f"book {expectation.optimal_key} ({format_vnd(expectation.optimal_price or 0)})"
            others = len(expectation.acceptable) - 1
            best += f", {others} other acceptable" if others else ""
        else:
            best = "book nothing, hand off"
        print(f"{scenario.id:<20} {scenario.title}")
        print(f"{'':<20} expected: {best}")
        print(f"{'':<20} probes:   {', '.join(scenario.probes)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = get_scenario(args.scenario, args.file)
    models = build_models(ModelSettings.from_env(args.env))
    approvers: dict[str, type[ConsoleApprover] | type[AutoApprover]] = {
        "console": ConsoleApprover,
        "auto": AutoApprover,
    }
    approver = approvers.get(args.approver)
    result, harness = run_scenario(
        scenario,
        args.pattern,
        models,
        trial=args.seed,
        sink=None if args.quiet else ConsolePrinter(),
        approver=approver() if approver else None,
    )
    print()
    print(result.render())
    if args.trace:
        path = harness.tracer.write_jsonl(args.trace)
        print(f"\nTrace written to {path}")
    return 0 if result.success else 1


def _recorded_traces(results: Path) -> list[Path]:
    return sorted(results.glob("*/traces/*__*__t*.jsonl"))


def _resolve_trace(args: argparse.Namespace) -> Path | None:
    if args.trace:
        return Path(args.trace)
    name = f"{args.scenario}__{args.pattern}__t{args.trial}.jsonl"
    candidates = [p for p in _recorded_traces(Path(args.results)) if p.name == name]
    if args.model:
        candidates = [p for p in candidates if p.parent.parent.name == args.model]
    return candidates[0] if candidates else None


def _list_recorded(results: Path, scenario_file: str | None) -> int:
    traces = _recorded_traces(results)
    if not traces:
        print(f"No recorded runs in {results}/", file=sys.stderr)
        return 2
    by_model: dict[str, list[Path]] = {}
    for path in traces:
        by_model.setdefault(path.parent.parent.name, []).append(path)
    parts = [p.stem.split("__") for p in traces]
    recorded = {s for s, _, _ in parts}
    suite = [s.id for s in load_scenarios(scenario_file) if s.id in recorded]
    print(f"Recorded runs in {results}/:")
    for model, paths in by_model.items():
        trials = sorted({int(p.stem.rsplit("__t", 1)[1]) for p in paths})
        print(f"  {model:<24} {len(paths)} runs, trials {trials[0]}-{trials[-1]}")
    print("  scenarios: " + ", ".join(suite + sorted(recorded - set(suite))))
    print("  patterns:  " + ", ".join(p for p in PATTERNS if any(q == p for _, q, _ in parts)))
    print("\nReplay one: flight-agent replay -s <scenario> -p <pattern> [--trial N] [--model M]")
    return 0


def _render_run_end(start: TraceEvent | None, end: TraceEvent) -> str:
    d = end.data
    pattern = start.data.get("pattern", "?") if start else "?"
    lines = [
        f"Result [{pattern}]: {'SUCCESS' if d.get('success') else 'NOT COMPLETED'}",
        f"Stop: {d.get('stop_reason')}",
        d.get("summary") or "",
    ]
    if d.get("handoff"):
        lines += ["", d["handoff"]]
    m = d.get("metrics") or {}
    lines += [
        "",
        f"Cost: {m.get('llm_calls', 0)} model calls, {m.get('tool_calls', 0)} tool calls, "
        f"{m.get('total_tokens', 0):,} tokens, {m.get('wall_seconds', 0):.1f}s",
    ]
    return "\n".join(lines)


def _cmd_replay(args: argparse.Namespace) -> int:
    results = Path(args.results)
    if args.list:
        return _list_recorded(results, args.file)
    path = _resolve_trace(args)
    if path is None or not path.exists():
        print(
            f"No recorded run for scenario={args.scenario} pattern={args.pattern} "
            f"trial={args.trial} in {results}/ (see: flight-agent replay --list)",
            file=sys.stderr,
        )
        return 2
    events = Tracer.read_jsonl(path)
    print(f"Replaying {path} (recorded run: no model is called)\n")
    printer = ConsolePrinter()
    clock = 0.0
    for event in events:
        if args.speed > 0:
            time.sleep(max(0.0, event.t - clock) / args.speed)
            clock = event.t
        printer(event)
    start = next((e for e in events if e.kind == "run_start"), None)
    end = next((e for e in reversed(events) if e.kind == "run_end"), None)
    if end is not None:
        print()
        print(_render_run_end(start, end))
    return 0


def _write_report(out_dir: Path, meta: dict) -> None:
    records = load_records(out_dir / "runs.jsonl")
    if not records:
        print(f"No runs recorded in {out_dir}", file=sys.stderr)
        return
    (out_dir / "report.md").write_text(render_markdown(records, meta), encoding="utf-8")
    summary = {
        "meta": meta,
        "patterns": [s.model_dump() for s in summarize_patterns(records)],
        "scenarios": scenario_matrix(records),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report written to {out_dir / 'report.md'}")


def _cmd_eval(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.file)
    if args.scenarios != "all":
        wanted = args.scenarios.split(",")
        scenarios = [s for s in scenarios if s.id in wanted]
    patterns = args.patterns.split(",")
    unknown = [p for p in patterns if p not in PATTERNS]
    if unknown:
        print(f"Unknown pattern(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    models = build_models(ModelSettings.from_env(args.env))
    out_dir = Path(args.out)
    config = HarnessConfig()
    meta_path = out_dir / "meta.json"
    # A resumed evaluation keeps its original start time but records the current scope.
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta = {
        "title": "Evaluation: ReAct vs Plan-then-Execute vs Hybrid",
        "model": models.name,
        "started": meta.get(
            "started", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ),
        "scenarios": len(scenarios),
        "trials per scenario and pattern": args.trials,
        "budget per run": (
            f"{config.budget.max_llm_calls} model calls, {config.budget.max_tool_calls} "
            f"tool calls, {config.budget.max_total_tokens:,} tokens, "
            f"{config.budget.max_seconds:.0f}s"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    try:
        run_evaluation(
            scenarios=scenarios,
            patterns=patterns,
            trials=args.trials,
            models=models,
            out_dir=out_dir,
            config=config,
            infra_retries=args.infra_retries,
            cooldown_s=args.cooldown,
            max_consecutive_infra_errors=args.max_infra_errors,
            sink_factory=(lambda: ConsolePrinter()) if args.verbose else (lambda: None),
        )
    except ProviderUnavailable as error:
        print(f"Stopped: {error}. Run the same command later to resume.", file=sys.stderr)
        _write_report(out_dir, meta)
        return 3
    _write_report(out_dir, meta)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    by_source = {}
    for directory in args.results:
        out_dir = Path(directory)
        meta_path = out_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        _write_report(out_dir, meta)
        by_source[meta.get("model", out_dir.name)] = [
            r for r in load_records(out_dir / "runs.jsonl") if not r.infra_error
        ]
    if args.pooled:
        pooled = [r for records in by_source.values() for r in records]
        trials = len({(r.model, r.trial) for r in pooled})
        meta = {
            "title": "Evaluation: ReAct vs Plan-then-Execute vs Hybrid",
            "serving models": ", ".join(by_source),
            "trials per scenario and pattern": f"{trials} (each trial served by one model)",
            "runs": len(pooled),
        }
        out = Path(args.pooled)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(render_pooled(by_source, meta), encoding="utf-8")
        summary = {
            "meta": meta,
            "patterns": [s.model_dump() for s in summarize_patterns(pooled)],
            "scenarios": scenario_matrix(pooled),
            "by_model": {
                model: [s.model_dump() for s in summarize_patterns(records)]
                for model, records in by_source.items()
            },
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Pooled report written to {out / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flight-agent", description=__doc__.split("\n\n")[0])
    parser.add_argument("--env", default=".env", help="dotenv file with model settings")
    parser.add_argument(
        "--file", default=None, help="scenario YAML (defaults to the built-in suite)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scenarios", help="list evaluation scenarios").set_defaults(func=_cmd_scenarios)

    run = sub.add_parser("run", help="run one pattern on one scenario")
    run.add_argument("-p", "--pattern", choices=sorted(PATTERNS), default="react")
    run.add_argument("-s", "--scenario", default="baseline")
    run.add_argument(
        "--approver",
        choices=["scenario", "console", "auto"],
        default="scenario",
        help="who answers approval requests: the scenario's simulated human, you, or auto-approve",
    )
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--trace", default=None, help="write the JSONL trace to this path")
    run.add_argument("-q", "--quiet", action="store_true", help="do not print the live trace")
    run.set_defaults(func=_cmd_run)

    rp = sub.add_parser(
        "replay", help="replay a recorded run from results/ (no model call, no API key)"
    )
    rp.add_argument("trace", nargs="?", help="path to a trace .jsonl (overrides -s/-p/--trial)")
    rp.add_argument("-s", "--scenario", default="stale_availability")
    rp.add_argument("-p", "--pattern", choices=sorted(PATTERNS), default="hybrid")
    rp.add_argument("--trial", type=int, default=1)
    rp.add_argument("--model", default=None, help="serving model directory under --results")
    rp.add_argument("--results", default="results", help="directory with recorded runs")
    rp.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="replay pace: 0 prints at once (default), 1 is real time, 10 is ten times faster",
    )
    rp.add_argument("--list", action="store_true", help="list the recorded runs and exit")
    rp.set_defaults(func=_cmd_replay)

    ev = sub.add_parser("eval", help="compare the patterns across scenarios")
    ev.add_argument("--patterns", default=",".join(PATTERNS))
    ev.add_argument("--scenarios", default="all", help="comma-separated ids, or 'all'")
    ev.add_argument("--trials", type=int, default=5)
    ev.add_argument("--out", default=f"runs/eval-{dt.date.today():%Y%m%d}")
    ev.add_argument("--infra-retries", type=int, default=2)
    ev.add_argument("--cooldown", type=float, default=30.0, help="seconds before retrying a run")
    ev.add_argument(
        "--max-infra-errors",
        type=int,
        default=3,
        help="stop the evaluation after this many consecutive runs lost to provider errors",
    )
    ev.add_argument("-v", "--verbose", action="store_true", help="print every run's live trace")
    ev.set_defaults(func=_cmd_eval)

    rep = sub.add_parser("report", help="rebuild report.md and summary.json from runs.jsonl")
    rep.add_argument("results", nargs="+", help="one or more result directories")
    rep.add_argument(
        "--pooled",
        default=None,
        help="also write one report over all given directories into this directory",
    )
    rep.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    # The Gemini SDK warns about automatic function calling on every first request;
    # the agent never uses that feature (tools run through the harness).
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
