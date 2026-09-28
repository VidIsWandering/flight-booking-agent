"""Run scenarios x patterns x trials and record scored results.

Results are appended to ``runs.jsonl`` as each run finishes, so an interrupted
evaluation (rate limits, outages) resumes where it stopped. Runs that end with an
infrastructure error are retried after a cool-down and never counted against a
pattern. When several runs in a row are lost to the provider (a daily quota, an
outage), the evaluation stops instead of burning through the remaining runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from flight_agent.agents import build_agent
from flight_agent.evaluation.oracle import derive_expectation
from flight_agent.evaluation.scenarios import Scenario
from flight_agent.evaluation.scoring import RunRecord, score_run
from flight_agent.harness.approval import Approver
from flight_agent.harness.handoff import StopReason
from flight_agent.harness.result import RunResult
from flight_agent.harness.runtime import Harness, HarnessConfig
from flight_agent.harness.trace import TraceEvent, Tracer
from flight_agent.llm import Models


class ProviderUnavailable(RuntimeError):
    """Several consecutive runs were lost to provider errors; resume later."""


def run_scenario(
    scenario: Scenario,
    pattern: str,
    models: Models,
    *,
    trial: int = 1,
    config: HarnessConfig | None = None,
    sink: Callable[[TraceEvent], None] | None = None,
    approver: Approver | None = None,
) -> tuple[RunResult, Harness]:
    """Run one pattern on one scenario in a fresh world with a fresh harness.

    ``approver`` defaults to the scenario's simulated human.
    """
    backend = scenario.build_backend(seed=trial)
    harness = Harness(
        constraints=scenario.constraints,
        backend=backend,
        policy=scenario.policy,
        approver=approver or scenario.approver(),
        config=config,
        tracer=Tracer(sink),
        pricing=models.pricing,
    )
    agent = build_agent(pattern, models.main, executor_model=models.executor)
    result = agent.run(harness, label=f"{scenario.id} (trial {trial})")
    return result, harness


def load_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    return [
        RunRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_evaluation(
    *,
    scenarios: list[Scenario],
    patterns: list[str],
    trials: int,
    models: Models,
    out_dir: Path,
    config: HarnessConfig | None = None,
    infra_retries: int = 2,
    cooldown_s: float = 30.0,
    max_consecutive_infra_errors: int = 3,
    log: Callable[[str], None] | None = None,
    sink_factory: Callable[[], Callable[[TraceEvent], None] | None] = lambda: None,
) -> list[RunRecord]:
    log = log or (lambda line: print(line, flush=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    records = [r for r in load_records(runs_path) if not r.infra_error]
    runs_path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    done = {(r.scenario, r.pattern, r.trial) for r in records}
    expectations = {s.id: derive_expectation(s) for s in scenarios}
    total = len(scenarios) * len(patterns) * trials
    lost_in_a_row = 0

    # Trials outermost and patterns innermost: provider conditions that drift over time
    # (latency, rate limits) affect every pattern alike.
    for trial in range(1, trials + 1):
        for scenario in scenarios:
            for pattern in patterns:
                if (scenario.id, pattern, trial) in done:
                    continue
                attempt = 0
                while True:
                    attempt += 1
                    result, harness = run_scenario(
                        scenario, pattern, models, trial=trial, config=config, sink=sink_factory()
                    )
                    if result.stop_reason is not StopReason.INFRA_ERROR or attempt > infra_retries:
                        break
                    wait = cooldown_s * attempt
                    log(f"   infra error ({result.stop_detail[:120]}); retrying in {wait:.0f}s")
                    time.sleep(wait)
                trace_file = out_dir / "traces" / f"{scenario.id}__{pattern}__t{trial}.jsonl"
                harness.tracer.write_jsonl(trace_file)
                record = score_run(
                    scenario_id=scenario.id,
                    pattern=pattern,
                    trial=trial,
                    model=models.name,
                    expectation=expectations[scenario.id],
                    result=result,
                    backend=harness.backend,
                    trace_file=str(trace_file.relative_to(out_dir)),
                    attempts=attempt,
                )
                with runs_path.open("a", encoding="utf-8") as fh:
                    fh.write(record.model_dump_json() + "\n")
                records.append(record)
                m = record.metrics
                log(
                    f"[{len(records):>3}/{total}] {scenario.id:<20} {pattern:<13} t{trial} "
                    f"{'PASS' if record.success else 'FAIL'}  {record.stop_reason:<17} "
                    f"llm={m.llm_calls:<2} tools={m.tool_calls:<2} tok={m.total_tokens:>7,} "
                    f"{m.wall_seconds:6.1f}s"
                )
                lost_in_a_row = lost_in_a_row + 1 if record.infra_error else 0
                if lost_in_a_row >= max_consecutive_infra_errors:
                    raise ProviderUnavailable(
                        f"{lost_in_a_row} consecutive runs ended with provider errors "
                        f"(last: {record.stop_detail[:160]})"
                    )
    return records
