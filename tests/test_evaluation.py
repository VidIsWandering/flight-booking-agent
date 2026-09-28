from __future__ import annotations

from flight_agent.evaluation import derive_expectation, load_records, run_evaluation, run_scenario
from flight_agent.evaluation.report import render_markdown, summarize_patterns, wilson
from flight_agent.evaluation.scoring import score_run
from flight_agent.llm import Models

from fakes import DAY, PASSENGER, ScriptedChatModel, call, pay_latest, plan, step


def _happy_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            plan(
                step("search_flights", "find flights"),
                step("check_seat", "check the cheapest"),
                step("book_seat", "hold it"),
                step("pay_booking", "pay"),
            ),
            call("search_flights", origin="SGN", destination="DAD", date=DAY),
            call("check_seat", flight_no="VJ620", date=DAY),
            call("book_seat", flight_no="VJ620", date=DAY, passenger_name=PASSENGER),
            pay_latest,
        ]
    )


def test_oracle_derives_expected_outcomes(scenarios):
    expected = {sid: derive_expectation(s) for sid, s in scenarios.items()}
    assert expected["baseline"].optimal_key == "VJ620@2026-10-07"
    assert expected["stale_availability"].optimal_key == "VJ624@2026-10-07"
    assert expected["stale_fare"].optimal_key == "VU750@2026-10-07"
    assert "VJ624@2026-10-07" in expected["stale_fare"].acceptable
    assert expected["flaky_fare_service"].optimal_key == "VJ624@2026-10-07"
    assert expected["approval_declined"].optimal_key == "VU750@2026-10-07"
    assert "VJ620@2026-10-07" not in expected["approval_declined"].acceptable
    assert expected["infeasible_budget"].outcome == "no_booking"
    assert expected["prompt_injection"].optimal_key == "VJ620@2026-10-07"


def test_score_and_report(scenarios):
    scenario = scenarios["baseline"]
    models = Models(main=_happy_model(), name="scripted")
    result, harness = run_scenario(scenario, "plan_execute", models)
    record = score_run(
        scenario_id=scenario.id,
        pattern="plan_execute",
        trial=1,
        model=models.name,
        expectation=derive_expectation(scenario),
        result=result,
        backend=harness.backend,
    )
    assert record.success and record.price_regret == 0 and record.leaked_holds == 0

    summary = summarize_patterns([record])[0]
    assert summary.success_rate == 1.0 and summary.pass_hat_k == 1.0
    report = render_markdown([record], {"title": "Test report", "model": "scripted"})
    assert "| Plan-then-Execute | 1/1 = 100%" in report
    assert "| baseline | 1/1 |" in report
    assert "## Debuggability" in report and "| Trace events per run (mean) |" in report

    failed = record.model_copy(update={"success": False})
    both = summarize_patterns([record, failed])[0]
    assert both.success_rate == 0.5 and both.pass_hat_k == 0.0


def test_wilson_interval_is_sane():
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_run_evaluation_records_and_resumes(tmp_path, scenarios):
    scenario = scenarios["baseline"]
    models = Models(main=_happy_model(), name="scripted")
    logs: list[str] = []
    records = run_evaluation(
        scenarios=[scenario],
        patterns=["plan_execute"],
        trials=1,
        models=models,
        out_dir=tmp_path,
        log=logs.append,
    )
    assert len(records) == 1 and records[0].success
    assert (tmp_path / "traces" / "baseline__plan_execute__t1.jsonl").exists()
    assert "PASS" in logs[0]

    # A second invocation finds the recorded run and does not execute it again.
    rerun = run_evaluation(
        scenarios=[scenario],
        patterns=["plan_execute"],
        trials=1,
        models=Models(main=ScriptedChatModel(), name="scripted"),
        out_dir=tmp_path,
        log=logs.append,
    )
    assert len(rerun) == 1 and len(load_records(tmp_path / "runs.jsonl")) == 1


def test_evaluation_stops_when_the_provider_keeps_failing(tmp_path, scenarios):
    from langchain_core.exceptions import ModelRateLimitError

    from flight_agent.evaluation.runner import ProviderUnavailable

    def quota_exhausted(messages, kwargs):
        raise ModelRateLimitError("daily quota exhausted")

    models = Models(main=ScriptedChatModel(policy=quota_exhausted), name="scripted")
    logs: list[str] = []
    try:
        run_evaluation(
            scenarios=[scenarios["baseline"], scenarios["stale_fare"]],
            patterns=["react", "plan_execute"],
            trials=1,
            models=models,
            out_dir=tmp_path,
            infra_retries=0,
            max_consecutive_infra_errors=2,
            log=logs.append,
        )
    except ProviderUnavailable as error:
        assert "2 consecutive runs" in str(error)
    else:
        raise AssertionError("the evaluation should have stopped")
    # Two runs were attempted, not four; both are dropped (and re-run) on resume.
    assert len(logs) == 2
    assert all(r.infra_error for r in load_records(tmp_path / "runs.jsonl"))


def test_pooled_report_over_serving_models(scenarios):
    from flight_agent.evaluation.report import render_pooled

    scenario = scenarios["baseline"]
    models = Models(main=_happy_model(), name="model-a")
    result, harness = run_scenario(scenario, "plan_execute", models)
    record = score_run(
        scenario_id=scenario.id,
        pattern="plan_execute",
        trial=1,
        model="model-a",
        expectation=derive_expectation(scenario),
        result=result,
        backend=harness.backend,
    )
    failed = record.model_copy(update={"success": False, "model": "model-b"})
    report = render_pooled({"model-a": [record], "model-b": [failed]}, {"title": "Pooled"})
    assert "## Consistency across serving models" in report
    assert "| model-a | 1/1 |" in report and "| model-b | 0/1 |" in report
    assert "| Plan-then-Execute | 1/2 = 50%" in report and "| baseline | 1/2 |" in report
