from __future__ import annotations

import pytest

from flight_agent.cli import build_parser, main


def test_scenarios_command_lists_expectations(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "book VJ620@2026-10-07 (1,390,000 VND)" in out
    assert "infeasible_budget" in out and "book nothing, hand off" in out


def test_parser_rejects_unknown_pattern():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "-p", "tree_of_thoughts"])


def test_model_settings_require_a_key_or_a_factory(tmp_path, monkeypatch):
    from flight_agent.llm import ModelSettings

    for name in ("GOOGLE_API_KEY", "FLIGHT_AGENT_MODEL_FACTORY", "FLIGHT_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        ModelSettings.from_env(None)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = ModelSettings.from_env(None)
    assert settings.model == "gemini-3.6-flash" and settings.temperature is None


def test_custom_model_factory_is_loaded_from_a_file(tmp_path):
    from flight_agent.llm import ModelSettings, build_models

    factory = tmp_path / "my_factory.py"
    factory.write_text(
        "from langchain_core.language_models.fake_chat_models import FakeListChatModel\n"
        "def build(model, settings):\n"
        "    return FakeListChatModel(name=model, responses=['ok'])\n",
        encoding="utf-8",
    )
    settings = ModelSettings(model="m1", executor_model="m2", factory=f"{factory}:build")
    models = build_models(settings)
    assert models.main.name == "m1" and models.executor.name == "m2"
    assert models.name == "m1 + m2"


def _record_trace(tmp_path, scenarios):
    from flight_agent.evaluation.runner import run_scenario
    from flight_agent.llm import Models

    from fakes import DAY, PASSENGER, ScriptedChatModel, call, pay_latest, plan, step

    model = ScriptedChatModel(
        responses=[
            plan(*(step(t) for t in ("search_flights", "check_seat", "book_seat", "pay_booking"))),
            call("search_flights", origin="SGN", destination="DAD", date=DAY),
            call("check_seat", flight_no="VJ620", date=DAY),
            call("book_seat", flight_no="VJ620", date=DAY, passenger_name=PASSENGER),
            pay_latest,
        ]
    )
    _, harness = run_scenario(
        scenarios["baseline"], "plan_execute", Models(main=model, name="scripted")
    )
    return harness.tracer.write_jsonl(
        tmp_path / "scripted" / "traces" / "baseline__plan_execute__t1.jsonl"
    )


def test_replay_prints_a_recorded_run_without_a_model(tmp_path, scenarios, capsys):
    _record_trace(tmp_path, scenarios)
    args = ["replay", "--results", str(tmp_path), "-s", "baseline", "-p", "plan_execute"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Replaying" in out and "no model is called" in out
    assert "M1  planner    -> Plan(4 steps)" in out
    assert "T4  pay_booking(" in out
    assert "Result [plan_execute]: SUCCESS" in out and "Cost: 5 model calls" in out


def test_replay_lists_runs_and_reports_missing_ones(tmp_path, scenarios, capsys):
    path = _record_trace(tmp_path, scenarios)
    assert main(["replay", "--results", str(tmp_path), "--list"]) == 0
    listing = capsys.readouterr().out
    assert "scripted" in listing and "1 runs, trials 1-1" in listing
    assert "scenarios: baseline" in listing and "patterns:  plan_execute" in listing

    assert main(["replay", "--results", str(tmp_path), "-s", "stale_fare"]) == 2
    assert "No recorded run" in capsys.readouterr().err

    assert main(["replay", str(path)]) == 0  # an explicit trace path works too


def test_committed_traces_still_replay(capsys):
    from pathlib import Path

    results = Path(__file__).resolve().parents[1] / "results"
    traces = sorted(results.glob("*/traces/*.jsonl"))
    if not traces:
        pytest.skip("no recorded results in this checkout")
    assert main(["replay", str(traces[0])]) == 0
    assert "Result [" in capsys.readouterr().out
