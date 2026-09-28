"""Unit tests of the harness layers, driven directly through Harness.run_tool."""

from __future__ import annotations

import datetime as dt

from flight_agent.domain import parse_hhmm
from flight_agent.harness import (
    AutoApprover,
    Budget,
    DenyAllApprover,
    HarnessConfig,
    LoopConfig,
    LoopDetector,
    StallConfig,
    StallDetector,
    StopReason,
    TripConstraints,
    check_grounding,
)
from flight_agent.harness.budget import exceeded

from fakes import DAY, PASSENGER


def _book(h, flight="VJ620"):
    return h.run_tool("book_seat", {"flight_no": flight, "date": DAY, "passenger_name": PASSENGER})


def _check(h, flight="VJ620"):
    return h.run_tool("check_seat", {"flight_no": flight, "date": DAY})


def _pay(h, code, method="corporate_card"):
    return h.run_tool("pay_booking", {"booking_code": code, "payment_method": method})


# ----------------------------------------------------------- constraints as data
def test_constraints_render_request_and_pinned_block(scenarios):
    c = scenarios["baseline"].constraints
    assert "SGN to DAD on 2026-10-07" in c.request_text()
    assert "departing before 12:00" in c.request_text()
    block = c.pinned_block()
    assert "at most 2,000,000 VND" in block and PASSENGER in block


def test_constraints_report_each_violation():
    c = TripConstraints(
        origin="sgn",
        destination="dad",
        depart_date=dt.date(2026, 10, 7),
        depart_before=parse_hhmm("12:00"),
        max_price=2_000_000,
        passenger=PASSENGER,
        refundable_only=True,
    )
    violations = c.check(
        origin="SGN",
        destination="HAN",
        date=dt.date(2026, 10, 8),
        depart=parse_hhmm("13:15"),
        price=2_100_000,
        refundable=False,
    )
    assert {v.field for v in violations} == {
        "route",
        "date",
        "departure_time",
        "price",
        "refundable",
    }
    assert c.satisfied(violations) == 0
    assert (
        c.check(
            origin="SGN", destination="DAD", date=dt.date(2026, 10, 7), depart=parse_hhmm("11:59")
        )
        == []
    )


# ----------------------------------------------------------- permission gate
def test_booking_requires_a_live_quote_first(make_harness):
    h = make_harness()
    out = _book(h)
    assert out.payload["status"] == "denied"
    assert out.payload["rule"] == "live_fare_required"
    assert not out.executed and h.backend.all_bookings() == []
    _check(h)
    assert _book(h).payload["booking"]["state"] == "held"


def test_gate_blocks_flights_that_violate_requirements(make_harness):
    h = make_harness("prompt_injection")
    _check(h, "VJ632")
    out = _book(h, "VJ632")
    assert out.payload["rule"] == "constraint_violation"
    assert any("departure_time" in v for v in out.payload["violations"])
    assert h.metrics().constraint_blocks == 1


def test_gate_blocks_live_fares_over_budget(make_harness):
    h = make_harness("stale_fare")
    _check(h, "VJ620")
    out = _book(h, "VJ620")
    assert out.payload["rule"] == "constraint_violation"
    assert "price" in out.payload["violations"][0]


def test_only_one_active_hold(make_harness):
    h = make_harness()
    _check(h, "VJ620")
    _check(h, "VJ624")
    _book(h, "VJ620")
    out = _book(h, "VJ624")
    assert out.payload["rule"] == "one_active_hold"


def test_wrong_passenger_and_payment_method_are_denied(make_harness):
    h = make_harness()
    _check(h)
    wrong = h.run_tool(
        "book_seat", {"flight_no": "VJ620", "date": DAY, "passenger_name": "Someone Else"}
    )
    assert wrong.payload["rule"] == "passenger_mismatch"
    code = _book(h).payload["booking"]["booking_code"]
    assert _pay(h, code, "personal_card").payload["rule"] == "payment_method"


def test_payment_within_authority_needs_no_approval(make_harness):
    h = make_harness(approver=DenyAllApprover())
    _check(h)
    code = _book(h).payload["booking"]["booking_code"]
    out = _pay(h, code)
    assert out.payload["booking"]["state"] == "confirmed"
    assert h.metrics().approvals_requested == 0


def test_payment_above_limit_asks_the_approver(make_harness):
    h = make_harness("approval_required", approver=AutoApprover())
    _check(h, "VU750")
    code = _book(h, "VU750").payload["booking"]["booking_code"]
    out = _pay(h, code)
    assert out.executed and out.payload["booking"]["paid"] is True
    assert h.metrics().approvals_requested == 1
    assert h.world.approval_for(code).decision.approved


def test_declined_payment_is_not_asked_twice(make_harness):
    h = make_harness("approval_declined")  # simulated human declines non-refundable fares
    _check(h)
    code = _book(h).payload["booking"]["booking_code"]
    first = _pay(h, code)
    assert first.payload["status"] == "denied" and first.payload["by"] == "human"
    second = _pay(h, code)
    assert second.payload["rule"] == "human_declined"
    assert h.metrics().approvals_requested == 1
    assert h.backend.read_booking(code).paid is False


def test_unknown_tools_and_bad_arguments_are_rejected_without_execution(make_harness):
    h = make_harness()
    unknown = h.run_tool("get_order_price", {"order_id": 1})
    assert (
        unknown.payload["status"] == "invalid_param"
        and "search_flights" in unknown.payload["allowed"]
    )
    bad = h.run_tool("check_seat", {"flight": "VJ620"})
    assert bad.payload["status"] == "invalid_param" and not bad.executed


# ----------------------------------------------------------- completion criteria
def test_completion_is_verified_by_read_back(make_harness):
    h = make_harness()
    _check(h)
    code = _book(h).payload["booking"]["booking_code"]
    assert not h.verify().done
    assert "held but unpaid" in h.verify().unmet[0].detail
    _pay(h, code)
    report = h.verify()
    assert report.done
    assert {c.kind for c in report.criteria} == {"predicate", "schema", "cross_check", "human"}
    assert h.stop_signal.reason is StopReason.GOAL_REACHED


def test_calls_after_the_harness_stopped_are_skipped(make_harness):
    h = make_harness()
    h.stop(StopReason.BUDGET_EXHAUSTED, "test")
    out = _check(h)
    assert out.payload["status"] == "skipped" and not out.executed


# ----------------------------------------------------------- loop and stall
def test_loop_detector_warns_then_stops():
    d = LoopDetector(LoopConfig(window=6, repeat_limit=3))
    obs = {"status": "error", "code": "timeout"}
    assert d.observe("check_seat", {"flight_no": "VJ620"}, obs) is None
    assert d.observe("check_seat", {"flight_no": "vj620 "}, obs).level == "warn"
    assert d.observe("check_seat", {"flight_no": "VJ620"}, obs).level == "stop"


def test_polling_a_status_is_not_a_loop_too_early():
    d = LoopDetector(LoopConfig(repeat_limit=3, polling_limits={"get_booking": 5}))
    ok = {"status": "ok"}
    signals = [d.observe("get_booking", {"booking_code": "ABC123"}, ok) for _ in range(5)]
    assert [s.level if s else None for s in signals] == [None, None, None, "warn", "stop"]


def test_same_failure_with_different_arguments_is_a_loop():
    d = LoopDetector(LoopConfig(repeat_limit=3))
    bad = {"status": "invalid_param", "param": "origin"}
    signals = [
        d.observe("search_flights", {"origin": o}, bad) for o in ["Saigon", "Sai Gon", "HCMC"]
    ]
    assert [s.level if s else None for s in signals] == [None, "warn", "stop"]


def test_stall_detector_tracks_best_progress():
    s = StallDetector(StallConfig(patience=3))
    assert s.observe((1, 0, 0)) is None
    assert s.observe((1, 3, 1)) is None
    assert s.observe((1, 3, 1)) is None
    assert s.observe((1, 2, 1)).level == "warn"
    assert s.observe((1, 3, 1)).level == "stop"


def test_off_route_searches_do_not_count_as_progress(make_harness):
    h = make_harness(config=HarnessConfig(stall=StallConfig(patience=3)))
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    for date in ["2026-10-08", "2026-10-09", "2026-10-10"]:
        h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": date})
    assert h.stop_signal.reason is StopReason.STALLED
    assert h.metrics().off_task_calls == 3


# ----------------------------------------------------------- budget
def test_budget_dimensions():
    budget = Budget(
        max_llm_calls=5, max_tool_calls=5, max_total_tokens=1000, max_seconds=10, max_cost_usd=0.01
    )
    assert (
        exceeded(budget, llm_calls=4, tool_calls=4, total_tokens=999, seconds=9, cost_usd=0.001)
        is None
    )
    assert exceeded(
        budget, llm_calls=5, tool_calls=0, total_tokens=0, seconds=0, cost_usd=None
    ).startswith("model calls")
    assert exceeded(
        budget, llm_calls=0, tool_calls=0, total_tokens=1000, seconds=0, cost_usd=None
    ).startswith("tokens")
    assert exceeded(
        budget, llm_calls=0, tool_calls=0, total_tokens=0, seconds=0, cost_usd=0.02
    ).startswith("cost")


def test_tool_budget_stops_the_run(make_harness):
    h = make_harness(config=HarnessConfig(budget=Budget(max_tool_calls=2)))
    _check(h, "VJ620")
    _check(h, "VJ624")
    assert h.stop_signal.reason is StopReason.BUDGET_EXHAUSTED


# ----------------------------------------------------------- grounding
def test_grounding_flags_invented_facts():
    sources = [
        '{"flight_no": "VJ620", "price": 1390000, "depart": "05:40", "booking_code": "4XJ2QK"}'
    ]
    report = check_grounding(
        "Booked VJ620 at 5:40, code 4XJ2QK, total 1,390,000 VND plus a 30.000 fee, card VISA4412.",
        sources,
    )
    assert {c.text for c in report.ungrounded} == {"30.000", "VISA4412"}
    assert len(report.claims) == 6


# ----------------------------------------------------------- handoff
def test_handoff_points_at_an_unverified_option(make_harness):
    h = make_harness("infeasible_budget")
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    _check(h, "VJ620")
    result = h.finalize(pattern="test", final_message="No flight fits the budget.")
    assert not result.success and result.stop_reason is StopReason.AGENT_FINISHED
    assert "listed, not verified" in result.handoff.question


def test_handoff_for_infeasible_request_lists_closest_options(make_harness):
    h = make_harness("infeasible_budget", config=HarnessConfig(stall=StallConfig(patience=10)))
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    for flight in ["VJ620", "VN122", "QH118", "VJ624", "VN126", "VU750"]:
        _check(h, flight)
    result = h.finalize(pattern="test", final_message="No flight fits the budget.")
    handoff = result.handoff
    assert "No flight satisfies every requirement" in handoff.question
    # one option per kind of compromise: a later flight, or a morning flight over budget
    assert "(a) VJ632 15:40 at 1,190,000 VND" in handoff.question
    assert "(b) VJ620 05:40 at 2,080,000 VND (live)" in handoff.question
    assert any("VJ620" in a and "2,080,000" in a for a in handoff.attempts)
    assert handoff.side_effects == ["None: no seat was held and nothing was paid."]
    assert "### Handoff" in handoff.to_markdown()


def test_handoff_asks_about_an_unpaid_hold(make_harness):
    h = make_harness()
    _check(h)
    code = _book(h).payload["booking"]["booking_code"]
    h.stop(StopReason.LOOP_DETECTED, "test")
    result = h.finalize(pattern="test")
    assert code in result.handoff.question and "pay it, or cancel it" in result.handoff.question
    assert any("held" in s for s in result.handoff.side_effects)


def test_pinned_context_reports_holds_and_declines(make_harness):
    h = make_harness("approval_declined")
    _check(h)
    code = _book(h).payload["booking"]["booking_code"]
    _pay(h, code)
    context = h.pinned_context()
    assert f"Seat on hold, unpaid: {code}" in context
    assert "The approver DECLINED" in context
    assert "TRIP REQUIREMENTS" in context


def test_exploring_after_a_setback_counts_as_progress():
    s = StallDetector(StallConfig(patience=3))
    s.observe((3, 4, 1))  # holding a compliant seat
    assert s.observe((2, 4, 1)) is None  # hold cancelled after a declined payment
    assert s.observe((2, 4, 2)) is None  # a new option was evaluated
    assert s.idle == 0


def test_handoff_does_not_recommend_a_flight_whose_fare_check_failed(make_harness):
    h = make_harness("flaky_fare_service")
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    _check(h, "VJ620")
    _check(h, "VJ620")
    h.stop(StopReason.PLAN_FAILED, "test")
    handoff = h.finalize(pattern="test").handoff
    assert handoff.question.startswith("VJ624 08:30 at 1,450,000 VND (listed, not verified)")
    assert any("VJ620" in o and "check failed" in o for o in handoff.options)


def test_handoff_after_a_declined_payment_proposes_an_alternative(make_harness):
    h = make_harness("approval_declined")
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    _check(h, "VJ620")
    code = _book(h).payload["booking"]["booking_code"]
    _pay(h, code)  # declined by the simulated human: non-refundable
    h.stop(StopReason.PLAN_FAILED, "test")
    question = h.finalize(pattern="test").handoff.question
    assert question.startswith(f"You declined paying for {code} (VJ620")
    assert "Should I cancel it and book VJ624 08:30" in question


def test_handoff_never_offers_a_declined_flight_as_compliant(make_harness):
    h = make_harness("approval_declined")
    h.run_tool("search_flights", {"origin": "SGN", "destination": "DAD", "date": DAY})
    _check(h, "VJ620")
    code = _book(h).payload["booking"]["booking_code"]
    _pay(h, code)  # declined by the simulated human: non-refundable
    h.stop(StopReason.PLAN_FAILED, "test")
    handoff = h.finalize(pattern="test").handoff
    vj620 = [o for o in handoff.options if o.startswith("VJ620")]
    assert vj620 and "approval: expected approved payment, got declined" in vj620[0]
    assert "meets every requirement" not in vj620[0]


def test_trace_printer_summarizes_structured_plan_answers():
    from flight_agent.harness.trace import ConsolePrinter, TraceEvent

    def line(role: str, text: str) -> str:
        event = TraceEvent(
            seq=1, t=0.0, kind="model_call", data={"role": role, "text": text, "tool_calls": []}
        )
        return ConsolePrinter().format(event) or ""

    plan = '{"steps": [{"tool": "search_flights"}, {"tool": "check_seat"}]}'
    assert "-> Plan(2 steps)" in line("planner", plan)
    assert "-> Replan(decision=give_up)" in line("replanner", '{"decision": "give_up", "reason": "')
    assert "-> answer: Booked VJ620." in line("react", "Booked VJ620.")
