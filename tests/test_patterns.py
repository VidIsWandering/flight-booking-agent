"""Integration tests of the three patterns inside the harness, with a scripted model.

These run the real LangChain ``create_agent`` loop and the real LangGraph graphs;
only the model is replaced, so they check the wiring between patterns and harness.
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage, ToolMessage

from flight_agent.agents import HybridAgent, PlanExecuteAgent, ReActAgent
from flight_agent.harness import Budget, DenyAllApprover, HarnessConfig, StopReason

from fakes import (
    DAY,
    PASSENGER,
    ScriptedChatModel,
    answer,
    call,
    calls,
    pay_latest,
    plan,
    replan,
    step,
)

SEARCH = call("search_flights", origin="SGN", destination="DAD", date=DAY)


def check(flight):
    return call("check_seat", flight_no=flight, date=DAY)


def book(flight):
    return call("book_seat", flight_no=flight, date=DAY, passenger_name=PASSENGER)


BOOKING_PLAN = plan(
    step("search_flights", "find flights", f"SGN, DAD, {DAY}"),
    step("check_seat", "live fare of the cheapest compliant flight"),
    step("book_seat", "hold that flight"),
    step("pay_booking", "pay with corporate_card"),
)


def exec_search(_):
    return call("search_flights", origin="SGN", destination="DAD", date=DAY)


# ======================================================================= ReAct
def test_react_happy_path_stops_on_verified_goal(make_harness):
    model = ScriptedChatModel(responses=[SEARCH, check("VJ620"), book("VJ620"), pay_latest])
    harness = make_harness("baseline")
    result = ReActAgent(model).run(harness)

    assert result.success and result.stop_reason is StopReason.GOAL_REACHED
    assert result.booking.flight_no == "VJ620"
    # The harness ends the loop as soon as completion is verified: no 5th model call.
    assert result.metrics.llm_calls == 4 and result.metrics.tool_calls == 4
    first_system = model.requests[0]["messages"][0]
    assert isinstance(first_system, SystemMessage) and "TRIP REQUIREMENTS" in first_system.content


def test_react_repeated_failing_call_is_stopped_as_a_loop(make_harness):
    model = ScriptedChatModel(responses=[SEARCH, check("VJ620"), check("VJ620"), check("VJ620")])
    result = ReActAgent(model).run(make_harness("flaky_fare_service"))

    assert result.stop_reason is StopReason.LOOP_DETECTED
    assert result.metrics.llm_calls == 4
    # The second identical call got a warning appended to its observation.
    tool_messages = [m for m in model.requests[3]["messages"] if isinstance(m, ToolMessage)]
    assert "harness_notice" in json.loads(tool_messages[-1].content)
    assert "check_seat VJ620@2026-10-07: failed 3 time(s)" in result.handoff.attempts


def test_react_unverified_done_is_pushed_back_then_handed_off(make_harness):
    model = ScriptedChatModel(
        responses=[
            SEARCH,
            check("VJ620"),
            book("VJ620"),
            answer("Your flight is booked!"),
            answer("Payment failed. The fare was 1,250,000 VND."),
        ]
    )
    result = ReActAgent(model).run(make_harness("baseline"))

    assert not result.success and result.stop_reason is StopReason.AGENT_FINISHED
    assert result.metrics.finish_nudges == 1 and result.metrics.llm_calls == 5
    assert "[harness] The booking is not complete" in model.requests[4]["messages"][-1].content
    assert "pay it, or cancel it" in result.handoff.question
    assert [c.text for c in result.grounding.ungrounded] == ["1,250,000"]


def test_react_budget_is_enforced_before_the_next_model_call(make_harness):
    model = ScriptedChatModel(responses=[SEARCH, check("VJ620"), book("VJ620")])
    harness = make_harness(config=HarnessConfig(budget=Budget(max_llm_calls=2)))
    result = ReActAgent(model).run(harness)

    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert result.metrics.llm_calls == 2


def test_react_injected_instruction_is_blocked_by_the_gate(make_harness):
    model = ScriptedChatModel(
        responses=[SEARCH, check("VJ632"), book("VJ632"), check("VJ620"), book("VJ620"), pay_latest]
    )
    result = ReActAgent(model).run(make_harness("prompt_injection"))

    assert result.success and result.booking.flight_no == "VJ620"
    assert result.metrics.constraint_blocks == 1


def test_react_parallel_tool_calls_are_serialized_by_the_harness(make_harness):
    parallel = calls(
        ("check_seat", {"flight_no": "VJ620", "date": DAY}),
        ("check_seat", {"flight_no": "VJ624", "date": DAY}),
    )
    model = ScriptedChatModel(responses=[SEARCH, parallel, book("VJ620"), pay_latest])
    harness = make_harness("baseline")
    result = ReActAgent(model).run(harness)

    assert result.success
    assert {q.flight_no for q in harness.world.quotes.values()} == {"VJ620", "VJ624"}


# =========================================================== Plan-then-Execute
def test_plan_execute_happy_path(make_harness):
    model = ScriptedChatModel(
        responses=[BOOKING_PLAN, exec_search, check("VJ620"), book("VJ620"), pay_latest]
    )
    result = PlanExecuteAgent(model).run(make_harness("baseline"))

    assert result.success and result.metrics.plan_approvals == 1
    assert result.metrics.llm_calls == 5
    assert result.metrics.llm_calls_by_role == {"planner": 1, "executor": 4}
    executor_request = model.requests[2]
    # Each step binds exactly its own tool and forces it.
    assert executor_request["tool_names"] == ["check_seat"]
    assert executor_request["tool_choice"] == "check_seat"


def test_plan_execute_does_not_replan_when_a_step_fails(make_harness):
    model = ScriptedChatModel(responses=[BOOKING_PLAN, exec_search, check("VJ620"), book("VJ620")])
    result = PlanExecuteAgent(model).run(make_harness("stale_availability"))

    assert result.stop_reason is StopReason.PLAN_FAILED
    assert "sold_out" in result.stop_detail
    assert result.metrics.llm_calls == 4
    assert result.handoff is not None


def test_plan_execute_rejected_plan_executes_nothing(make_harness):
    model = ScriptedChatModel(responses=[BOOKING_PLAN])
    result = PlanExecuteAgent(model).run(make_harness(approver=DenyAllApprover()))

    assert result.stop_reason is StopReason.NEEDS_HUMAN
    assert result.metrics.tool_calls == 0 and result.metrics.llm_calls == 1


def test_invalid_plan_gets_one_corrective_retry(make_harness):
    model = ScriptedChatModel(
        responses=[
            plan(step("teleport", "just book it")),
            BOOKING_PLAN,
            exec_search,
            check("VJ620"),
            book("VJ620"),
            pay_latest,
        ]
    )
    result = PlanExecuteAgent(model).run(make_harness("baseline"))

    assert result.success
    assert "unknown tools: teleport" in model.requests[1]["messages"][-1].content


def test_plan_sent_as_json_text_is_repaired_by_code(make_harness):
    steps = [
        step("search_flights", "find flights", f"SGN, DAD, {DAY}"),
        step("check_seat", "live fare of the cheapest compliant flight"),
        step("book_seat", "hold that flight"),
        step("pay_booking", "pay with corporate_card"),
    ]
    text_plan = answer("Here is the plan:\n```json\n" + json.dumps({"steps": steps}) + "\n```")
    model = ScriptedChatModel(
        responses=[text_plan, exec_search, check("VJ620"), book("VJ620"), pay_latest]
    )
    result = PlanExecuteAgent(model).run(make_harness("baseline"))

    assert result.success
    assert result.metrics.structured_recovered == 1 and result.metrics.structured_failures == 0
    assert result.metrics.llm_calls == 5  # the repair costs no model call


def test_unusable_plan_answer_costs_one_corrective_retry(make_harness):
    model = ScriptedChatModel(
        responses=[
            answer("I would search, then book."),
            BOOKING_PLAN,
            exec_search,
            check("VJ620"),
            book("VJ620"),
            pay_latest,
        ]
    )
    result = PlanExecuteAgent(model).run(make_harness("baseline"))

    assert result.success and result.metrics.structured_failures == 1
    assert "could not be parsed" in model.requests[1]["messages"][-1].content


def test_transient_errors_are_retried_by_code_not_by_the_model(make_harness):
    model = ScriptedChatModel(responses=[BOOKING_PLAN, exec_search, check("VJ620")])
    result = PlanExecuteAgent(model, step_retries=1).run(make_harness("flaky_fare_service"))

    assert result.stop_reason is StopReason.PLAN_FAILED
    assert result.metrics.llm_calls == 3  # planner + 2 executor calls; the retry was free
    assert result.metrics.tool_calls == 3  # search + check + retried check


# ====================================================================== Hybrid
def test_hybrid_replans_after_a_significant_deviation(make_harness):
    model = ScriptedChatModel(
        responses=[
            BOOKING_PLAN,
            exec_search,
            check("VJ620"),
            replan(
                "continue",
                "VJ620 is sold out; try the next cheapest",
                step("check_seat", "live fare of VJ624"),
                step("book_seat", "hold VJ624"),
                step("pay_booking", "pay"),
            ),
            check("VJ624"),
            book("VJ624"),
            pay_latest,
        ]
    )
    harness = make_harness("stale_availability")
    result = HybridAgent(model).run(harness)

    assert result.success and result.booking.flight_no == "VJ624"
    assert result.metrics.replans == 1
    assert result.metrics.llm_calls_by_role == {"planner": 1, "executor": 5, "replanner": 1}
    assert harness.tracer.of_kind("deviation")[0].data["reason"] == "VJ620 is sold out"


def test_hybrid_flags_fare_drift_as_a_deviation(make_harness):
    model = ScriptedChatModel(
        responses=[
            plan(
                step("search_flights", "find flights"),
                step("check_seat", "VJ624"),
                step("book_seat", "hold"),
                step("pay_booking", "pay"),
            ),
            exec_search,
            check("VJ624"),
            replan("give_up", "testing"),
        ]
    )
    harness = make_harness("stale_fare")
    HybridAgent(model).run(harness)
    reason = harness.tracer.of_kind("deviation")[0].data["reason"]
    assert "far above its listed 1,450,000 VND" in reason


def test_hybrid_gives_up_cleanly_when_nothing_fits(make_harness):
    model = ScriptedChatModel(
        responses=[
            BOOKING_PLAN,
            exec_search,
            check("VJ620"),
            replan("give_up", "every morning fare is above the budget"),
        ]
    )
    result = HybridAgent(model).run(make_harness("infeasible_budget"))

    assert result.stop_reason is StopReason.AGENT_FINISHED
    assert "every morning fare" in result.stop_detail
    assert result.handoff.side_effects == ["None: no seat was held and nothing was paid."]


def test_hybrid_replan_budget_is_bounded(make_harness):
    model = ScriptedChatModel(
        responses=[
            BOOKING_PLAN,
            exec_search,
            check("VJ620"),
            replan("continue", "try VN122", step("check_seat", "VN122")),
            check("VN122"),
        ]
    )
    result = HybridAgent(model, max_replans=1).run(make_harness("infeasible_budget"))

    assert result.stop_reason is StopReason.REPLAN_LIMIT
    assert result.metrics.replans == 1
