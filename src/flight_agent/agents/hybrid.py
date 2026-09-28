"""Pattern 3 - Hybrid (plan, execute, re-plan on significant deviation).

It starts like Plan-then-Execute (a reviewed plan, one tool call per step), but a
*computational* sensor checks every observation. When an observation deviates
significantly from what the plan assumed - a failure, a sold-out flight, a fare
that breaks the requirements or jumped far above its listed price, a declined
approval, or a plan that ran out without finishing - the re-planner revises the
remaining steps with everything observed so far. Re-plans are bounded.

Graph::

    START -> plan -> review -> execute -> monitor -+-> execute (on track)
                                                   +-> replan -> execute
                                                   +-> END
"""

from __future__ import annotations

import datetime as dt
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from flight_agent.agents.base import BookingAgent
from flight_agent.agents.planning import (
    Planner,
    PlanStep,
    Replanner,
    StepExecutor,
    StepResult,
)
from flight_agent.agents.prompts import HYBRID_PLANNER_SYSTEM, REPLANNER_SYSTEM
from flight_agent.domain import flight_key, format_vnd
from flight_agent.harness.handoff import StopReason
from flight_agent.harness.runtime import Harness


class HybridState(TypedDict, total=False):
    plan: list[PlanStep]
    cursor: int
    results: list[StepResult]
    deviation: str | None
    replans: int


def detect_deviation(harness: Harness, result: StepResult, drift_threshold: float) -> str | None:
    """Did the observation change significantly relative to what the plan assumed?

    A computational sensor: deterministic, no model call.
    """
    payload = result.payload
    status = payload.get("status")
    if status != "ok":
        return f"step {result.number} {result.tool} returned {status}: {result.summary}"
    if result.tool == "search_flights" and payload.get("count", 0) == 0:
        return "the search returned no flights"
    if result.tool == "check_seat":
        if not payload.get("available"):
            return f"{payload.get('flight_no')} is sold out"
        c = harness.constraints
        day = dt.date.fromisoformat(payload["date"])
        flight = harness.backend.lookup_flight(payload["flight_no"], day)
        if flight is not None:
            violations = c.check(
                origin=flight.origin,
                destination=flight.destination,
                date=flight.date,
                depart=flight.depart,
                price=int(payload["price"]),
                refundable=bool(payload["refundable"]),
            )
            if violations:
                return f"{flight.flight_no}: " + "; ".join(str(v) for v in violations)
        candidate = harness.world.candidates.get(flight_key(payload["flight_no"], day))
        if candidate and payload["price"] > candidate.listed_price * (1 + drift_threshold):
            return (
                f"the live fare of {candidate.flight_no} is {format_vnd(int(payload['price']))}, "
                f"far above its listed {format_vnd(candidate.listed_price)}; "
                "cheaper options may exist"
            )
    return None


class HybridAgent(BookingAgent):
    name = "hybrid"

    def __init__(
        self,
        model: BaseChatModel,
        *,
        executor_model: BaseChatModel | None = None,
        max_plan_steps: int = 12,
        step_retries: int = 1,
        max_replans: int = 3,
        drift_threshold: float = 0.10,
    ) -> None:
        self.planner = Planner(
            model, system_template=HYBRID_PLANNER_SYSTEM, max_steps=max_plan_steps
        )
        self.replanner = Replanner(
            model, system_template=REPLANNER_SYSTEM, max_steps=max_plan_steps
        )
        self.executor = StepExecutor(executor_model or model, step_retries=step_retries)
        self.max_replans = max_replans
        self.drift_threshold = drift_threshold

    def _run(self, harness: Harness) -> str | None:
        graph = self.build_graph(harness)
        graph.invoke({}, config={"recursion_limit": 6 * harness.config.budget.max_tool_calls + 30})
        return None

    def build_graph(self, harness: Harness) -> Any:
        def plan(state: HybridState) -> dict[str, Any]:
            drafted = self.planner.draft(harness)
            if drafted is None:
                harness.stop(StopReason.PLAN_FAILED, "the planner did not produce a valid plan")
                return {}
            harness.counters["plan_steps"] += len(drafted.steps)
            harness.tracer.emit("plan", steps=[s.model_dump() for s in drafted.steps])
            return {"plan": drafted.steps, "cursor": 0, "results": [], "replans": 0}

        def review(state: HybridState) -> dict[str, Any]:
            steps = state["plan"]
            estimate = (
                f"{len(steps) + 1} model calls (1 planner + {len(steps)} executor) and "
                f"{len(steps)} tool calls, plus up to {self.max_replans} re-plans if "
                "observations deviate"
            )
            decision = harness.request_plan_approval([s.render() for s in steps], estimate)
            if not decision.approved:
                harness.stop(StopReason.NEEDS_HUMAN, f"plan rejected by {decision.approver}")
            return {}

        def execute(state: HybridState) -> dict[str, Any]:
            index = state["cursor"]
            result = self.executor.run(harness, state["plan"], index, state["results"])
            if result is None:
                return {}
            return {"results": [*state["results"], result], "cursor": index + 1}

        def monitor(state: HybridState) -> dict[str, Any]:
            if harness.stopped or not state.get("results"):
                return {"deviation": None}
            deviation = detect_deviation(harness, state["results"][-1], self.drift_threshold)
            if deviation is None and state["cursor"] >= len(state["plan"]):
                unmet = "; ".join(c.detail for c in harness.verify().unmet[:3])
                deviation = f"every planned step ran but the booking is not complete ({unmet})"
            if deviation:
                harness.tracer.emit("deviation", reason=deviation)
            return {"deviation": deviation}

        def replan(state: HybridState) -> dict[str, Any]:
            replans = state.get("replans", 0)
            if replans >= self.max_replans:
                harness.stop(
                    StopReason.REPLAN_LIMIT,
                    f"{replans} re-plans used; last deviation: {state['deviation']}",
                )
                return {}
            revision = self.replanner.revise(
                harness,
                plan=state["plan"],
                cursor=state["cursor"],
                results=state["results"],
                deviation=state["deviation"] or "",
            )
            if harness.stopped:
                return {}
            harness.counters["replans"] += 1
            if revision is None:
                harness.stop(StopReason.PLAN_FAILED, "the re-planner did not produce a valid plan")
                return {"replans": replans + 1}
            harness.tracer.emit(
                "replan",
                decision=revision.decision,
                reason=revision.reason,
                steps=[s.model_dump() for s in revision.steps],
            )
            if revision.decision == "give_up":
                harness.stop(
                    StopReason.AGENT_FINISHED, f"the re-planner gave up: {revision.reason}"
                )
                return {"replans": replans + 1}
            harness.counters["plan_steps"] += len(revision.steps)
            done = state["plan"][: state["cursor"]]
            return {"plan": done + revision.steps, "replans": replans + 1, "deviation": None}

        def after_plan(state: HybridState) -> str:
            return END if harness.stopped else "review"

        def after_review(state: HybridState) -> str:
            return END if harness.stopped else "execute"

        def after_monitor(state: HybridState) -> str:
            if harness.stopped:
                return END
            if state.get("deviation"):
                return "replan"
            return "execute"

        def after_replan(state: HybridState) -> str:
            return END if harness.stopped else "execute"

        graph = StateGraph(HybridState)
        graph.add_node("plan", plan)
        graph.add_node("review", review)
        graph.add_node("execute", execute)
        graph.add_node("monitor", monitor)
        graph.add_node("replan", replan)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", after_plan, ["review", END])
        graph.add_conditional_edges("review", after_review, ["execute", END])
        graph.add_edge("execute", "monitor")
        graph.add_conditional_edges("monitor", after_monitor, ["execute", "replan", END])
        graph.add_conditional_edges("replan", after_replan, ["execute", END])
        return graph.compile()
