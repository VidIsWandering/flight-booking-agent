"""Pattern 2 - Plan-then-Execute.

One model call writes the whole plan; code validates it; a human approves it (the
plan is visible and its cost can be estimated before anything runs); an executor
then runs it step by step. There is no re-planning: if a step fails, the run stops
and hands off. That is the pattern's trade-off - reviewability and low cost in
exchange for adaptability.

Graph::

    START -> plan -> review -> execute -+-> execute (next step)
                                        +-> END (done, failed or stopped)
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from flight_agent.agents.base import BookingAgent
from flight_agent.agents.planning import (
    Planner,
    PlanStep,
    StepExecutor,
    StepResult,
)
from flight_agent.agents.prompts import PLANNER_SYSTEM
from flight_agent.harness.handoff import StopReason
from flight_agent.harness.runtime import Harness


class PlanExecuteState(TypedDict, total=False):
    plan: list[PlanStep]
    cursor: int
    results: list[StepResult]


class PlanExecuteAgent(BookingAgent):
    name = "plan_execute"

    def __init__(
        self,
        model: BaseChatModel,
        *,
        executor_model: BaseChatModel | None = None,
        max_plan_steps: int = 12,
        step_retries: int = 1,
    ) -> None:
        self.planner = Planner(model, system_template=PLANNER_SYSTEM, max_steps=max_plan_steps)
        self.executor = StepExecutor(executor_model or model, step_retries=step_retries)

    def _run(self, harness: Harness) -> str | None:
        graph = self.build_graph(harness)
        graph.invoke({}, config={"recursion_limit": 4 * harness.config.budget.max_tool_calls + 20})
        return None

    def build_graph(self, harness: Harness) -> Any:
        def plan(state: PlanExecuteState) -> dict[str, Any]:
            drafted = self.planner.draft(harness)
            if drafted is None:
                harness.stop(StopReason.PLAN_FAILED, "the planner did not produce a valid plan")
                return {}
            harness.counters["plan_steps"] += len(drafted.steps)
            harness.tracer.emit("plan", steps=[s.model_dump() for s in drafted.steps])
            return {"plan": drafted.steps, "cursor": 0, "results": []}

        def review(state: PlanExecuteState) -> dict[str, Any]:
            steps = state["plan"]
            estimate = (
                f"{len(steps) + 1} model calls (1 planner + {len(steps)} executor) and "
                f"{len(steps)} tool calls"
            )
            decision = harness.request_plan_approval([s.render() for s in steps], estimate)
            if not decision.approved:
                harness.stop(StopReason.NEEDS_HUMAN, f"plan rejected by {decision.approver}")
            return {}

        def execute(state: PlanExecuteState) -> dict[str, Any]:
            index, steps = state["cursor"], state["plan"]
            result = self.executor.run(harness, steps, index, state["results"])
            if result is None:  # the harness refused the model call (budget or stop)
                return {}
            results = [*state["results"], result]
            if not harness.stopped:
                if result.status != "ok":
                    harness.stop(
                        StopReason.PLAN_FAILED,
                        f"step {index + 1} ({result.tool}) failed: {result.summary}",
                    )
                elif index + 1 >= len(steps):
                    unmet = "; ".join(c.detail for c in harness.verify().unmet[:3])
                    harness.stop(
                        StopReason.PLAN_FAILED,
                        f"every planned step ran but the booking is not complete ({unmet})",
                    )
            return {"results": results, "cursor": index + 1}

        def after_plan(state: PlanExecuteState) -> str:
            return END if harness.stopped else "review"

        def after_review(state: PlanExecuteState) -> str:
            return END if harness.stopped else "execute"

        def after_execute(state: PlanExecuteState) -> str:
            if harness.stopped or state["cursor"] >= len(state["plan"]):
                return END
            return "execute"

        graph = StateGraph(PlanExecuteState)
        graph.add_node("plan", plan)
        graph.add_node("review", review)
        graph.add_node("execute", execute)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", after_plan, ["review", END])
        graph.add_conditional_edges("review", after_review, ["execute", END])
        graph.add_conditional_edges("execute", after_execute, ["execute", END])
        return graph.compile()
