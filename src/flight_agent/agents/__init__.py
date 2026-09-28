"""The three reasoning patterns, all running inside the same harness."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from flight_agent.agents.base import BookingAgent
from flight_agent.agents.hybrid import HybridAgent
from flight_agent.agents.plan_execute import PlanExecuteAgent
from flight_agent.agents.react import ReActAgent

PATTERNS: dict[str, type[BookingAgent]] = {
    ReActAgent.name: ReActAgent,
    PlanExecuteAgent.name: PlanExecuteAgent,
    HybridAgent.name: HybridAgent,
}


def build_agent(
    pattern: str, model: BaseChatModel, *, executor_model: BaseChatModel | None = None
) -> BookingAgent:
    if pattern == ReActAgent.name:
        return ReActAgent(model)
    if pattern == PlanExecuteAgent.name:
        return PlanExecuteAgent(model, executor_model=executor_model)
    if pattern == HybridAgent.name:
        return HybridAgent(model, executor_model=executor_model)
    raise ValueError(f"unknown pattern {pattern!r}; choose one of {', '.join(PATTERNS)}")


__all__ = [
    "PATTERNS",
    "BookingAgent",
    "HybridAgent",
    "PlanExecuteAgent",
    "ReActAgent",
    "build_agent",
]
