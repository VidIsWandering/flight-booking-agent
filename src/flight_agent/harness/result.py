"""The outcome of one agent run, as produced by the harness."""

from __future__ import annotations

from pydantic import BaseModel, Field

from flight_agent.domain import Booking
from flight_agent.harness.completion import CompletionReport
from flight_agent.harness.grounding import GroundingReport
from flight_agent.harness.handoff import Handoff, StopReason


class RunMetrics(BaseModel):
    llm_calls: int = 0
    llm_calls_by_role: dict[str, int] = Field(default_factory=dict)
    tool_calls: int = 0
    """Tool calls that were actually executed."""
    tool_calls_proposed: int = 0
    """Every tool call the agent proposed, including denied and invalid ones."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    llm_seconds: float = 0.0
    wall_seconds: float = 0.0
    cost_usd: float | None = None
    gate_denials: int = 0
    constraint_blocks: int = 0
    approvals_requested: int = 0
    approvals_declined: int = 0
    plan_approvals: int = 0
    loop_warnings: int = 0
    stall_warnings: int = 0
    off_task_calls: int = 0
    finish_nudges: int = 0
    plan_steps: int = 0
    replans: int = 0
    structured_failures: int = 0
    """Planner/re-planner answers that could not be used (each costs a corrective retry)."""
    structured_recovered: int = 0
    """Wrong-channel structured answers repaired by code (no extra model call)."""
    llm_errors: int = 0
    trace_events: int = 0
    """Trace events recorded in the run: how much a person must read to debug it."""


class RunResult(BaseModel):
    pattern: str
    success: bool
    """True only when the completion criteria were verified by code."""
    stop_reason: StopReason
    stop_detail: str
    summary: str
    final_message: str | None = None
    booking: Booking | None = None
    completion: CompletionReport
    handoff: Handoff | None = None
    grounding: GroundingReport | None = None
    metrics: RunMetrics

    def render(self) -> str:
        lines = [f"Result [{self.pattern}]: {'SUCCESS' if self.success else 'NOT COMPLETED'}"]
        lines.append(f"Stop: {self.stop_reason.value} - {self.stop_detail}")
        lines.append(self.summary)
        lines.append("Completion criteria:")
        for c in self.completion.criteria:
            lines.append(f"  [{'x' if c.passed else ' '}] {c.name} ({c.kind}): {c.detail}")
        if self.grounding and self.grounding.ungrounded:
            lines.append(
                "Ungrounded claims in the final answer: "
                + ", ".join(c.text for c in self.grounding.ungrounded)
            )
        if self.handoff:
            lines.append("")
            lines.append(self.handoff.to_markdown())
        m = self.metrics
        lines.append("")
        lines.append(
            f"Cost: {m.llm_calls} model calls, {m.tool_calls} tool calls, "
            f"{m.total_tokens:,} tokens, {m.wall_seconds:.1f}s"
        )
        return "\n".join(lines)
