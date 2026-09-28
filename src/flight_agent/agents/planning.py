"""Building blocks shared by the planning patterns (plan-then-execute and hybrid).

* :class:`Planner`      - one structured-output call that returns a whole plan,
  statically validated by code before anyone sees it;
* :class:`StepExecutor` - executes one plan step with exactly one tool call; only the
  step's own tool is bound, and it is forced;
* :class:`Replanner`    - (hybrid only) revises the remaining steps after a deviation.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, ValidationError

from flight_agent.agents.prompts import EXECUTOR_SYSTEM, tool_catalog
from flight_agent.harness.runtime import Harness
from flight_agent.harness.world import summarize


class PlanStep(BaseModel):
    tool: str = Field(description="Exactly one tool name from the tool list.")
    purpose: str = Field(description="What this step must achieve.")
    args_hint: str = Field(
        default="", description="Argument values, or how to derive them from earlier steps."
    )

    def render(self) -> str:
        hint = f" [{self.args_hint}]" if self.args_hint else ""
        return f"{self.tool}: {self.purpose}{hint}"


class Plan(BaseModel):
    """An ordered plan of tool calls that achieves the goal."""

    steps: list[PlanStep] = Field(description="Ordered steps, one tool call each.")


class Replan(BaseModel):
    """A revision of the remaining plan after the harness detected a deviation."""

    decision: Literal["continue", "give_up"] = Field(
        description="continue with new remaining steps, or give_up when nothing can satisfy "
        "every requirement."
    )
    reason: str = Field(description="Why this decision was taken.")
    steps: list[PlanStep] = Field(
        default_factory=list, description="The remaining steps when decision is continue."
    )


class StepResult(BaseModel):
    number: int
    """Execution order (1-based), across re-plans."""
    step: PlanStep
    tool: str
    args: dict[str, Any]
    status: str
    summary: str
    content: str
    payload: dict[str, Any]
    attempts: int = 1


def model_config(harness: Harness, role: str) -> RunnableConfig:
    return {"callbacks": [harness.usage], "metadata": {"agent_role": role}, "run_name": role}


S = TypeVar("S", bound=BaseModel)


def recover_structured(raw: BaseMessage | None, schema: type[S]) -> S | None:
    """Schema check with a repair step, run by code when structured output fails.

    Models sometimes answer a structured-output request in the wrong channel: as a
    function call when JSON was requested, or as JSON text (often in a code block)
    when a function call was requested. The content is often valid; only the
    channel is wrong. Returns ``None`` when nothing in the message validates.
    """
    if raw is None:
        return None
    candidates: list[Any] = [c.get("args") for c in getattr(raw, "tool_calls", None) or []]
    text = raw.text if isinstance(raw, AIMessage) else str(raw.content)
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        candidates.append(value)
    for candidate in candidates:
        try:
            return schema.model_validate(candidate)
        except ValidationError:
            continue
    return None


def invoke_structured(
    model: BaseChatModel,
    schema: type[S],
    messages: list[BaseMessage],
    harness: Harness,
    role: str,
) -> S:
    """One structured-output call. Uses the provider's default structured-output
    method (native JSON schema for Gemini) and repairs a wrong-channel answer by code.

    Raises :class:`OutputParserException` when the answer cannot be used.
    """
    structured = model.with_structured_output(schema, include_raw=True)
    out = structured.invoke(messages, config=model_config(harness, role))
    parsed = out.get("parsed") if isinstance(out, dict) else out
    if isinstance(parsed, schema):
        return parsed
    raw = out.get("raw") if isinstance(out, dict) else None
    recovered = recover_structured(raw, schema)
    if recovered is not None:
        harness.counters["structured_recovered"] += 1
        harness.tracer.emit("structured_recovered", role=role, schema=schema.__name__)
        return recovered
    error = out.get("parsing_error") if isinstance(out, dict) else None
    raise OutputParserException(
        f"no valid {schema.__name__} in the answer" + (f" ({error})" if error else "")
    )


def validate_steps(steps: list[PlanStep], tools: set[str], max_steps: int) -> list[str]:
    """Static checks run by code before a plan is shown to a human or executed."""
    errors = []
    if not steps:
        errors.append("the plan has no steps")
    if len(steps) > max_steps:
        errors.append(f"the plan has {len(steps)} steps; the maximum is {max_steps}")
    unknown = sorted({s.tool for s in steps if s.tool not in tools})
    if unknown:
        errors.append("unknown tools: " + ", ".join(unknown))
    return errors


def render_results(results: list[StepResult]) -> str:
    if not results:
        return "RESULTS SO FAR: none"
    lines = ["RESULTS SO FAR:"]
    for r in results:
        args = ", ".join(f"{k}={v}" for k, v in r.args.items())
        lines.append(f"[{r.number}] {r.tool}({args}) -> {r.content}")
    return "\n".join(lines)


def render_plan(plan: list[PlanStep], current: int | None = None) -> str:
    lines = []
    for i, step in enumerate(plan):
        mark = ""
        if current is not None:
            mark = "  (done)" if i < current else ("  <- CURRENT" if i == current else "")
        lines.append(f"{i + 1}. {step.render()}{mark}")
    return "\n".join(lines)


class Planner:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        system_template: str,
        max_steps: int = 12,
        role: str = "planner",
    ) -> None:
        self.model = model
        self.system_template = system_template
        self.max_steps = max_steps
        self.role = role

    def draft(self, harness: Harness) -> Plan | None:
        """Draft a plan; one corrective retry if it fails parsing or static validation."""
        system = self.system_template.format(
            catalog=tool_catalog(harness.tools), max_steps=self.max_steps
        )
        feedback = ""
        for _ in range(2):
            if not harness.before_model_call():
                return None
            user = (
                f"GOAL: {harness.constraints.request_text()}\n\n{harness.pinned_context()}"
                f"{feedback}\n\nReturn the plan."
            )
            try:
                plan = invoke_structured(
                    self.model,
                    Plan,
                    [SystemMessage(system), HumanMessage(user)],
                    harness,
                    self.role,
                )
            except OutputParserException as error:
                harness.counters["structured_failures"] += 1
                feedback = f"\n\nYour previous answer could not be parsed ({error}). Try again."
                continue
            errors = validate_steps(plan.steps, {t.name for t in harness.tools}, self.max_steps)
            if not errors:
                return plan
            feedback = "\n\nYour previous plan was invalid: " + "; ".join(errors) + ". Fix it."
        return None


class StepExecutor:
    def __init__(self, model: BaseChatModel, *, step_retries: int = 1) -> None:
        self.model = model
        self.step_retries = step_retries

    def run(
        self, harness: Harness, plan: list[PlanStep], index: int, results: list[StepResult]
    ) -> StepResult | None:
        """Execute ``plan[index]``. Returns ``None`` when the harness refused the model call."""
        step = plan[index]
        if not harness.before_model_call():
            return None
        (tool,) = harness.tools_named(step.tool)
        bound = self.model.bind_tools([tool], tool_choice=step.tool)
        user = (
            f"PLAN:\n{render_plan(plan, index)}\n\n"
            f"CURRENT STEP {index + 1}: {step.render()}\n\n"
            f"{render_results(results)}\n\n"
            f"Call {step.tool} now."
        )
        ai = bound.invoke(
            [SystemMessage(f"{EXECUTOR_SYSTEM}\n\n{harness.pinned_context()}"), HumanMessage(user)],
            config=model_config(harness, "executor"),
        )
        number = len(results) + 1
        if not ai.tool_calls:
            return StepResult(
                number=number,
                step=step,
                tool=step.tool,
                args={},
                status="no_tool_call",
                summary="the executor did not call the tool",
                content="",
                payload={},
            )
        call = ai.tool_calls[0]
        outcome = harness.run_tool(call["name"], call.get("args") or {})
        attempts = 1
        # Deterministic retry of transient failures: same call, no extra model call.
        while (
            outcome.payload.get("status") == "error"
            and outcome.payload.get("retryable")
            and attempts <= self.step_retries
            and not harness.stopped
        ):
            outcome = harness.run_tool(call["name"], outcome.args)
            attempts += 1
        return StepResult(
            number=number,
            step=step,
            tool=outcome.tool,
            args=outcome.args,
            status=str(outcome.payload.get("status")),
            summary=summarize(outcome.tool, outcome.payload),
            content=outcome.content,
            payload=outcome.payload,
            attempts=attempts,
        )


class Replanner:
    def __init__(self, model: BaseChatModel, *, system_template: str, max_steps: int = 10) -> None:
        self.model = model
        self.system_template = system_template
        self.max_steps = max_steps

    def revise(
        self,
        harness: Harness,
        *,
        plan: list[PlanStep],
        cursor: int,
        results: list[StepResult],
        deviation: str,
    ) -> Replan | None:
        system = self.system_template.format(
            catalog=tool_catalog(harness.tools), max_steps=self.max_steps
        )
        feedback = ""
        for _ in range(2):
            if not harness.before_model_call():
                return None
            user = (
                f"GOAL: {harness.constraints.request_text()}\n\n{harness.pinned_context()}\n\n"
                f"CURRENT PLAN:\n{render_plan(plan, cursor)}\n\n"
                f"{render_results(results)}\n\n"
                f"DEVIATION DETECTED BY THE HARNESS: {deviation}{feedback}\n\n"
                "Return your decision and the remaining steps."
            )
            try:
                revision = invoke_structured(
                    self.model,
                    Replan,
                    [SystemMessage(system), HumanMessage(user)],
                    harness,
                    "replanner",
                )
            except OutputParserException as error:
                harness.counters["structured_failures"] += 1
                feedback = f"\n\nYour previous answer could not be parsed ({error}). Try again."
                continue
            if revision.decision == "give_up":
                return revision
            errors = validate_steps(revision.steps, {t.name for t in harness.tools}, self.max_steps)
            if not errors:
                return revision
            feedback = "\n\nYour previous revision was invalid: " + "; ".join(errors) + ". Fix it."
        return None
