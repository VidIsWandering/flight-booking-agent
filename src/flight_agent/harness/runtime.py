"""The harness runtime: everything around the model that is code, not the model.

Per agent loop iteration the model only proposes the next action. The harness

1. builds the context (pinned requirements + harness status),
2. gates the proposed tool call (**permission check, before execution**),
3. executes it and records the observation,
4. checks, in this order: **completion**, **loop**, **stall**, **budget**,
5. and decides: continue, wait for a human, or stop (with a handoff).

The same :class:`Harness` instance serves all three reasoning patterns, so the
patterns are compared under identical rules.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from flight_agent.backend import MockAirline
from flight_agent.contracts import Status, dump, envelope, load, status_of
from flight_agent.domain import BookingState, format_vnd
from flight_agent.harness.approval import (
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    DenyAllApprover,
)
from flight_agent.harness.budget import Budget, CallRecord, Pricing, UsageMeter, exceeded
from flight_agent.harness.completion import CompletionReport, verify_completion
from flight_agent.harness.constraints import TripConstraints
from flight_agent.harness.detectors import LoopConfig, LoopDetector, StallConfig, StallDetector
from flight_agent.harness.grounding import check_grounding
from flight_agent.harness.handoff import StopReason, StopSignal, build_handoff
from flight_agent.harness.permissions import Decision, GateResult, PermissionGate, PermissionPolicy
from flight_agent.harness.result import RunMetrics, RunResult
from flight_agent.harness.trace import Tracer
from flight_agent.harness.world import WorldModel, summarize
from flight_agent.tools import build_tools


class HarnessConfig(BaseModel):
    budget: Budget = Field(default_factory=Budget)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    stall: StallConfig = Field(default_factory=StallConfig)
    finish_nudges: int = 1
    """How many times the harness rejects an unverified "done" before accepting the stop."""


class ToolOutcome(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]
    content: str
    payload: dict[str, Any]
    executed: bool
    gate_rule: str | None = None

    @property
    def status(self) -> Status | None:
        return status_of(self.payload)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


class Harness:
    def __init__(
        self,
        *,
        constraints: TripConstraints,
        backend: MockAirline,
        policy: PermissionPolicy | None = None,
        approver: Approver | None = None,
        config: HarnessConfig | None = None,
        tracer: Tracer | None = None,
        pricing: Pricing | None = None,
    ) -> None:
        self.constraints = constraints
        self.backend = backend
        self.policy = policy or PermissionPolicy()
        self.approver: Approver = approver or DenyAllApprover()
        self.config = config or HarnessConfig()
        self.tracer = tracer or Tracer()
        self.tools: list[BaseTool] = build_tools(backend)
        self._tools = {t.name: t for t in self.tools}
        self.world = WorldModel()
        self.gate = PermissionGate(
            policy=self.policy, constraints=constraints, backend=backend, world=self.world
        )
        self.loops = LoopDetector(self.config.loop)
        self.stalls = StallDetector(self.config.stall)
        self.usage = UsageMeter(on_call=self._on_model_call, pricing=pricing)
        self.counters: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._stop: StopSignal | None = None
        self._t0 = time.monotonic()
        self._nudges_left = self.config.finish_nudges
        self.steps = 0

    # ================================================================ lifecycle
    @property
    def stop_signal(self) -> StopSignal | None:
        return self._stop

    @property
    def stopped(self) -> bool:
        return self._stop is not None

    def stop(self, reason: StopReason, detail: str) -> None:
        """Record why the run ends. The first stop wins; later ones are ignored."""
        with self._lock:
            if self._stop is None:
                self._stop = StopSignal(reason=reason, detail=detail, step=self.steps)
                self.tracer.emit("stop", reason=reason.value, detail=detail)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def tools_named(self, *names: str) -> list[BaseTool]:
        return [self._tools[n] for n in names]

    def start(self, *, pattern: str, label: str = "") -> None:
        self.tracer.emit(
            "run_start", pattern=pattern, label=label, request=self.constraints.request_text()
        )

    # ============================================================ guides (context)
    def pinned_context(self) -> str:
        """Requirements and harness status, re-sent with every model call."""
        lines = [self.constraints.pinned_block(), "", "HARNESS STATUS:"]
        bookings = self.backend.all_bookings()
        for b in bookings:
            if b.state is BookingState.HELD:
                lines.append(
                    f"- Seat on hold, unpaid: {b.code} ({b.flight_no} {b.depart:%H:%M}, "
                    f"{format_vnd(b.price)}, {'refundable' if b.refundable else 'non-refundable'})."
                )
            elif b.state is BookingState.CONFIRMED:
                lines.append(f"- Confirmed and paid: {b.code} ({b.flight_no}).")
        for record in self.world.approvals:
            if record.request.kind == "action" and not record.decision.approved:
                lines.append(f"- The approver DECLINED: {record.request.action}.")
        lines.append(
            f"- Payments above {format_vnd(self.policy.auto_approve_limit)}"
            + (
                " or for non-refundable fares"
                if self.policy.nonrefundable_requires_approval
                else ""
            )
            + " need human approval; the harness requests it automatically when you call "
            "pay_booking."
        )
        budget = self.config.budget
        lines.append(
            f"- Budget left: {max(budget.max_llm_calls - self.usage.llm_calls, 0)} model calls, "
            f"{max(budget.max_tool_calls - self._executed_calls(), 0)} tool calls."
        )
        return "\n".join(lines)

    # ================================================= model-call boundary checks
    def before_model_call(self) -> bool:
        """Return ``False`` when the run must not spend another model call."""
        with self._lock:
            if self._stop is not None:
                return False
            reason = self._budget_exceeded()
            if reason:
                self.stop(StopReason.BUDGET_EXHAUSTED, reason)
                return False
            return True

    def _on_model_call(self, record: CallRecord) -> None:
        if record.error:
            self.counters["llm_errors"] += 1
        self.tracer.emit("model_call", **record.model_dump())

    # ================================================================ tool calls
    def run_tool(
        self,
        name: str,
        args: dict[str, Any] | None,
        *,
        execute: Callable[[], Any] | None = None,
    ) -> ToolOutcome:
        """Gate, execute, record and check one proposed tool call.

        ``execute`` lets a framework adapter run the call its own way (for example the
        LangChain tool node); by default the harness invokes the tool itself.
        """
        with self._lock:
            self.steps += 1
            step = self.steps
            args = dict(args or {})
            self.counters["proposed"] += 1

            if self._stop is not None:
                payload = envelope(
                    Status.SKIPPED,
                    message=f"The harness stopped the run ({self._stop.reason.value}); "
                    "this call was not executed.",
                )
                return self._outcome(step, name, args, payload, executed=False, checks=False)

            tool = self._tools.get(name)
            if tool is None:
                payload = envelope(
                    Status.INVALID_PARAM,
                    param="tool",
                    message=f"Unknown tool {name!r}.",
                    allowed=sorted(self._tools),
                )
                return self._outcome(step, name, args, payload, executed=False)

            try:
                args = tool.args_schema.model_validate(args).model_dump()  # type: ignore[union-attr]
            except ValidationError as error:
                first = error.errors()[0]
                payload = envelope(
                    Status.INVALID_PARAM,
                    param=".".join(str(p) for p in first.get("loc", ())) or "args",
                    message=first.get("msg", "invalid arguments"),
                    expected=str(sorted(tool.args)),
                )
                return self._outcome(step, name, args, payload, executed=False)

            gate = self.gate.evaluate(name, args)
            if gate.decision is Decision.ASK:
                gate = self._ask(step, gate)
            if gate.decision is Decision.DENY:
                self.counters["gate_denials"] += 1
                if gate.rule == "constraint_violation":
                    self.counters["constraint_blocks"] += 1
                self.world.record_denial(step, f"{name}({_fmt_args(args)})", gate.reason)
                payload = envelope(
                    Status.DENIED,
                    by="human" if gate.rule == "human_declined" else "policy",
                    rule=gate.rule,
                    reason=gate.reason,
                    violations=[str(v) for v in gate.violations] or None,
                    hint=gate.hint,
                )
                return self._outcome(step, name, args, payload, executed=False, gate=gate)

            raw = execute() if execute is not None else tool.invoke(args)
            if isinstance(raw, ToolMessage):
                raw = raw.content
            payload = load(raw) or envelope(
                Status.ERROR,
                code="unparseable_observation",
                retryable=False,
                message=str(raw)[:300],
            )
            return self._outcome(step, name, args, payload, executed=True, gate=gate)

    def _ask(self, step: int, gate: GateResult) -> GateResult:
        assert gate.approval is not None
        decision = self._decide(gate.approval, step)
        if decision.approved:
            return GateResult(
                decision=Decision.ALLOW, rule="approved", reason=f"approved by {decision.approver}"
            )
        note = f" ({decision.note})" if decision.note else ""
        return GateResult(
            decision=Decision.DENY,
            rule="human_declined",
            reason=f"The approver declined: {gate.approval.action}{note}.",
            hint="Do not retry this action. Choose another option, or stop and explain.",
        )

    def _decide(self, request: ApprovalRequest, step: int) -> ApprovalDecision:
        counter = "plan_approvals" if request.kind == "plan" else "approvals_requested"
        self.counters[counter] += 1
        decision = self.approver.decide(request)
        self.world.record_approval(step, request, decision)
        if not decision.approved and request.kind == "action":
            self.counters["approvals_declined"] += 1
        self.tracer.emit(
            "approval",
            kind=request.kind,
            action=request.action,
            why=request.why,
            approved=decision.approved,
            approver=decision.approver,
            note=decision.note,
        )
        return decision

    def request_plan_approval(self, steps: list[str], estimate: str) -> ApprovalDecision:
        """Plan-then-execute: a human reviews the whole plan before anything runs."""
        with self._lock:
            request = ApprovalRequest(
                kind="plan",
                where="A plan has been drafted; nothing has been executed yet.",
                action="\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)),
                why=f"Plans are reviewed before execution. Estimated cost: {estimate}.",
            )
            return self._decide(request, self.steps)

    def _outcome(
        self,
        step: int,
        name: str,
        args: dict[str, Any],
        payload: dict[str, Any],
        *,
        executed: bool,
        gate: GateResult | None = None,
        checks: bool = True,
    ) -> ToolOutcome:
        pending_stop: tuple[StopReason, str] | None = None
        notices: list[str] = []
        if checks:
            self.world.record(
                step=step,
                tool=name,
                args=args,
                payload=payload,
                raw=dump(payload),
                executed=executed,
            )
            if self._off_task(name, args):
                self.counters["off_task_calls"] += 1
            notices, pending_stop = self._after_observation(name, args, payload)
            if notices:
                payload = {**payload, "harness_notice": " ".join(notices)}
        self.tracer.emit(
            "tool_call",
            step=step,
            tool=name,
            args=args,
            status=payload.get("status"),
            summary=summarize(name, payload),
            executed=executed,
            gate=gate.rule if gate else None,
            notice=" ".join(notices) or None,
            observation=payload,
        )
        if pending_stop is not None:
            self.stop(*pending_stop)
        return ToolOutcome(
            step=step,
            tool=name,
            args=args,
            content=dump(payload),
            payload=payload,
            executed=executed,
            gate_rule=gate.rule if gate else None,
        )

    def _after_observation(
        self, name: str, args: dict[str, Any], payload: dict[str, Any]
    ) -> tuple[list[str], tuple[StopReason, str] | None]:
        """Post-observation checklist. Order matters: completion, loop, stall, budget."""
        notices: list[str] = []
        report = self.verify()
        if report.done:
            assert report.booking is not None
            return notices, (
                StopReason.GOAL_REACHED,
                f"{report.booking.code} confirmed, paid and verified by read-back",
            )
        loop = self.loops.observe(name, args, payload)
        if loop is not None:
            if loop.level == "stop":
                return [loop.message], (StopReason.LOOP_DETECTED, loop.message)
            self.counters["loop_warnings"] += 1
            notices.append(loop.message)
        stall = self.stalls.observe(self.progress())
        if stall is not None:
            if stall.level == "stop":
                return [*notices, stall.message], (StopReason.STALLED, stall.message)
            self.counters["stall_warnings"] += 1
            notices.append(stall.message)
        reason = self._budget_exceeded()
        if reason:
            return notices, (StopReason.BUDGET_EXHAUSTED, reason)
        return notices, None

    # ================================================================ sensors
    def verify(self) -> CompletionReport:
        return verify_completion(
            constraints=self.constraints, policy=self.policy, backend=self.backend, world=self.world
        )

    def progress(self) -> tuple[int, int, int]:
        """Task progress: (funnel stage, best requirement score, live fares seen on route)."""
        c = self.constraints
        route_quotes = []
        for quote in self.world.quotes.values():
            flight = self.backend.lookup_flight(quote.flight_no, quote.date)
            if flight and (flight.origin, flight.destination, flight.date) == (
                c.origin,
                c.destination,
                c.depart_date,
            ):
                route_quotes.append((flight, quote))
        best, compliant = 0, False
        for flight, quote in route_quotes:
            violations = c.check(
                origin=flight.origin,
                destination=flight.destination,
                date=flight.date,
                depart=flight.depart,
                price=quote.price,
                refundable=quote.refundable,
            )
            best = max(best, c.satisfied(violations))
            compliant = compliant or (not violations and quote.available)
        bookings = self.backend.all_bookings()
        held = any(b.state is BookingState.HELD and not c.check_booking(b) for b in bookings)
        paid = any(b.state is BookingState.CONFIRMED and not c.check_booking(b) for b in bookings)
        searched = any(
            (s.get("origin"), s.get("destination"), s.get("date"))
            == (c.origin, c.destination, c.depart_date.isoformat())
            and s.get("count", 0) > 0
            for s in self.world.searches
        )
        stage = 4 if paid else 3 if held else 2 if compliant else 1 if searched else 0
        return stage, best, len(route_quotes)

    def _off_task(self, name: str, args: dict[str, Any]) -> bool:
        if name == "get_airline_reviews":
            return True
        if name == "search_flights":
            c = self.constraints
            return (
                str(args.get("origin", "")).strip().upper(),
                str(args.get("destination", "")).strip().upper(),
                str(args.get("date", "")).strip(),
            ) != (c.origin, c.destination, c.depart_date.isoformat())
        return False

    def _executed_calls(self) -> int:
        return sum(1 for a in self.world.actions if a.executed)

    def _budget_exceeded(self) -> str | None:
        return exceeded(
            self.config.budget,
            llm_calls=self.usage.llm_calls,
            tool_calls=self._executed_calls(),
            total_tokens=self.usage.total_tokens,
            seconds=self.elapsed(),
            cost_usd=self.usage.cost_usd,
        )

    # ============================================================ finishing
    def review_finish(self, final_text: str) -> str | None:
        """The model wants to stop. Verify before trusting; maybe push back once.

        Returns a message to send back to the model, or ``None`` to accept the stop.
        """
        with self._lock:
            if self._stop is not None:
                return None
            report = self.verify()
            if report.done:
                self.stop(StopReason.GOAL_REACHED, "verified when the agent finished")
                return None
            if self._nudges_left <= 0:
                return None
            self._nudges_left -= 1
            self.counters["finish_nudges"] += 1
            unmet = "; ".join(f"{c.name}: {c.detail}" for c in report.unmet[:4])
            message = (
                f"[harness] The booking is not complete ({unmet}). Continue with the tools if "
                "an option can still satisfy every requirement. If none can, do not book "
                "anything: reply with a short explanation instead."
            )
            self.tracer.emit("nudge", message=message)
            return message

    def finalize(self, *, pattern: str, final_message: str | None = None) -> RunResult:
        with self._lock:
            report = self.verify()
            if report.done and self._stop is None:
                self.stop(StopReason.GOAL_REACHED, "verified at the end of the run")
            if self._stop is None:
                self.stop(StopReason.AGENT_FINISHED, "the agent ended without a verified booking")
            assert self._stop is not None
            grounding = None
            if final_message:
                # Everything the model was shown: every observation (including denials
                # and harness notices), the request and the pinned context.
                seen = [dump(e.data["observation"]) for e in self.tracer.of_kind("tool_call")]
                grounding = check_grounding(
                    final_message,
                    [*seen, self.constraints.request_text(), self.pinned_context()],
                )
            handoff = None
            if report.done:
                b = report.booking
                assert b is not None
                summary = (
                    f"Booked {b.flight_no} {b.origin}->{b.destination} on {b.date} at "
                    f"{b.depart:%H:%M} for {b.passenger}: booking {b.code}, paid "
                    f"{format_vnd(b.price)} "
                    f"({'refundable' if b.refundable else 'non-refundable'}). "
                    "Verified by reading the booking back from the airline."
                )
            else:
                handoff = build_handoff(
                    stop=self._stop,
                    constraints=self.constraints,
                    world=self.world,
                    backend=self.backend,
                    completion=report,
                )
                summary = f"Not booked. {handoff.question}"
            metrics = self.metrics()
            result = RunResult(
                pattern=pattern,
                success=report.done,
                stop_reason=self._stop.reason,
                stop_detail=self._stop.detail,
                summary=summary,
                final_message=final_message,
                booking=report.booking if report.done else None,
                completion=report,
                handoff=handoff,
                grounding=grounding,
                metrics=metrics,
            )
            self.tracer.emit(
                "run_end",
                success=result.success,
                stop_reason=result.stop_reason.value,
                summary=result.summary,
                final_message=final_message,
                ungrounded=[c.text for c in grounding.ungrounded] if grounding else [],
                handoff=handoff.to_markdown() if handoff else None,
                metrics=metrics.model_dump(),
            )
            return result

    def metrics(self) -> RunMetrics:
        by_role: Counter[str] = Counter(c.role for c in self.usage.calls)
        return RunMetrics(
            llm_calls=self.usage.llm_calls,
            llm_calls_by_role=dict(by_role),
            tool_calls=self._executed_calls(),
            tool_calls_proposed=self.counters["proposed"],
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            reasoning_tokens=self.usage.reasoning_tokens,
            total_tokens=self.usage.total_tokens,
            llm_seconds=round(self.usage.llm_seconds, 3),
            wall_seconds=round(self.elapsed(), 3),
            cost_usd=self.usage.cost_usd,
            gate_denials=self.counters["gate_denials"],
            constraint_blocks=self.counters["constraint_blocks"],
            approvals_requested=self.counters["approvals_requested"],
            approvals_declined=self.counters["approvals_declined"],
            plan_approvals=self.counters["plan_approvals"],
            loop_warnings=self.counters["loop_warnings"],
            stall_warnings=self.counters["stall_warnings"],
            off_task_calls=self.counters["off_task_calls"],
            finish_nudges=self.counters["finish_nudges"],
            plan_steps=self.counters["plan_steps"],
            replans=self.counters["replans"],
            structured_failures=self.counters["structured_failures"],
            structured_recovered=self.counters["structured_recovered"],
            llm_errors=self.counters["llm_errors"],
            trace_events=len(self.tracer.events),
        )


def _fmt_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())
