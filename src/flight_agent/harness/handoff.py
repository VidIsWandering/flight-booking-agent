"""Harness layer: stop reasons and the handoff to a human.

Whenever a run stops without a verified booking, the harness writes a handoff that
a person can act on in about thirty seconds:

* **state**    - what has been done, including every side effect and its status;
* **attempts** - which directions were tried and why they failed;
* **question** - one specific question, with concrete options.

The handoff is built by code from the harness's records, not written by the model,
so it cannot omit an inconvenient side effect.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from flight_agent.backend import MockAirline
from flight_agent.domain import BookingState, format_vnd
from flight_agent.harness.completion import CompletionReport
from flight_agent.harness.constraints import TripConstraints, Violation
from flight_agent.harness.world import WorldModel


class StopReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    """Completion criteria verified by code."""
    AGENT_FINISHED = "agent_finished"
    """The model stopped on its own without a verified booking."""
    NEEDS_HUMAN = "needs_human"
    """A human decision is required to continue (e.g. the plan was rejected)."""
    LOOP_DETECTED = "loop_detected"
    STALLED = "stalled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PLAN_FAILED = "plan_failed"
    """Plan-then-execute: a step failed and the pattern does not re-plan."""
    REPLAN_LIMIT = "replan_limit"
    """Hybrid: the maximum number of re-plans was reached."""
    INFRA_ERROR = "infra_error"
    """The model provider failed (rate limit, outage) after retries."""

    @property
    def abnormal(self) -> bool:
        return self in {
            StopReason.LOOP_DETECTED,
            StopReason.STALLED,
            StopReason.BUDGET_EXHAUSTED,
            StopReason.PLAN_FAILED,
            StopReason.REPLAN_LIMIT,
            StopReason.INFRA_ERROR,
        }


class StopSignal(BaseModel):
    reason: StopReason
    detail: str
    step: int


class Handoff(BaseModel):
    stop_reason: StopReason
    headline: str
    state: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    question: str
    unmet_criteria: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        parts = [f"### Handoff: {self.headline}", ""]

        def section(title: str, items: list[str]) -> None:
            if items:
                parts.append(f"**{title}**")
                parts.extend(f"- {item}" for item in items)
                parts.append("")

        section("State", self.state)
        section("Side effects", self.side_effects)
        section("What was tried", self.attempts)
        section("Options", self.options)
        section("Unmet completion criteria", self.unmet_criteria)
        parts.append(f"**Question:** {self.question}")
        return "\n".join(parts)


class _Option(BaseModel):
    flight_no: str
    label: str
    violations: list[str]
    fields: frozenset[str]
    price: int


def build_handoff(
    *,
    stop: StopSignal,
    constraints: TripConstraints,
    world: WorldModel,
    backend: MockAirline,
    completion: CompletionReport,
) -> Handoff:
    c = constraints
    bookings = backend.all_bookings()
    holds = [b for b in bookings if b.state is BookingState.HELD]

    on_route = [
        cand
        for cand in world.candidates.values()
        if (cand.origin, cand.destination, cand.date) == (c.origin, c.destination, c.depart_date)
    ]
    route_quotes = [q for q in world.quotes.values() if q.key in {cand.key for cand in on_route}]

    state = [
        f"Searched {len(world.searches)} time(s); {len(on_route)} flight(s) listed for "
        f"{c.origin}->{c.destination} on {c.depart_date}.",
        f"Checked live fares for {len(route_quotes)} of them.",
    ]
    side_effects = []
    for b in bookings:
        status = b.state.value + (", paid" if b.paid else "") + (", refunded" if b.refunded else "")
        side_effects.append(
            f"{b.code}: {b.flight_no} {b.date} {b.depart:%H:%M}, {format_vnd(b.price)} - {status}"
        )
    if not side_effects:
        side_effects.append("None: no seat was held and nothing was paid.")

    attempts: list[str] = []
    for q in sorted(route_quotes, key=lambda q: q.step):
        problems = [
            str(v)
            for v in c.check(
                origin=c.origin,
                destination=c.destination,
                date=q.date,
                depart=q.depart,
                price=q.price,
                refundable=q.refundable,
            )
        ]
        if not q.available:
            problems.insert(0, "sold out")
        verdict = "; ".join(problems) if problems else "satisfies every requirement"
        attempts.append(
            f"{q.flight_no} {q.depart:%H:%M}: live fare {format_vnd(q.price)} - {verdict}"
        )
    for key, count in world.quote_failures.items():
        attempts.append(f"check_seat {key}: failed {count} time(s)")
    for record in world.approvals:
        if not record.decision.approved:
            note = f" ({record.decision.note})" if record.decision.note else ""
            attempts.append(f"Approver declined: {record.request.action}{note}")
    for _, action, reason in world.denials[-5:]:
        attempts.append(f"Harness denied {action}: {reason}")

    declined = [r for r in world.approvals if not r.decision.approved]
    declined_flights = set()
    for record in declined:
        booking = (
            backend.read_booking(record.request.booking_code)
            if record.request.booking_code
            else None
        )
        if booking is not None:
            declined_flights.add(booking.flight_key)
    ranked = _rank_options(c, world, on_route, declined_flights)
    compliant = [o for o in ranked if not o.violations]
    options = _one_per_compromise(ranked)

    if holds:
        b = holds[0]
        held = f"{b.code} ({b.flight_no} {b.depart:%H:%M}, {format_vnd(b.price)})"
        if world.denied_payment(b.code):
            others = [o for o in compliant if o.flight_no != b.flight_no]
            instead = f" and book {others[0].label} instead" if others else ""
            question = (
                f"You declined paying for {held}, which is still on hold. "
                f"Should I cancel it{instead}?"
            )
        else:
            question = f"Booking {held} is on hold but unpaid. Should I pay it, or cancel it?"
    elif stop.reason is StopReason.INFRA_ERROR:
        question = "The model provider failed before the task finished. Should I retry the run?"
    elif compliant:
        question = (
            f"{compliant[0].label} appears to satisfy every requirement, but the run stopped "
            f"({stop.reason.value}). Should I book it?"
        )
    elif options:
        labels = "; ".join(f"({chr(97 + i)}) {o.label}" for i, o in enumerate(options[:3]))
        question = (
            "No flight satisfies every requirement. Should I book one of the closest options "
            f"instead, or stop? {labels}"
        )
    else:
        question = "No bookable flight was found. Should I search another date or route?"
    if declined and not holds:
        question = f"You declined: {declined[-1].request.action}. " + question

    return Handoff(
        stop_reason=stop.reason,
        headline=f"{stop.reason.value.replace('_', ' ')} - {stop.detail}",
        state=state,
        side_effects=side_effects,
        attempts=attempts,
        options=[
            o.label
            + (" - " + "; ".join(o.violations) if o.violations else " - meets every requirement")
            for o in options[:3]
        ],
        question=question,
        unmet_criteria=[f"{u.name}: {u.detail}" for u in completion.unmet],
    )


def _rank_options(
    c: TripConstraints, world: WorldModel, on_route: list, declined: set[str]
) -> list[_Option]:
    """Flights a person could still choose, fewest compromises first. A flight whose
    payment the approver declined is shown with that compromise, never as compliant."""
    options: list[_Option] = []
    for cand in on_route:
        quote = world.quotes.get(cand.key)
        if quote is not None and not quote.available:
            continue
        price = quote.price if quote else cand.listed_price
        refundable = quote.refundable if quote else None
        failures = world.quote_failures.get(cand.key, 0)
        if quote:
            source = "live"
        elif failures:
            source = f"listed; live fare check failed {failures} time(s)"
        else:
            source = "listed, not verified"
        violations = c.check(
            origin=cand.origin,
            destination=cand.destination,
            date=cand.date,
            depart=cand.depart,
            price=price,
            refundable=refundable,
        )
        if quote is None and failures:
            violations.append(
                Violation(field="live_fare", expected="a live fare", actual="check failed")
            )
        if cand.key in declined:
            violations.append(
                Violation(field="approval", expected="approved payment", actual="declined")
            )
        options.append(
            _Option(
                flight_no=cand.flight_no,
                label=f"{cand.flight_no} {cand.depart:%H:%M} at {format_vnd(price)} ({source})",
                violations=[str(v) for v in violations],
                fields=frozenset(v.field for v in violations),
                price=price,
            )
        )
    return sorted(options, key=lambda o: (len(o.fields), o.price))


def _one_per_compromise(options: list[_Option]) -> list[_Option]:
    """One option per kind of compromise ("later flight", "over budget"...), the cheapest
    of each kind, fewest compromises first, so the person sees real alternatives."""
    best: dict[frozenset[str], _Option] = {}
    for option in sorted(options, key=lambda o: o.price):
        best.setdefault(option.fields, option)
    return sorted(best.values(), key=lambda o: (len(o.fields), o.price))
