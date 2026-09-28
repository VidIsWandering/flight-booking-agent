"""Harness layer: permission checks, run *before* a tool executes.

The agent's authority is data (:class:`PermissionPolicy`). For every proposed call
the gate returns one of three decisions:

* ``ALLOW`` - execute the call;
* ``DENY``  - do not execute; tell the model which rule refused it and what to do;
* ``ASK``   - pause and ask a human approver (the "needs a human" stop condition).

Reversible actions (holding a seat) are allowed without approval as long as they
satisfy the requirements; irreversible ones (paying, cancelling a paid ticket) need
a human when they exceed the agent's limits.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from flight_agent.backend import MockAirline
from flight_agent.domain import BookingState, format_vnd
from flight_agent.harness.approval import ApprovalRequest
from flight_agent.harness.constraints import TripConstraints, Violation, names_match
from flight_agent.harness.world import WorldModel
from flight_agent.tools import READ_ONLY_TOOLS


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionPolicy(BaseModel):
    """What the agent may do on its own."""

    auto_approve_limit: int = 1_500_000
    """Payments above this amount need a human approval."""
    nonrefundable_requires_approval: bool = True
    """Paying for a non-refundable fare needs a human approval."""
    max_active_holds: int = 1
    """How many unpaid holds may exist at once."""
    require_live_quote: bool = True
    """A seat may only be held after its live fare was checked in this run."""
    allowed_payment_methods: list[str] = Field(default_factory=lambda: ["corporate_card"])
    cancel_paid_requires_approval: bool = True
    """Cancelling a paid (ticketed) booking needs a human approval."""


class GateResult(BaseModel):
    decision: Decision
    rule: str
    reason: str
    violations: list[Violation] = Field(default_factory=list)
    hint: str | None = None
    approval: ApprovalRequest | None = None


def _allow(rule: str, reason: str = "") -> GateResult:
    return GateResult(decision=Decision.ALLOW, rule=rule, reason=reason)


def _deny(
    rule: str, reason: str, *, hint: str | None = None, violations: list[Violation] | None = None
) -> GateResult:
    return GateResult(
        decision=Decision.DENY, rule=rule, reason=reason, hint=hint, violations=violations or []
    )


class PermissionGate:
    def __init__(
        self,
        *,
        policy: PermissionPolicy,
        constraints: TripConstraints,
        backend: MockAirline,
        world: WorldModel,
    ) -> None:
        self.policy = policy
        self.constraints = constraints
        self.backend = backend
        self.world = world

    def evaluate(self, tool: str, args: dict[str, Any]) -> GateResult:
        if tool in READ_ONLY_TOOLS:
            return _allow("read_only")
        if tool == "book_seat":
            return self._book(args)
        if tool == "pay_booking":
            return self._pay(args)
        if tool == "cancel_booking":
            return self._cancel(args)
        return _deny("not_allowlisted", f"Tool {tool!r} is not on the allowlist.")

    # ---------------------------------------------------------------- book_seat
    def _book(self, args: dict[str, Any]) -> GateResult:
        c = self.constraints
        if not names_match(args["passenger_name"], c.passenger):
            return _deny(
                "passenger_mismatch",
                f"The passenger must be {c.passenger!r}.",
                hint="Use the passenger name from the trip requirements.",
            )
        bookings = self.backend.all_bookings()
        confirmed = [b for b in bookings if b.state is BookingState.CONFIRMED]
        if confirmed:
            return _deny(
                "already_booked",
                f"Booking {confirmed[0].code} is already confirmed for this trip.",
                hint="Do not book again; verify it with get_booking.",
            )
        flight = self.backend.lookup_flight(args["flight_no"], args["date"])
        if flight is None:
            return _allow(
                "unknown_flight", "The backend will report that the flight does not exist."
            )
        violations = c.check(
            origin=flight.origin,
            destination=flight.destination,
            date=flight.date,
            depart=flight.depart,
        )
        if violations:
            return _deny(
                "constraint_violation",
                f"{flight.flight_no} on {flight.date} does not satisfy the trip requirements.",
                violations=violations,
                hint="Only book flights that satisfy every pinned requirement.",
            )
        if self.policy.require_live_quote:
            quote = self.world.quotes.get(flight.key)
            if quote is None:
                return _deny(
                    "live_fare_required",
                    f"No live fare was checked for {flight.flight_no} on {flight.date}.",
                    hint="Call check_seat for this flight first; listed prices can be stale.",
                )
            violations = c.check(
                origin=flight.origin,
                destination=flight.destination,
                date=flight.date,
                depart=flight.depart,
                price=quote.price,
                refundable=quote.refundable,
            )
            if violations:
                return _deny(
                    "constraint_violation",
                    f"The live fare of {flight.flight_no} does not satisfy the trip requirements.",
                    violations=violations,
                    hint="Choose another flight whose live fare satisfies every requirement.",
                )
        holds = [b for b in bookings if b.state is BookingState.HELD]
        if len(holds) >= self.policy.max_active_holds:
            codes = ", ".join(b.code for b in holds)
            return _deny(
                "one_active_hold",
                f"Booking {codes} is already on hold.",
                hint="Pay the existing hold, or cancel it with cancel_booking before holding "
                "another seat.",
            )
        return _allow("within_authority")

    # -------------------------------------------------------------- pay_booking
    def _pay(self, args: dict[str, Any]) -> GateResult:
        booking = self.backend.read_booking(args["booking_code"])
        if booking is None:
            return _allow(
                "unknown_booking", "The backend will report that the booking does not exist."
            )
        method = args["payment_method"].strip()
        if method not in self.policy.allowed_payment_methods:
            return _deny(
                "payment_method",
                f"Payment method {method!r} is not authorised for this agent.",
                hint=f"Use one of: {', '.join(self.policy.allowed_payment_methods)}.",
            )
        violations = self.constraints.check_booking(booking)
        if violations:
            return _deny(
                "constraint_violation",
                f"Booking {booking.code} does not satisfy the trip requirements.",
                violations=violations,
                hint="Cancel this hold and choose a compliant flight.",
            )
        if booking.state is not BookingState.HELD:
            return _allow("not_held", "The backend will report the booking state.")

        reasons: list[str] = []
        if booking.price > self.policy.auto_approve_limit:
            reasons.append(
                f"the amount {format_vnd(booking.price)} exceeds the auto-approval limit "
                f"{format_vnd(self.policy.auto_approve_limit)}"
            )
        if self.policy.nonrefundable_requires_approval and not booking.refundable:
            reasons.append("the fare is non-refundable")
        if not reasons:
            return _allow("within_authority")

        prior = self.world.approval_for(booking.code)
        if prior is not None:
            if not prior.decision.approved:
                note = f" ({prior.decision.note})" if prior.decision.note else ""
                return _deny(
                    "human_declined",
                    f"The approver declined paying for {booking.code}{note}.",
                    hint="Do not retry this payment. Cancel the hold and choose another option, "
                    "or stop and explain.",
                )
            if prior.request.amount == booking.price:
                return _allow("previously_approved")

        refund = "refundable" if booking.refundable else "non-refundable"
        return GateResult(
            decision=Decision.ASK,
            rule="approval_required",
            reason="; ".join(reasons),
            approval=ApprovalRequest(
                kind="action",
                where=self._where(),
                action=(
                    f"pay_booking({booking.code}) for {booking.flight_no} on {booking.date} "
                    f"{booking.depart:%H:%M}, {format_vnd(booking.price)}, {refund}"
                ),
                why=" and ".join(reasons).capitalize() + ".",
                tool="pay_booking",
                args=dict(args),
                amount=booking.price,
                refundable=booking.refundable,
                booking_code=booking.code,
            ),
        )

    # ----------------------------------------------------------- cancel_booking
    def _cancel(self, args: dict[str, Any]) -> GateResult:
        booking = self.backend.read_booking(args["booking_code"])
        if booking is None or not booking.paid or booking.state is BookingState.CANCELLED:
            return _allow("reversible")
        if not self.policy.cancel_paid_requires_approval:
            return _allow("within_authority")
        consequence = "a refund is issued" if booking.refundable else "no refund (non-refundable)"
        return GateResult(
            decision=Decision.ASK,
            rule="approval_required",
            reason="cancelling a paid ticket",
            approval=ApprovalRequest(
                kind="action",
                where=self._where(),
                action=(
                    f"cancel_booking({booking.code}) for paid ticket {booking.flight_no}; "
                    f"{consequence}"
                ),
                why="Cancelling a ticketed booking is outside the agent's authority.",
                tool="cancel_booking",
                args=dict(args),
                amount=booking.price,
                refundable=booking.refundable,
                booking_code=booking.code,
            ),
        )

    def _where(self) -> str:
        quotes = [q for q in self.world.quotes.values() if q.date == self.constraints.depart_date]
        holds = [b for b in self.backend.all_bookings() if b.state is BookingState.HELD]
        parts = [f"{len(quotes)} live fare(s) checked"]
        if holds:
            parts.append(
                "holding "
                + ", ".join(
                    f"{b.code} ({b.flight_no} {b.depart:%H:%M}, {format_vnd(b.price)})"
                    for b in holds
                )
            )
        return "; ".join(parts)
