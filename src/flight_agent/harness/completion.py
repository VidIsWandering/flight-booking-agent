"""Harness layer: completion criteria checked by code.

"Done" is decided by reading the booking back from the backend and checking it,
never by the model saying so. The criteria use four kinds of verifiable checks:

* ``predicate``   - a boolean rule on the read-back record (state, price, date...);
* ``schema``      - the booking observation the agent received parses as a typed model;
* ``cross_check`` - independent sources agree (ledger charge == booking price == live quote);
* ``human``       - a required human approval was actually obtained.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ValidationError

from flight_agent.backend import MockAirline
from flight_agent.contracts import load
from flight_agent.domain import Booking, BookingState, format_vnd
from flight_agent.harness.constraints import TripConstraints
from flight_agent.harness.permissions import PermissionPolicy
from flight_agent.harness.world import WorldModel


class BookingView(BaseModel):
    """Schema of the booking object that tools return to the agent."""

    booking_code: str
    state: Literal["held", "confirmed", "cancelled"]
    paid: bool
    flight_no: str
    date: dt.date
    depart: str
    origin: str
    destination: str
    passenger: str
    price: int
    currency: str
    refundable: bool


class Criterion(BaseModel):
    name: str
    kind: Literal["predicate", "schema", "cross_check", "human"]
    passed: bool
    detail: str


class CompletionReport(BaseModel):
    done: bool
    booking: Booking | None = None
    criteria: list[Criterion]

    @property
    def unmet(self) -> list[Criterion]:
        return [c for c in self.criteria if not c.passed]


def verify_completion(
    *,
    constraints: TripConstraints,
    policy: PermissionPolicy,
    backend: MockAirline,
    world: WorldModel,
) -> CompletionReport:
    bookings = backend.all_bookings()
    confirmed = [b for b in bookings if b.state is BookingState.CONFIRMED]
    holds = [b for b in bookings if b.state is BookingState.HELD]

    if not confirmed:
        detail = "no confirmed booking"
        if holds:
            detail += " (held but unpaid: " + ", ".join(b.code for b in holds) + ")"
        return CompletionReport(
            done=False,
            criteria=[
                Criterion(name="confirmed_booking", kind="predicate", passed=False, detail=detail)
            ],
        )

    booking = confirmed[-1]
    criteria = [
        Criterion(
            name="confirmed_booking",
            kind="predicate",
            passed=len(confirmed) == 1,
            detail=(
                f"{booking.code} is confirmed"
                if len(confirmed) == 1
                else "more than one confirmed booking: " + ", ".join(b.code for b in confirmed)
            ),
        ),
        Criterion(
            name="paid",
            kind="predicate",
            passed=booking.paid,
            detail="paid" if booking.paid else "not paid",
        ),
    ]

    violations = constraints.check_booking(booking)
    criteria.append(
        Criterion(
            name="meets_requirements",
            kind="predicate",
            passed=not violations,
            detail="; ".join(str(v) for v in violations)
            or "route, date, time window, price, refund policy and passenger all satisfied",
        )
    )

    criteria.append(_schema_criterion(booking.code, world))

    charged = sum(
        e.amount for e in backend.ledger if e.booking_code == booking.code and e.action == "pay"
    )
    quote = world.quotes.get(booking.flight_key)
    consistent = charged == booking.price and (quote is None or quote.price == booking.price)
    quote_text = format_vnd(quote.price) if quote else "not seen"
    criteria.append(
        Criterion(
            name="amounts_agree",
            kind="cross_check",
            passed=consistent,
            detail=(
                f"charged {format_vnd(charged)}, booking {format_vnd(booking.price)}, "
                f"live quote {quote_text}"
            ),
        )
    )

    needs_approval = booking.price > policy.auto_approve_limit or (
        policy.nonrefundable_requires_approval and not booking.refundable
    )
    record = world.approval_for(booking.code)
    approver = record.decision.approver if record and record.decision.approved else None
    criteria.append(
        Criterion(
            name="approval",
            kind="human",
            passed=(not needs_approval) or approver is not None,
            detail=(
                "not required"
                if not needs_approval
                else (f"approved by {approver}" if approver else "required but missing")
            ),
        )
    )

    criteria.append(
        Criterion(
            name="no_leftover_holds",
            kind="predicate",
            passed=not holds,
            detail="no other active holds"
            if not holds
            else "active holds left: " + ", ".join(b.code for b in holds),
        )
    )
    return CompletionReport(
        done=all(c.passed for c in criteria), booking=booking, criteria=criteria
    )


def _schema_criterion(code: str, world: WorldModel) -> Criterion:
    for raw in reversed(world.observations):
        payload = load(raw)
        if not payload or not isinstance(payload.get("booking"), dict):
            continue
        if payload["booking"].get("booking_code") != code:
            continue
        try:
            BookingView.model_validate(payload["booking"])
        except ValidationError as error:
            return Criterion(
                name="booking_schema",
                kind="schema",
                passed=False,
                detail=str(error).splitlines()[0],
            )
        return Criterion(
            name="booking_schema",
            kind="schema",
            passed=True,
            detail="booking observation parses as BookingView",
        )
    return Criterion(
        name="booking_schema",
        kind="schema",
        passed=False,
        detail=f"the agent never observed booking {code}",
    )
