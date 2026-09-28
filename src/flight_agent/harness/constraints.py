"""Harness layer: constraints are data.

The user's requirements live in a typed, immutable object instead of only in the
prompt. The same object

* renders the natural-language request and the pinned requirements block that is
  re-sent with every model call, so requirements never drift out of context;
* is evaluated by code before any commitment (``book_seat`` / ``pay_booking``);
* is evaluated by code in the completion criteria and the progress metric.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from flight_agent.domain import Booking, format_vnd


class Violation(BaseModel):
    """One requirement that a flight, fare or booking does not satisfy."""

    field: str
    expected: str
    actual: str

    def __str__(self) -> str:
        return f"{self.field}: expected {self.expected}, got {self.actual}"


class TripConstraints(BaseModel):
    """Hard requirements of a one-way booking request."""

    model_config = ConfigDict(frozen=True)

    origin: str
    destination: str
    depart_date: dt.date
    depart_after: dt.time | None = None
    """Earliest allowed departure (inclusive)."""
    depart_before: dt.time | None = None
    """Latest allowed departure (exclusive)."""
    max_price: int
    passenger: str
    refundable_only: bool = False
    payment_method: str = "corporate_card"
    preference: Literal["cheapest", "earliest"] = "cheapest"

    @field_validator("origin", "destination")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    # ---------------------------------------------------------------- rendering
    def window_text(self) -> str:
        if self.depart_after and self.depart_before:
            return f"departing between {self.depart_after:%H:%M} and {self.depart_before:%H:%M}"
        if self.depart_before:
            return f"departing before {self.depart_before:%H:%M}"
        if self.depart_after:
            return f"departing at or after {self.depart_after:%H:%M}"
        return "any departure time"

    def request_text(self) -> str:
        """The natural-language request given to the agent, rendered from the data."""
        refund = " The fare must be refundable." if self.refundable_only else ""
        pref = (
            "Choose the cheapest option that satisfies everything."
            if self.preference == "cheapest"
            else "Choose the earliest departure that satisfies everything."
        )
        return (
            f"Book a one-way economy ticket from {self.origin} to {self.destination} on "
            f"{self.depart_date.isoformat()} for {self.passenger}, {self.window_text()}, "
            f"with a total price of at most {format_vnd(self.max_price)}.{refund} {pref} "
            f"Pay with {self.payment_method}."
        )

    def pinned_block(self) -> str:
        """Authoritative requirements, re-sent with every model call."""
        lines = [
            "TRIP REQUIREMENTS (authoritative; pinned by the harness and checked by code):",
            f"- Route: {self.origin} -> {self.destination}, one-way, passenger: {self.passenger}",
            f"- Date: {self.depart_date.isoformat()}, {self.window_text()}",
            f"- Total price: at most {format_vnd(self.max_price)}",
            f"- Refundable fare required: {'yes' if self.refundable_only else 'no'}",
            f"- Preference: {self.preference} option that satisfies every requirement",
            f"- Payment method: {self.payment_method}",
        ]
        return "\n".join(lines)

    # --------------------------------------------------------------- evaluation
    def in_window(self, depart: dt.time) -> bool:
        if self.depart_after and depart < self.depart_after:
            return False
        return not (self.depart_before and depart >= self.depart_before)

    def check(
        self,
        *,
        origin: str,
        destination: str,
        date: dt.date,
        depart: dt.time,
        price: int | None = None,
        refundable: bool | None = None,
    ) -> list[Violation]:
        """Evaluate a departure (and optionally its fare) against the requirements.

        Unknown fields (``None``) are not judged; callers decide whether unknown is
        acceptable (the permission gate, for instance, requires a known live fare).
        """
        violations: list[Violation] = []
        if (origin.upper(), destination.upper()) != (self.origin, self.destination):
            violations.append(
                Violation(
                    field="route",
                    expected=f"{self.origin}->{self.destination}",
                    actual=f"{origin.upper()}->{destination.upper()}",
                )
            )
        if date != self.depart_date:
            violations.append(
                Violation(field="date", expected=self.depart_date.isoformat(), actual=str(date))
            )
        if not self.in_window(depart):
            violations.append(
                Violation(
                    field="departure_time", expected=self.window_text(), actual=f"{depart:%H:%M}"
                )
            )
        if price is not None and price > self.max_price:
            violations.append(
                Violation(
                    field="price",
                    expected=f"<= {format_vnd(self.max_price)}",
                    actual=format_vnd(price),
                )
            )
        if self.refundable_only and refundable is False:
            violations.append(
                Violation(field="refundable", expected="refundable fare", actual="non-refundable")
            )
        return violations

    def check_booking(self, booking: Booking) -> list[Violation]:
        violations = self.check(
            origin=booking.origin,
            destination=booking.destination,
            date=booking.date,
            depart=booking.depart,
            price=booking.price,
            refundable=booking.refundable,
        )
        if _normalize_name(booking.passenger) != _normalize_name(self.passenger):
            violations.append(
                Violation(field="passenger", expected=self.passenger, actual=booking.passenger)
            )
        return violations

    @property
    def dimensions(self) -> int:
        """Number of requirements a fare is judged on (used by the progress metric)."""
        return 5 if self.refundable_only else 4

    def satisfied(self, violations: list[Violation]) -> int:
        judged = {"route", "date", "departure_time", "price"} | (
            {"refundable"} if self.refundable_only else set()
        )
        failed = {v.field for v in violations if v.field in judged}
        return len(judged) - len(failed)


def _normalize_name(name: str) -> str:
    return " ".join(name.split()).casefold()


def names_match(a: str, b: str) -> bool:
    return _normalize_name(a) == _normalize_name(b)
