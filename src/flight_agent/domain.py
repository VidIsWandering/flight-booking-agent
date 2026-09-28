"""Domain entities of the simulated airline world.

These types describe *the world*, not the agent: flights on a timetable, their
live fares, and bookings. The harness and the evaluation read them; the agent only
ever sees their JSON projection through tools.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

CURRENCY = "VND"


def format_vnd(amount: int) -> str:
    """Render an amount the way tools and reports display it: ``1,390,000 VND``."""
    return f"{amount:,} {CURRENCY}"


def parse_hhmm(value: str | dt.time) -> dt.time:
    """Parse ``"HH:MM"`` into a :class:`datetime.time` (idempotent for ``time`` inputs)."""
    if isinstance(value, dt.time):
        return value
    hours, minutes = value.strip().split(":")
    return dt.time(int(hours), int(minutes))


def flight_key(flight_no: str, date: dt.date | str) -> str:
    """Stable identifier of one departure, e.g. ``VN122@2026-10-07``."""
    return f"{flight_no.strip().upper()}@{date}"


class Flight(BaseModel):
    """A scheduled departure as published on the timetable."""

    model_config = ConfigDict(frozen=True)

    flight_no: str
    airline: str
    origin: str
    destination: str
    date: dt.date
    depart: dt.time
    arrive: dt.time
    listed_price: int = Field(description="Price shown by search; may be stale.")

    @property
    def key(self) -> str:
        return flight_key(self.flight_no, self.date)


class Fare(BaseModel):
    """Live commercial state of a departure (what ``check_seat`` reveals)."""

    price: int
    seats: int
    refundable: bool
    fare_class: str = "Economy"


class BookingState(str, Enum):
    HELD = "held"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class Booking(BaseModel):
    """A booking record as stored by the airline backend."""

    code: str
    flight_no: str
    date: dt.date
    origin: str
    destination: str
    depart: dt.time
    passenger: str
    price: int
    refundable: bool
    state: BookingState
    paid: bool = False
    payment_method: str | None = None
    refunded: bool = False

    @property
    def flight_key(self) -> str:
        return flight_key(self.flight_no, self.date)
