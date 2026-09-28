"""In-memory airline backend with scenario-driven perturbations.

The backend models the parts of a real booking system that make agents fail:

* search results are a cache, so listed prices and availability can be stale;
* the live fare service can time out;
* holding a seat, paying and cancelling are side effects recorded in a ledger.

It exposes two surfaces:

* the **agent-facing API** (``search`` ... ``cancel``), which the tools wrap and
  which raises typed :class:`BackendError` subclasses;
* a **read-only harness API** (``lookup_flight``, ``read_booking`` ...), which is
  never perturbed and is what the harness uses to verify outcomes independently of
  anything the model says.
"""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from flight_agent.domain import Booking, BookingState, Fare, Flight, flight_key

BOOKING_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class BackendError(Exception):
    """Base class for errors raised by the agent-facing API."""

    code = "error"
    retryable = False

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class NotFound(BackendError):
    code = "not_found"


class InvalidParam(BackendError):
    code = "invalid_param"

    def __init__(
        self,
        message: str,
        *,
        param: str,
        allowed: list[str] | None = None,
        expected: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.param = param
        self.allowed = allowed
        self.expected = expected


class Rejected(BackendError):
    """A business rule refused the request (sold out, already paid...)."""

    def __init__(self, message: str, *, code: str, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.code = code


class ServiceError(BackendError):
    """A technical failure of a downstream service."""

    def __init__(
        self, message: str, *, code: str, retryable: bool, hint: str | None = None
    ) -> None:
        super().__init__(message, hint=hint)
        self.code = code
        self.retryable = retryable


class Promotion(BaseModel):
    """Third-party promotional content attached to search results (untrusted text)."""

    origin: str
    destination: str
    date: dt.date
    flight_no: str
    price: int
    note: str


class LedgerEntry(BaseModel):
    """One side effect performed on the backend."""

    seq: int
    action: Literal["hold", "pay", "cancel", "refund"]
    booking_code: str
    flight_key: str
    amount: int


class MockAirline:
    """A deterministic airline backend for one run."""

    def __init__(
        self,
        *,
        flights: Iterable[Flight],
        fares: dict[str, Fare],
        airports: dict[str, str],
        payment_methods: Iterable[str],
        today: dt.date,
        fare_timeouts: dict[str, int] | None = None,
        promotions: Iterable[Promotion] = (),
        airline_ratings: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        self._flights: dict[str, Flight] = {f.key: f for f in flights}
        missing = set(self._flights) - set(fares)
        if missing:
            raise ValueError(f"missing live fares for: {sorted(missing)}")
        self._fares = {key: fare.model_copy() for key, fare in fares.items()}
        self._airports = dict(airports)
        self._payment_methods = list(payment_methods)
        self._today = today
        # key -> remaining timeouts before the fare service answers; -1 means "always"
        self._fare_timeouts = dict(fare_timeouts or {})
        self._promotions = list(promotions)
        self._ratings = dict(airline_ratings or {})
        self._rng = random.Random(seed)
        self._bookings: dict[str, Booking] = {}
        self.ledger: list[LedgerEntry] = []

    # ------------------------------------------------------------------ helpers
    @property
    def today(self) -> dt.date:
        return self._today

    @property
    def payment_methods(self) -> list[str]:
        return list(self._payment_methods)

    def _airport(self, code: str, param: str) -> str:
        normalized = code.strip().upper()
        if normalized not in self._airports:
            raise InvalidParam(
                f"Unknown airport code {code!r}.",
                param=param,
                allowed=sorted(self._airports),
                hint="Use a 3-letter IATA code; call list_airports for the supported list.",
            )
        return normalized

    def _date(self, value: str | dt.date, param: str = "date") -> dt.date:
        if isinstance(value, dt.date):
            parsed = value
        else:
            try:
                parsed = dt.date.fromisoformat(value.strip())
            except ValueError:
                raise InvalidParam(
                    f"Cannot parse date {value!r}.", param=param, expected="YYYY-MM-DD"
                ) from None
        if parsed < self._today:
            raise InvalidParam(
                f"Date {parsed} is in the past (today is {self._today}).",
                param=param,
                expected=f"a date on or after {self._today}",
            )
        return parsed

    def _flight(self, flight_no: str, date: str | dt.date) -> Flight:
        parsed = self._date(date)
        flight = self._flights.get(flight_key(flight_no, parsed))
        if flight is None:
            raise NotFound(
                f"No flight {flight_no.strip().upper()} on {parsed}.",
                hint="Use flight numbers returned by search_flights for that date.",
            )
        return flight

    def _booking(self, code: str) -> Booking:
        booking = self._bookings.get(code.strip().upper())
        if booking is None:
            raise NotFound(
                f"No booking with code {code!r}.",
                hint="Use the booking_code returned by book_seat.",
            )
        return booking

    def _new_code(self) -> str:
        while True:
            code = "".join(self._rng.choice(BOOKING_CODE_ALPHABET) for _ in range(6))
            if code not in self._bookings:
                return code

    def _log(self, action: str, booking: Booking, amount: int) -> None:
        self.ledger.append(
            LedgerEntry(
                seq=len(self.ledger) + 1,
                action=action,  # type: ignore[arg-type]
                booking_code=booking.code,
                flight_key=booking.flight_key,
                amount=amount,
            )
        )

    # -------------------------------------------------------- agent-facing API
    def list_airports(self) -> dict[str, str]:
        return dict(sorted(self._airports.items()))

    def search(
        self, origin: str, destination: str, date: str | dt.date
    ) -> tuple[list[Flight], list[Promotion]]:
        src = self._airport(origin, "origin")
        dst = self._airport(destination, "destination")
        if src == dst:
            raise InvalidParam("Origin and destination must differ.", param="destination")
        day = self._date(date)
        flights = sorted(
            (
                f
                for f in self._flights.values()
                if f.origin == src and f.destination == dst and f.date == day
            ),
            key=lambda f: f.depart,
        )
        promos = [
            p
            for p in self._promotions
            if p.origin == src and p.destination == dst and p.date == day
        ]
        return flights, promos

    def quote(self, flight_no: str, date: str | dt.date) -> tuple[Flight, Fare]:
        flight = self._flight(flight_no, date)
        remaining = self._fare_timeouts.get(flight.key, 0)
        if remaining != 0:
            if remaining > 0:
                self._fare_timeouts[flight.key] = remaining - 1
            raise ServiceError(
                f"Fare service timed out for {flight.flight_no}.",
                code="timeout",
                retryable=True,
                hint="The fare service may recover later; other flights can be checked.",
            )
        return flight, self._fares[flight.key].model_copy()

    def hold(self, flight_no: str, date: str | dt.date, passenger: str) -> Booking:
        flight = self._flight(flight_no, date)
        name = " ".join(passenger.split())
        if not name:
            raise InvalidParam("Passenger name is required.", param="passenger_name")
        fare = self._fares[flight.key]
        if fare.seats <= 0:
            raise Rejected(
                f"{flight.flight_no} on {flight.date} is sold out.",
                code="sold_out",
                hint="Pick another flight; availability shown by search can be stale.",
            )
        fare.seats -= 1
        booking = Booking(
            code=self._new_code(),
            flight_no=flight.flight_no,
            date=flight.date,
            origin=flight.origin,
            destination=flight.destination,
            depart=flight.depart,
            passenger=name,
            price=fare.price,
            refundable=fare.refundable,
            state=BookingState.HELD,
        )
        self._bookings[booking.code] = booking
        self._log("hold", booking, 0)
        return booking.model_copy()

    def pay(self, code: str, payment_method: str) -> Booking:
        booking = self._booking(code)
        method = payment_method.strip()
        if method not in self._payment_methods:
            raise InvalidParam(
                f"Payment method {payment_method!r} is not accepted.",
                param="payment_method",
                allowed=self.payment_methods,
            )
        if booking.state is BookingState.CANCELLED:
            raise Rejected(f"Booking {booking.code} is cancelled.", code="booking_cancelled")
        if booking.paid:
            raise Rejected(f"Booking {booking.code} is already paid.", code="already_paid")
        booking.paid = True
        booking.payment_method = method
        booking.state = BookingState.CONFIRMED
        self._log("pay", booking, booking.price)
        return booking.model_copy()

    def get(self, code: str) -> Booking:
        return self._booking(code).model_copy()

    def cancel(self, code: str) -> Booking:
        booking = self._booking(code)
        if booking.state is BookingState.CANCELLED:
            raise Rejected(
                f"Booking {booking.code} is already cancelled.", code="already_cancelled"
            )
        booking.state = BookingState.CANCELLED
        self._fares[booking.flight_key].seats += 1
        self._log("cancel", booking, 0)
        if booking.paid and booking.refundable:
            booking.refunded = True
            self._log("refund", booking, booking.price)
        return booking.model_copy()

    def airline_rating(self, airline: str) -> tuple[str, float]:
        code = airline.strip().upper()
        if code not in self._ratings:
            raise NotFound(f"No reviews for airline {airline!r}.")
        return code, self._ratings[code]

    # ------------------------------------------------ harness read-only API
    def lookup_flight(self, flight_no: str, date: str | dt.date) -> Flight | None:
        """Static timetable lookup. Never perturbed, no side effects."""
        try:
            day = date if isinstance(date, dt.date) else dt.date.fromisoformat(str(date).strip())
        except ValueError:
            return None
        return self._flights.get(flight_key(flight_no, day))

    def read_booking(self, code: str) -> Booking | None:
        """Read-back of a booking record, independent of what the model reported."""
        booking = self._bookings.get(str(code).strip().upper())
        return booking.model_copy() if booking else None

    def all_bookings(self) -> list[Booking]:
        return [b.model_copy() for b in self._bookings.values()]

    def flights(self) -> list[Flight]:
        return list(self._flights.values())

    def live_fare(self, key: str) -> Fare | None:
        """Ground-truth live fare (used by the evaluation oracle, never by agents)."""
        fare = self._fares.get(key)
        return fare.model_copy() if fare else None

    def fare_always_times_out(self, key: str) -> bool:
        return self._fare_timeouts.get(key, 0) < 0
