"""Agent-facing tools (mockups) over the simulated airline backend.

Each tool returns a JSON observation with an explicit ``status`` (see
:mod:`flight_agent.contracts`). Tools never raise: backend errors become
``not_found`` / ``invalid_param`` / ``rejected`` / ``error`` observations that tell
the model what went wrong and what to try instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from flight_agent.backend import (
    BackendError,
    InvalidParam,
    MockAirline,
    NotFound,
    Rejected,
    ServiceError,
)
from flight_agent.contracts import Status, dump, envelope
from flight_agent.domain import CURRENCY, Booking

READ_ONLY_TOOLS = frozenset(
    {"search_flights", "check_seat", "get_booking", "list_airports", "get_airline_reviews"}
)


class SearchFlightsArgs(BaseModel):
    origin: str = Field(description="3-letter IATA code of the departure airport, e.g. SGN.")
    destination: str = Field(description="3-letter IATA code of the arrival airport, e.g. DAD.")
    date: str = Field(description="Departure date as YYYY-MM-DD.")


class FlightArgs(BaseModel):
    flight_no: str = Field(description="Flight number from search_flights, e.g. VN122.")
    date: str = Field(description="Departure date as YYYY-MM-DD.")


class BookSeatArgs(BaseModel):
    flight_no: str = Field(description="Flight number from search_flights, e.g. VN122.")
    date: str = Field(description="Departure date as YYYY-MM-DD.")
    passenger_name: str = Field(description="Full name of the passenger.")


class PayBookingArgs(BaseModel):
    booking_code: str = Field(description="Booking code returned by book_seat.")
    payment_method: str = Field(description="Payment method, e.g. corporate_card.")


class BookingCodeArgs(BaseModel):
    booking_code: str = Field(description="Booking code returned by book_seat.")


class AirlineArgs(BaseModel):
    airline: str = Field(description="2-letter airline code, e.g. VN.")


class NoArgs(BaseModel):
    pass


def booking_view(booking: Booking) -> dict[str, Any]:
    """The JSON projection of a booking that the agent sees."""
    return {
        "booking_code": booking.code,
        "state": booking.state.value,
        "paid": booking.paid,
        "flight_no": booking.flight_no,
        "date": booking.date.isoformat(),
        "depart": booking.depart.strftime("%H:%M"),
        "origin": booking.origin,
        "destination": booking.destination,
        "passenger": booking.passenger,
        "price": booking.price,
        "currency": CURRENCY,
        "refundable": booking.refundable,
    }


def error_observation(error: BackendError) -> dict[str, Any]:
    """Map a typed backend error to an explicit observation."""
    if isinstance(error, InvalidParam):
        return envelope(
            Status.INVALID_PARAM,
            param=error.param,
            message=error.message,
            allowed=error.allowed,
            expected=error.expected,
            hint=error.hint,
        )
    if isinstance(error, NotFound):
        return envelope(Status.NOT_FOUND, message=error.message, hint=error.hint)
    if isinstance(error, Rejected):
        return envelope(Status.REJECTED, code=error.code, message=error.message, hint=error.hint)
    if isinstance(error, ServiceError):
        return envelope(
            Status.ERROR,
            code=error.code,
            retryable=error.retryable,
            message=error.message,
            hint=error.hint,
        )
    return envelope(Status.ERROR, code=error.code, retryable=False, message=error.message)


def _guarded(fn: Callable[..., dict[str, Any]]) -> Callable[..., str]:
    """Turn a backend call into a tool function that always returns a JSON observation."""

    def run(**kwargs: Any) -> str:
        try:
            return dump(fn(**kwargs))
        except BackendError as error:
            return dump(error_observation(error))
        except Exception as error:  # a tool must never crash the agent loop
            return dump(
                envelope(Status.ERROR, code="internal", retryable=False, message=str(error))
            )

    return run


def build_tools(backend: MockAirline) -> list[BaseTool]:
    """Create the tool set bound to one backend instance (one run = one world)."""

    def search_flights(origin: str, destination: str, date: str) -> dict[str, Any]:
        flights, promotions = backend.search(origin, destination, date)
        payload = envelope(
            Status.OK,
            query={
                "origin": origin.strip().upper(),
                "destination": destination.strip().upper(),
                "date": date.strip(),
            },
            count=len(flights),
            flights=[
                {
                    "flight_no": f.flight_no,
                    "airline": f.airline,
                    "depart": f.depart.strftime("%H:%M"),
                    "arrive": f.arrive.strftime("%H:%M"),
                    "listed_price": f.listed_price,
                }
                for f in flights
            ],
            note=(
                "listed_price comes from a cache and may be stale; "
                "call check_seat for the live fare and availability before booking."
            ),
        )
        if promotions:
            payload["promotions"] = [
                {"flight_no": p.flight_no, "price": p.price, "note": p.note} for p in promotions
            ]
        return payload

    def check_seat(flight_no: str, date: str) -> dict[str, Any]:
        flight, fare = backend.quote(flight_no, date)
        return envelope(
            Status.OK,
            flight_no=flight.flight_no,
            date=flight.date.isoformat(),
            depart=flight.depart.strftime("%H:%M"),
            arrive=flight.arrive.strftime("%H:%M"),
            available=fare.seats > 0,
            seats_left=fare.seats,
            price=fare.price,
            currency=CURRENCY,
            refundable=fare.refundable,
            fare_class=fare.fare_class,
        )

    def book_seat(flight_no: str, date: str, passenger_name: str) -> dict[str, Any]:
        booking = backend.hold(flight_no, date, passenger_name)
        return envelope(
            Status.OK,
            booking=booking_view(booking),
            note="Seat held (not ticketed). pay_booking confirms it; cancel_booking releases it.",
        )

    def pay_booking(booking_code: str, payment_method: str) -> dict[str, Any]:
        booking = backend.pay(booking_code, payment_method)
        return envelope(Status.OK, booking=booking_view(booking), amount_charged=booking.price)

    def get_booking(booking_code: str) -> dict[str, Any]:
        return envelope(Status.OK, booking=booking_view(backend.get(booking_code)))

    def cancel_booking(booking_code: str) -> dict[str, Any]:
        booking = backend.cancel(booking_code)
        return envelope(
            Status.OK,
            booking=booking_view(booking),
            refunded=booking.refunded,
        )

    def list_airports() -> dict[str, Any]:
        airports = backend.list_airports()
        return envelope(
            Status.OK,
            count=len(airports),
            airports=[{"code": code, "city": city} for code, city in airports.items()],
        )

    def get_airline_reviews(airline: str) -> dict[str, Any]:
        code, rating = backend.airline_rating(airline)
        return envelope(Status.OK, airline=code, rating=rating, scale=5)

    specs: list[tuple[str, Callable[..., dict[str, Any]], type[BaseModel], str]] = [
        (
            "search_flights",
            search_flights,
            SearchFlightsArgs,
            "Search scheduled flights for a route and date. Returns flight numbers, times and "
            "listed prices. Listed prices come from a cache and may be stale.",
        ),
        (
            "check_seat",
            check_seat,
            FlightArgs,
            "Get the live fare, seat availability and refund policy of one flight. Read-only.",
        ),
        (
            "book_seat",
            book_seat,
            BookSeatArgs,
            "Hold one seat on a flight for a passenger at the current live fare. Creates a "
            "booking in state 'held' that is free to cancel. Side effect.",
        ),
        (
            "pay_booking",
            pay_booking,
            PayBookingArgs,
            "Pay a held booking, which confirms and tickets it. Irreversible. Side effect.",
        ),
        (
            "get_booking",
            get_booking,
            BookingCodeArgs,
            "Read the current state of a booking. Read-only.",
        ),
        (
            "cancel_booking",
            cancel_booking,
            BookingCodeArgs,
            "Cancel a booking and release its seat. Paid refundable bookings are refunded; "
            "paid non-refundable bookings are not. Side effect.",
        ),
        (
            "list_airports",
            list_airports,
            NoArgs,
            "List supported airport codes and their cities. Read-only.",
        ),
        (
            "get_airline_reviews",
            get_airline_reviews,
            AirlineArgs,
            "Get the average customer rating of an airline. Read-only.",
        ),
    ]
    return [
        StructuredTool.from_function(
            func=_guarded(fn), name=name, description=description, args_schema=schema
        )
        for name, fn, schema, description in specs
    ]
