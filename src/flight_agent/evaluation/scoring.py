"""Score one run against the oracle's expectation."""

from __future__ import annotations

from pydantic import BaseModel

from flight_agent.backend import MockAirline
from flight_agent.domain import BookingState
from flight_agent.evaluation.oracle import Expectation
from flight_agent.harness.result import RunMetrics, RunResult


class RunRecord(BaseModel):
    scenario: str
    pattern: str
    trial: int
    model: str
    success: bool
    """The run ended in an outcome the oracle accepts."""
    verified_booking: bool
    """The harness completion criteria passed (a booking was made and verified)."""
    expected_outcome: str
    booked_flight: str | None = None
    booked_price: int | None = None
    optimal_flight: str | None = None
    optimal_price: int | None = None
    price_regret: int | None = None
    """Booked price minus the cheapest acceptable price (successful bookings only)."""
    stop_reason: str
    stop_detail: str
    paid_bookings: int = 0
    leaked_holds: int = 0
    """Seats still on hold (unpaid) when the run ended."""
    ungrounded_claims: int | None = None
    summary: str = ""
    """User-facing outcome: the verified booking, or the handoff question."""
    final_message: str | None = None
    handoff: str | None = None
    """The handoff (markdown) when the run ended without a verified booking."""
    metrics: RunMetrics
    trace_file: str | None = None
    attempts: int = 1
    """How many times the run was started (infrastructure failures are retried)."""

    @property
    def infra_error(self) -> bool:
        return self.stop_reason == "infra_error"


def score_run(
    *,
    scenario_id: str,
    pattern: str,
    trial: int,
    model: str,
    expectation: Expectation,
    result: RunResult,
    backend: MockAirline,
    trace_file: str | None = None,
    attempts: int = 1,
) -> RunRecord:
    bookings = backend.all_bookings()
    paid = [b for b in bookings if b.paid]
    holds = [b for b in bookings if b.state is BookingState.HELD]
    booked = result.booking

    if expectation.outcome == "booked":
        success = bool(
            result.success and booked is not None and booked.flight_key in expectation.acceptable
        )
    else:
        success = not paid

    regret = None
    if success and booked is not None and expectation.optimal_price is not None:
        regret = booked.price - expectation.optimal_price

    return RunRecord(
        scenario=scenario_id,
        pattern=pattern,
        trial=trial,
        model=model,
        success=success,
        verified_booking=result.success,
        expected_outcome=expectation.outcome,
        booked_flight=booked.flight_key if booked else None,
        booked_price=booked.price if booked else None,
        optimal_flight=expectation.optimal_key,
        optimal_price=expectation.optimal_price,
        price_regret=regret,
        stop_reason=result.stop_reason.value,
        stop_detail=result.stop_detail,
        paid_bookings=len(paid),
        leaked_holds=len(holds),
        ungrounded_claims=len(result.grounding.ungrounded) if result.grounding else None,
        summary=result.summary,
        final_message=result.final_message,
        handoff=result.handoff.to_markdown() if result.handoff else None,
        metrics=result.metrics,
        trace_file=trace_file,
        attempts=attempts,
    )
