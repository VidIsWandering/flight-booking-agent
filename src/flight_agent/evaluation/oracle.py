"""Ground truth: derive the correct outcome of a scenario from its world data.

The oracle reads the unperturbed truth of the world (live fares, availability,
timeouts) and the simulated human's policy, and lists every flight that a perfect
agent could legitimately end up booking. No expected answer is written by hand.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from flight_agent.evaluation.scenarios import Scenario


class Expectation(BaseModel):
    outcome: Literal["booked", "no_booking"]
    acceptable: dict[str, int]
    """Flight key -> live price, for every flight a correct run may book."""
    optimal_key: str | None = None
    optimal_price: int | None = None


def derive_expectation(scenario: Scenario) -> Expectation:
    backend = scenario.build_backend()
    c, policy, human = scenario.constraints, scenario.policy, scenario.human
    acceptable: dict[str, tuple[int, str]] = {}
    for flight in backend.flights():
        fare = backend.live_fare(flight.key)
        if fare is None or fare.seats <= 0:
            continue
        if policy.require_live_quote and backend.fare_always_times_out(flight.key):
            continue  # its fare can never be checked, so it can never be held
        violations = c.check(
            origin=flight.origin,
            destination=flight.destination,
            date=flight.date,
            depart=flight.depart,
            price=fare.price,
            refundable=fare.refundable,
        )
        if violations:
            continue
        needs_approval = fare.price > policy.auto_approve_limit or (
            policy.nonrefundable_requires_approval and not fare.refundable
        )
        if needs_approval and not human.payment_verdict(fare.price, fare.refundable)[0]:
            continue
        acceptable[flight.key] = (fare.price, flight.depart.isoformat())

    if not acceptable:
        expectation = Expectation(outcome="no_booking", acceptable={})
    else:
        rank = (
            (lambda k: (acceptable[k][0], acceptable[k][1]))
            if c.preference == "cheapest"
            else (lambda k: (acceptable[k][1], acceptable[k][0]))
        )
        best = min(acceptable, key=rank)
        expectation = Expectation(
            outcome="booked",
            acceptable={k: v[0] for k, v in acceptable.items()},
            optimal_key=best,
            optimal_price=acceptable[best][0],
        )
    if scenario.expected_outcome and scenario.expected_outcome != expectation.outcome:
        raise ValueError(
            f"scenario {scenario.id}: declared outcome {scenario.expected_outcome!r} but the "
            f"world implies {expectation.outcome!r}"
        )
    return expectation
