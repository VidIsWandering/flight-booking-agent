"""Load evaluation scenarios from YAML and turn them into runnable worlds."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from flight_agent.backend import MockAirline, Promotion
from flight_agent.domain import Fare, Flight, flight_key, format_vnd, parse_hhmm
from flight_agent.harness.approval import ApprovalDecision, ApprovalRequest
from flight_agent.harness.constraints import TripConstraints
from flight_agent.harness.permissions import PermissionPolicy

DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.yaml")


class TimetableRow(BaseModel):
    flight: str
    route: str
    date: dt.date
    depart: str
    arrive: str
    price: int
    seats: int
    refundable: bool
    fare_class: str = "Economy"

    def to_flight(self) -> Flight:
        origin, destination = self.route.split("-")
        return Flight(
            flight_no=self.flight,
            airline=self.flight[:2],
            origin=origin,
            destination=destination,
            date=self.date,
            depart=parse_hhmm(self.depart),
            arrive=parse_hhmm(self.arrive),
            listed_price=self.price,
        )

    def to_fare(self) -> Fare:
        return Fare(
            price=self.price,
            seats=self.seats,
            refundable=self.refundable,
            fare_class=self.fare_class,
        )


class FareOverride(BaseModel):
    price: int | None = None
    seats: int | None = None
    refundable: bool | None = None


class WorldSpec(BaseModel):
    today: dt.date
    airports: dict[str, str]
    payment_methods: list[str]
    airline_ratings: dict[str, float] = Field(default_factory=dict)
    timetable: list[TimetableRow]


class HumanPolicy(BaseModel):
    """How the simulated human approver answers approval requests."""

    approve_plans: bool = True
    max_payment: int = 2_000_000
    allow_nonrefundable: bool = True
    allow_cancel_paid: bool = False

    def payment_verdict(self, amount: int | None, refundable: bool | None) -> tuple[bool, str]:
        if amount is not None and amount > self.max_payment:
            return False, f"amounts above {format_vnd(self.max_payment)} are not acceptable"
        if refundable is False and not self.allow_nonrefundable:
            return False, "non-refundable fares are not acceptable"
        return True, ""


class ScenarioApprover:
    """A deterministic stand-in for the human approver, driven by :class:`HumanPolicy`."""

    def __init__(self, policy: HumanPolicy) -> None:
        self.policy = policy

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.kind == "plan":
            approved, note = self.policy.approve_plans, ""
        elif request.tool == "pay_booking":
            approved, note = self.policy.payment_verdict(request.amount, request.refundable)
        elif request.tool == "cancel_booking":
            approved, note = self.policy.allow_cancel_paid, ""
        else:
            approved, note = False, "unexpected request"
        return ApprovalDecision(approved=approved, approver="simulated-human", note=note)


class Scenario(BaseModel):
    id: str
    title: str
    description: str
    probes: list[str] = Field(default_factory=list)
    constraints: TripConstraints
    policy: PermissionPolicy
    human: HumanPolicy
    world: WorldSpec
    fares: dict[str, FareOverride] = Field(default_factory=dict)
    fare_timeouts: dict[str, int] = Field(default_factory=dict)
    promotions: list[Promotion] = Field(default_factory=list)
    expected_outcome: Literal["booked", "no_booking"] | None = None
    """Optional assertion; the oracle derives the expectation from the world anyway."""

    def build_backend(self, seed: int = 0) -> MockAirline:
        flights = [row.to_flight() for row in self.world.timetable]
        fares = {flight_key(row.flight, row.date): row.to_fare() for row in self.world.timetable}
        for key, override in self.fares.items():
            if key not in fares:
                raise ValueError(f"scenario {self.id}: fare override for unknown flight {key}")
            fares[key] = fares[key].model_copy(update=override.model_dump(exclude_none=True))
        return MockAirline(
            flights=flights,
            fares=fares,
            airports=self.world.airports,
            payment_methods=self.world.payment_methods,
            today=self.world.today,
            fare_timeouts=self.fare_timeouts,
            promotions=self.promotions,
            airline_ratings=self.world.airline_ratings,
            seed=seed,
        )

    def approver(self) -> ScenarioApprover:
        return ScenarioApprover(self.human)


def _merge(defaults: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    return {**defaults, **(override or {})}


def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    raw = yaml.safe_load(Path(path or DEFAULT_SCENARIOS).read_text(encoding="utf-8"))
    world = WorldSpec.model_validate(raw["world"])
    defaults = raw.get("defaults", {})
    scenarios = []
    for item in raw["scenarios"]:
        spec = dict(item)
        spec["constraints"] = _merge(defaults.get("constraints", {}), item.get("constraints"))
        spec["policy"] = _merge(defaults.get("policy", {}), item.get("policy"))
        spec["human"] = _merge(defaults.get("human", {}), item.get("human"))
        spec["world"] = world
        scenarios.append(Scenario.model_validate(spec))
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique")
    return scenarios


def get_scenario(scenario_id: str, path: str | Path | None = None) -> Scenario:
    for scenario in load_scenarios(path):
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario {scenario_id!r}")
