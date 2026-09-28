"""The harness's record of what the agent has observed.

This is *not* ground truth: it only contains what tool observations revealed during
the run. The permission gate uses it (e.g. "was a live fare seen before booking?"),
the stall detector derives progress from it, and the handoff reports from it.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any

from pydantic import BaseModel

from flight_agent.contracts import Status, status_of
from flight_agent.domain import flight_key, format_vnd, parse_hhmm
from flight_agent.harness.approval import ApprovalDecision, ApprovalRequest


class Candidate(BaseModel):
    """A flight listed by ``search_flights`` (listed price may be stale)."""

    key: str
    flight_no: str
    airline: str
    origin: str
    destination: str
    date: dt.date
    depart: dt.time
    arrive: dt.time
    listed_price: int


class Quote(BaseModel):
    """A live fare observed through ``check_seat``."""

    key: str
    flight_no: str
    date: dt.date
    depart: dt.time
    price: int
    available: bool
    seats_left: int
    refundable: bool
    step: int


class ActionRecord(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]
    status: str
    summary: str
    executed: bool


class ApprovalRecord(BaseModel):
    step: int
    request: ApprovalRequest
    decision: ApprovalDecision


class WorldModel:
    def __init__(self) -> None:
        self.actions: list[ActionRecord] = []
        self.candidates: dict[str, Candidate] = {}
        self.quotes: dict[str, Quote] = {}
        self.quote_failures: Counter[str] = Counter()
        self.searches: list[dict[str, Any]] = []
        self.approvals: list[ApprovalRecord] = []
        self.denials: list[tuple[int, str, str]] = []  # (step, action, reason)
        self.observations: list[str] = []

    # ------------------------------------------------------------------ record
    def record(
        self,
        *,
        step: int,
        tool: str,
        args: dict[str, Any],
        payload: dict[str, Any],
        raw: str,
        executed: bool,
    ) -> ActionRecord:
        status = status_of(payload)
        if executed:
            self.observations.append(raw)
            if status is Status.OK:
                self._absorb(step, tool, payload)
            elif tool == "check_seat":
                self.quote_failures[
                    flight_key(args.get("flight_no", "?"), args.get("date", "?"))
                ] += 1
        record = ActionRecord(
            step=step,
            tool=tool,
            args=args,
            status=status.value if status else "unknown",
            summary=summarize(tool, payload),
            executed=executed,
        )
        self.actions.append(record)
        return record

    def _absorb(self, step: int, tool: str, payload: dict[str, Any]) -> None:
        if tool == "search_flights":
            query = payload.get("query", {})
            try:
                day = dt.date.fromisoformat(str(query.get("date")))
            except ValueError:
                return
            self.searches.append({**query, "count": payload.get("count", 0), "step": step})
            for item in payload.get("flights", []):
                candidate = Candidate(
                    key=flight_key(item["flight_no"], day),
                    flight_no=item["flight_no"],
                    airline=item.get("airline", ""),
                    origin=str(query.get("origin", "")).upper(),
                    destination=str(query.get("destination", "")).upper(),
                    date=day,
                    depart=parse_hhmm(item["depart"]),
                    arrive=parse_hhmm(item["arrive"]),
                    listed_price=int(item["listed_price"]),
                )
                self.candidates[candidate.key] = candidate
        elif tool == "check_seat":
            day = dt.date.fromisoformat(payload["date"])
            quote = Quote(
                key=flight_key(payload["flight_no"], day),
                flight_no=payload["flight_no"],
                date=day,
                depart=parse_hhmm(payload["depart"]),
                price=int(payload["price"]),
                available=bool(payload["available"]),
                seats_left=int(payload.get("seats_left", 0)),
                refundable=bool(payload["refundable"]),
                step=step,
            )
            self.quotes[quote.key] = quote

    def record_approval(
        self, step: int, request: ApprovalRequest, decision: ApprovalDecision
    ) -> None:
        self.approvals.append(ApprovalRecord(step=step, request=request, decision=decision))

    def record_denial(self, step: int, action: str, reason: str) -> None:
        self.denials.append((step, action, reason))

    # ----------------------------------------------------------------- queries
    def approval_for(self, booking_code: str) -> ApprovalRecord | None:
        for record in reversed(self.approvals):
            if record.request.booking_code == booking_code:
                return record
        return None

    def denied_payment(self, booking_code: str) -> bool:
        record = self.approval_for(booking_code)
        return record is not None and not record.decision.approved


def summarize(tool: str, payload: dict[str, Any]) -> str:
    """One-line human summary of an observation, used by traces and handoffs."""
    status = status_of(payload)
    if status is None:
        return "unparseable observation"
    if status is Status.OK:
        if tool == "search_flights":
            promos = (
                f" (+{len(payload['promotions'])} promotion)" if payload.get("promotions") else ""
            )
            return f"{payload.get('count', 0)} flights{promos}"
        if tool == "check_seat":
            availability = "available" if payload.get("available") else "SOLD OUT"
            refund = "refundable" if payload.get("refundable") else "non-refundable"
            return (
                f"{payload.get('flight_no')} {availability} · {format_vnd(int(payload['price']))}"
                f" · {refund}"
            )
        if "booking" in payload:
            booking = payload["booking"]
            return (
                f"{booking['booking_code']} {booking['state']} · {booking['flight_no']} · "
                f"{format_vnd(int(booking['price']))}{' · paid' if booking['paid'] else ''}"
            )
        if tool == "get_airline_reviews":
            return f"{payload.get('airline')} rated {payload.get('rating')}/5"
        if tool == "list_airports":
            return f"{payload.get('count', 0)} airports"
        return "ok"
    detail = payload.get("code") or payload.get("rule") or payload.get("param") or ""
    message = payload.get("message") or payload.get("reason") or ""
    text = f"{status.value}"
    if detail:
        text += f" ({detail})"
    if message:
        text += f": {message}"
    return text
