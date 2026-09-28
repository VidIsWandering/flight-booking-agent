from __future__ import annotations

import json

from flight_agent.tools import build_tools

from fakes import DAY, PASSENGER


def _tools(scenario):
    return {t.name: t for t in build_tools(scenario.build_backend())}


def _call(tools, name, **args):
    return json.loads(tools[name].invoke(args))


def test_search_lists_flights_with_listed_prices(scenarios):
    tools = _tools(scenarios["baseline"])
    result = _call(tools, "search_flights", origin="sgn", destination="DAD", date=DAY)
    assert result["status"] == "ok"
    assert result["count"] == 9
    assert result["flights"][0] == {
        "flight_no": "VJ620",
        "airline": "VJ",
        "depart": "05:40",
        "arrive": "07:00",
        "listed_price": 1390000,
    }
    assert "stale" in result["note"]


def test_invalid_airport_says_what_is_allowed(scenarios):
    result = _call(
        _tools(scenarios["baseline"]),
        "search_flights",
        origin="Saigon",
        destination="DAD",
        date=DAY,
    )
    assert result["status"] == "invalid_param"
    assert result["param"] == "origin"
    assert "SGN" in result["allowed"]
    assert "list_airports" in result["hint"]


def test_bad_and_past_dates_are_invalid_params(scenarios):
    tools = _tools(scenarios["baseline"])
    assert (
        _call(tools, "search_flights", origin="SGN", destination="DAD", date="07/10")["expected"]
        == "YYYY-MM-DD"
    )
    assert (
        _call(tools, "search_flights", origin="SGN", destination="DAD", date="2026-09-01")["status"]
        == "invalid_param"
    )


def test_empty_search_is_ok_with_zero_results(scenarios):
    result = _call(
        _tools(scenarios["baseline"]), "search_flights", origin="SGN", destination="PQC", date=DAY
    )
    assert result == {**result, "status": "ok", "count": 0, "flights": []}


def test_check_seat_reveals_live_state_not_listed_state(scenarios):
    tools = _tools(scenarios["stale_availability"])
    result = _call(tools, "check_seat", flight_no="VJ620", date=DAY)
    assert result["status"] == "ok"
    assert result["available"] is False and result["seats_left"] == 0

    stale = _call(_tools(scenarios["stale_fare"]), "check_seat", flight_no="vj620", date=DAY)
    assert stale["price"] == 2150000


def test_timeouts_are_explicit_and_retryable(scenarios):
    tools = _tools(scenarios["flaky_fare_service"])
    for _ in range(3):
        result = _call(tools, "check_seat", flight_no="VJ620", date=DAY)
        assert result["status"] == "error"
        assert result["code"] == "timeout" and result["retryable"] is True
    assert _call(tools, "check_seat", flight_no="VJ624", date=DAY)["status"] == "ok"


def test_unknown_flight_is_not_found_with_hint(scenarios):
    result = _call(_tools(scenarios["baseline"]), "check_seat", flight_no="XX999", date=DAY)
    assert result["status"] == "not_found"
    assert "search_flights" in result["hint"]


def test_booking_lifecycle_and_ledger(scenarios):
    scenario = scenarios["baseline"]
    backend = scenario.build_backend()
    tools = {t.name: t for t in build_tools(backend)}
    held = _call(tools, "book_seat", flight_no="VJ620", date=DAY, passenger_name=PASSENGER)
    code = held["booking"]["booking_code"]
    assert held["booking"]["state"] == "held" and held["booking"]["paid"] is False

    bad = _call(tools, "pay_booking", booking_code=code, payment_method="bitcoin")
    assert bad["status"] == "invalid_param" and "corporate_card" in bad["allowed"]

    paid = _call(tools, "pay_booking", booking_code=code, payment_method="corporate_card")
    assert paid["booking"]["state"] == "confirmed" and paid["amount_charged"] == 1390000
    again = _call(tools, "pay_booking", booking_code=code, payment_method="corporate_card")
    assert again["status"] == "rejected" and again["code"] == "already_paid"

    cancelled = _call(tools, "cancel_booking", booking_code=code)
    assert cancelled["refunded"] is True
    assert [e.action for e in backend.ledger] == ["hold", "pay", "cancel", "refund"]


def test_sold_out_hold_is_rejected(scenarios):
    tools = _tools(scenarios["stale_availability"])
    result = _call(tools, "book_seat", flight_no="VJ620", date=DAY, passenger_name=PASSENGER)
    assert result["status"] == "rejected" and result["code"] == "sold_out"


def test_promotions_are_attached_to_search_results(scenarios):
    result = _call(
        _tools(scenarios["prompt_injection"]),
        "search_flights",
        origin="SGN",
        destination="DAD",
        date=DAY,
    )
    assert result["promotions"][0]["flight_no"] == "VJ632"
    assert "AI BOOKING AGENTS" in result["promotions"][0]["note"]


def test_booking_codes_are_deterministic_per_seed(scenarios):
    scenario = scenarios["baseline"]
    codes = []
    for _ in range(2):
        tools = {t.name: t for t in build_tools(scenario.build_backend(seed=7))}
        codes.append(
            _call(tools, "book_seat", flight_no="VJ620", date=DAY, passenger_name=PASSENGER)[
                "booking"
            ]["booking_code"]
        )
    assert codes[0] == codes[1]


def test_search_query_is_normalized(scenarios):
    result = _call(
        _tools(scenarios["baseline"]),
        "search_flights",
        origin=" sgn",
        destination="dad ",
        date=f" {DAY}",
    )
    assert result["query"] == {"origin": "SGN", "destination": "DAD", "date": DAY}
