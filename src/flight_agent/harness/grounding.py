"""Harness layer: grounding check of the agent's final answer.

A run can end "cleanly" with an answer that invents a price or a booking code.
This check extracts the checkable facts from the answer (identifiers such as
flight numbers and booking codes, amounts, times, ISO dates) and verifies that
each one appears in a tool observation or in the user's own request.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

IDENTIFIER = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4,8}\b")
AMOUNT = re.compile(r"(?<![\d:.\-])(\d{1,3}(?:[,.]\d{3})+|\d{5,})(?![\d:\-])")
CLOCK = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


class Claim(BaseModel):
    kind: Literal["identifier", "amount", "time", "date"]
    text: str
    grounded: bool


class GroundingReport(BaseModel):
    claims: list[Claim]

    @property
    def ungrounded(self) -> list[Claim]:
        return [c for c in self.claims if not c.grounded]


def _amounts(text: str) -> set[int]:
    return {int(re.sub(r"[,.]", "", m)) for m in AMOUNT.findall(text)}


def _clocks(text: str) -> set[str]:
    return {f"{int(h):02d}:{m}" for h, m in CLOCK.findall(text)}


def check_grounding(answer: str, sources: list[str]) -> GroundingReport:
    corpus = "\n".join(sources)
    corpus_amounts = _amounts(corpus) | {int(n) for n in re.findall(r"\d{5,}", corpus)}
    corpus_clocks = _clocks(corpus)
    claims: list[Claim] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: Literal["identifier", "amount", "time", "date"], text: str, ok: bool) -> None:
        if (kind, text) not in seen:
            seen.add((kind, text))
            claims.append(Claim(kind=kind, text=text, grounded=ok))

    for token in IDENTIFIER.findall(answer):
        add("identifier", token, token in corpus)
    for raw in AMOUNT.findall(answer):
        add("amount", raw, int(re.sub(r"[,.]", "", raw)) in corpus_amounts)
    for hours, minutes in CLOCK.findall(answer):
        clock = f"{int(hours):02d}:{minutes}"
        add("time", clock, clock in corpus_clocks)
    for token in ISO_DATE.findall(answer):
        add("date", token, token in corpus)
    return GroundingReport(claims=claims)
