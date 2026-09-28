"""Harness layer: structured trace of every model call, tool call and harness decision.

Traces are JSON lines, one event per line, so a failed run can be replayed and the
first wrong step located without reading the final answer.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel


class TraceEvent(BaseModel):
    seq: int
    t: float
    kind: str
    data: dict[str, Any]


class Tracer:
    def __init__(self, sink: Callable[[TraceEvent], None] | None = None) -> None:
        self.events: list[TraceEvent] = []
        self._t0 = time.monotonic()
        self._sink = sink

    def emit(self, event_kind: str, /, **data: Any) -> TraceEvent:
        event = TraceEvent(
            seq=len(self.events) + 1,
            t=round(time.monotonic() - self._t0, 3),
            kind=event_kind,
            data=data,
        )
        self.events.append(event)
        if self._sink is not None:
            self._sink(event)
        return event

    @staticmethod
    def read_jsonl(path: str | Path) -> list[TraceEvent]:
        """Load a trace written by :meth:`write_jsonl`."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [TraceEvent.model_validate_json(line) for line in lines if line.strip()]

    def of_kind(self, kind: str) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind]

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(json.dumps(event.model_dump(), ensure_ascii=False, default=str) + "\n")
        return path


def _args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _structured_summary(name: str, value: dict[str, Any]) -> str:
    """Plans are printed in full by their own event; keep the model line short."""
    steps = len(value.get("steps") or [])
    if name == "Replan" or "decision" in value:
        return f"Replan(decision={value.get('decision')}, {steps} steps)"
    return f"Plan({steps} steps)"


def _call_summary(call: dict[str, Any]) -> str:
    args = call.get("args", {})
    if call["name"] in {"Plan", "Replan"}:
        return _structured_summary(call["name"], args)
    return f"{call['name']}({_args(args)})"


def _answer_summary(role: str, text: str) -> str:
    """Planner answers arrive as JSON text (native structured output). The recorded
    text may be truncated, so fall back to the decision alone when it does not parse."""
    if role in {"planner", "replanner"}:
        name = "Replan" if role == "replanner" else "Plan"
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return _structured_summary(name, value)
        decision = re.search(r'"decision"\s*:\s*"(\w+)"', text)
        return f"{name}(decision={decision.group(1)})" if decision else name
    return "answer: " + " ".join(text.split())[:160]


class ConsolePrinter:
    """Renders trace events as a compact, readable log (the "read the trace" view)."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._model_calls = 0

    def __call__(self, event: TraceEvent) -> None:
        line = self.format(event)
        if line:
            print(line, file=self.stream, flush=True)

    def format(self, event: TraceEvent) -> str | None:
        d = event.data
        kind = event.kind
        if kind == "run_start":
            return f"== {d['pattern']} | {d.get('label', '')}\n   request: {d['request']}"
        if kind == "model_call":
            self._model_calls += 1
            if d.get("error"):
                return f"  M{self._model_calls:<3}{d['role']:<10} !! {d['error']}"
            if d.get("tool_calls"):
                what = "; ".join(_call_summary(c) for c in d["tool_calls"])
            else:
                what = _answer_summary(d["role"], d.get("text") or "")
            return (
                f"  M{self._model_calls:<3}{d['role']:<10} -> {what}  "
                f"[{d.get('total_tokens', 0):,} tok, {d.get('seconds', 0):.1f}s]"
            )
        if kind == "tool_call":
            flag = "" if d.get("executed") else "  (not executed)"
            notice = f"\n        ! {d['notice']}" if d.get("notice") else ""
            return (
                f"  T{d['step']:<3}{d['tool']}({_args(d['args'])}) <- {d['summary']}{flag}{notice}"
            )
        if kind == "approval":
            verdict = "APPROVED" if d["approved"] else "DECLINED"
            note = f" ({d['note']})" if d.get("note") else ""
            action = (
                d["action"]
                if d["kind"] == "action"
                else f"{d['action'].count(chr(10)) + 1}-step plan"
            )
            return f"  ?   approval [{d['kind']}] {verdict} by {d['approver']}{note}: {action}"
        if kind in {"plan", "replan"}:
            steps = "\n".join(
                f"        {i}. {s['tool']}: {s['purpose']}"
                + (f" [{s['args_hint']}]" if s.get("args_hint") else "")
                for i, s in enumerate(d.get("steps", []), 1)
            )
            head = "plan" if kind == "plan" else f"re-plan ({d.get('reason', '')})"
            return f"  #   {head}\n{steps}" if steps else f"  #   {head}: {d.get('decision', '')}"
        if kind == "deviation":
            return f"  ~   deviation: {d['reason']}"
        if kind == "nudge":
            return f"  ^   harness: {d['message']}"
        if kind == "stop":
            return f"  ##  stop: {d['reason']} - {d['detail']}"
        return None
