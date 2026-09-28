"""Harness layer: loop and stall detection (frameworks do not provide these).

Three signals, checked after every observation:

1. **repeated action**  - the same ``(tool, args)`` recurs within a sliding window;
2. **repeated failure** - different arguments keep producing the same failure;
3. **no progress**      - a task-specific progress measure stops improving.

Each detector first *warns* (the warning is appended to the observation, so the
model can change course) and then *stops* the run.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, Field


class LoopConfig(BaseModel):
    window: int = 6
    """How many recent actions are compared."""
    repeat_limit: int = 3
    """The N-th identical call (or N-th identical failure in a row) stops the run."""
    polling_limits: dict[str, int] = Field(default_factory=lambda: {"get_booking": 5})
    """Read-only status polling is legitimate and gets a larger allowance."""


class StallConfig(BaseModel):
    patience: int = 6
    """Stop after this many actions without progress."""


class Signal(BaseModel):
    level: Literal["warn", "stop"]
    kind: Literal["loop", "stall"]
    message: str
    """Text appended to the observation as ``harness_notice``."""


def fingerprint(tool: str, args: dict[str, Any]) -> tuple[str, str]:
    normalized = {k: v.strip().upper() if isinstance(v, str) else v for k, v in args.items()}
    return tool, json.dumps(normalized, sort_keys=True, default=str)


class LoopDetector:
    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()
        self._recent: deque[tuple[str, str]] = deque(maxlen=self.config.window)
        self._failure: tuple[Any, ...] | None = None
        self._failure_streak = 0

    def observe(self, tool: str, args: dict[str, Any], payload: dict[str, Any]) -> Signal | None:
        fp = fingerprint(tool, args)
        limit = self.config.polling_limits.get(tool, self.config.repeat_limit)
        repeats = self._recent.count(fp) + 1
        self._recent.append(fp)

        if payload.get("status") == "ok":
            self._failure, self._failure_streak = None, 0
        else:
            digest = (
                tool,
                payload.get("status"),
                payload.get("code") or payload.get("rule") or payload.get("param"),
            )
            self._failure_streak = self._failure_streak + 1 if digest == self._failure else 1
            self._failure = digest

        if repeats >= limit:
            return Signal(
                level="stop",
                kind="loop",
                message=f"{tool} was called {repeats} times with identical arguments "
                f"within the last {self.config.window} actions.",
            )
        streak_limit = self.config.repeat_limit
        if self._failure is not None and self._failure_streak >= streak_limit:
            return Signal(
                level="stop",
                kind="loop",
                message=f"The same failure ({self._failure[1]}: {self._failure[2]}) was "
                f"returned {self._failure_streak} times in a row.",
            )
        if repeats == limit - 1:
            return Signal(
                level="warn",
                kind="loop",
                message=f"This exact {tool} call has now been made {repeats} times. One more "
                "identical call will stop the run; change your approach.",
            )
        if self._failure_streak == streak_limit - 1:
            return Signal(
                level="warn",
                kind="loop",
                message=f"The same failure was returned {self._failure_streak} times in a row. "
                "One more will stop the run; change your approach.",
            )
        return None


class StallDetector:
    """Stops when no component of the progress measure has improved for ``patience`` actions.

    Progress has several components (e.g. furthest funnel stage reached, best
    requirement score, distinct options evaluated). Each is compared with its own best
    so far: exploring a new option after a setback (a declined payment, a cancelled
    hold) still counts as progress, while re-checking known options does not.
    """

    def __init__(self, config: StallConfig | None = None) -> None:
        self.config = config or StallConfig()
        self.best: tuple[int, ...] | None = None
        self.idle = 0

    def observe(self, progress: tuple[int, ...]) -> Signal | None:
        if self.best is None or any(p > b for p, b in zip(progress, self.best, strict=True)):
            self.best = (
                progress
                if self.best is None
                else tuple(max(p, b) for p, b in zip(progress, self.best, strict=True))
            )
            self.idle = 0
            return None
        self.idle += 1
        if self.idle >= self.config.patience:
            return Signal(
                level="stop",
                kind="stall",
                message=f"No measurable progress in the last {self.idle} actions.",
            )
        if self.idle == self.config.patience - 1:
            return Signal(
                level="warn",
                kind="stall",
                message=f"No measurable progress in the last {self.idle} actions. Either make "
                "progress on the pinned requirements or stop and explain what blocks you.",
            )
        return None
