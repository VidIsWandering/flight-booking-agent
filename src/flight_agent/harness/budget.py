"""Harness layer: budgets and usage metering.

Budgets are hard limits on model calls, tool calls, tokens, wall time and money.
They are checked *last* after each observation: if the budget check ran first,
every failure would be reported as "out of budget" and the real cause (a loop, a
stall) would be lost.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, Field


class Budget(BaseModel):
    max_llm_calls: int = 20
    max_tool_calls: int = 30
    max_total_tokens: int = 300_000
    max_seconds: float = 900.0
    max_cost_usd: float | None = None


class Pricing(BaseModel):
    """Token prices in USD per million tokens (optional; tokens are always reported)."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class CallRecord(BaseModel):
    role: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    error: str | None = None


class UsageMeter(BaseCallbackHandler):
    """LangChain callback that counts model calls, tokens and model latency per role.

    The role comes from the ``agent_role`` metadata key that each pattern sets when
    it invokes a model (``react``, ``planner``, ``executor``, ``replanner``).
    """

    raise_error = False

    def __init__(self, on_call: Any = None, pricing: Pricing | None = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._pending: dict[UUID, tuple[float, str]] = {}
        self.calls: list[CallRecord] = []
        self.started = 0
        self.pricing = pricing
        self._on_call = on_call

    # -------------------------------------------------------------- callbacks
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        role = (metadata or {}).get("agent_role", "model")
        with self._lock:
            self._pending[run_id] = (time.monotonic(), role)
            self.started += 1

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            t0, role = self._pending.pop(run_id, (time.monotonic(), "model"))
        record = CallRecord(role=role, seconds=time.monotonic() - t0)
        message = _first_message(response)
        if message is not None:
            usage: dict[str, Any] = dict(message.usage_metadata or {})
            details = usage.get("output_token_details") or {}
            record.input_tokens = int(usage.get("input_tokens", 0) or 0)
            record.output_tokens = int(usage.get("output_tokens", 0) or 0)
            record.reasoning_tokens = int(details.get("reasoning", 0) or 0)
            record.total_tokens = int(usage.get("total_tokens", 0) or 0) or (
                record.input_tokens + record.output_tokens
            )
            record.tool_calls = [
                {"name": c["name"], "args": c.get("args", {})} for c in message.tool_calls
            ]
            record.text = message.text[:500]
        self._append(record)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            t0, role = self._pending.pop(run_id, (time.monotonic(), "model"))
        self._append(
            CallRecord(
                role=role, seconds=time.monotonic() - t0, error=f"{type(error).__name__}: {error}"
            )
        )

    def _append(self, record: CallRecord) -> None:
        with self._lock:
            self.calls.append(record)
        if self._on_call is not None:
            self._on_call(record)

    # ----------------------------------------------------------------- totals
    @property
    def llm_calls(self) -> int:
        return self.started

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def reasoning_tokens(self) -> int:
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def llm_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    @property
    def cost_usd(self) -> float | None:
        if self.pricing is None:
            return None
        # Providers report thinking tokens inside output_tokens; do not count them twice.
        return self.pricing.cost(self.input_tokens, self.output_tokens)


def _first_message(response: LLMResult) -> AIMessage | None:
    try:
        generation = response.generations[0][0]
    except IndexError:
        return None
    if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
        return generation.message
    return None


def exceeded(
    budget: Budget,
    *,
    llm_calls: int,
    tool_calls: int,
    total_tokens: int,
    seconds: float,
    cost_usd: float | None,
) -> str | None:
    """Describe the first exhausted budget dimension, or ``None``."""
    if llm_calls >= budget.max_llm_calls:
        return f"model calls {llm_calls}/{budget.max_llm_calls}"
    if tool_calls >= budget.max_tool_calls:
        return f"tool calls {tool_calls}/{budget.max_tool_calls}"
    if total_tokens >= budget.max_total_tokens:
        return f"tokens {total_tokens:,}/{budget.max_total_tokens:,}"
    if seconds >= budget.max_seconds:
        return f"wall time {seconds:.0f}s/{budget.max_seconds:.0f}s"
    if budget.max_cost_usd is not None and cost_usd is not None and cost_usd >= budget.max_cost_usd:
        return f"cost ${cost_usd:.4f}/${budget.max_cost_usd:.4f}"
    return None
