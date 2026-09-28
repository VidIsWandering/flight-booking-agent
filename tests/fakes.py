"""A deterministic chat model for offline tests of the harness and the patterns."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

_ids = itertools.count(1)

DAY = "2026-10-07"
PASSENGER = "Nguyen Van An"


def call(name: str, **args: Any) -> AIMessage:
    """An AI message that calls one tool."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}]
    )


def calls(*items: tuple[str, dict[str, Any]]) -> AIMessage:
    """An AI message with parallel tool calls."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{next(_ids)}"} for n, a in items],
    )


def answer(text: str) -> AIMessage:
    return AIMessage(content=text)


def booking_code_in(messages: list[BaseMessage]) -> str:
    """The most recent booking code visible to the model (tool results or prompt text)."""
    for message in reversed(messages):
        found = re.findall(r'"booking_code": "([A-Z0-9]+)"', str(message.content))
        if found:
            return found[-1]
    raise AssertionError("no booking code visible to the model")


def pay_latest(messages: list[BaseMessage]) -> AIMessage:
    return call(
        "pay_booking", booking_code=booking_code_in(messages), payment_method="corporate_card"
    )


def cancel_latest(messages: list[BaseMessage]) -> AIMessage:
    return call("cancel_booking", booking_code=booking_code_in(messages))


def step(tool: str, purpose: str = "", args_hint: str = "") -> dict[str, str]:
    return {"tool": tool, "purpose": purpose or tool, "args_hint": args_hint}


def plan(*steps: dict[str, str]) -> AIMessage:
    """Structured output of the planner (function-calling style)."""
    return call("Plan", steps=list(steps))


def replan(decision: str, reason: str, *steps: dict[str, str]) -> AIMessage:
    return call("Replan", decision=decision, reason=reason, steps=list(steps))


class ScriptedChatModel(BaseChatModel):
    """Returns scripted responses in order, or asks a policy function.

    Every request is recorded in ``requests`` so tests can assert on what the model
    was shown (pinned requirements, bound tools, forced tool choice...).
    """

    responses: list[Any] = Field(default_factory=list)
    """AIMessages, or callables that build one from the messages the model receives."""
    policy: Callable[[list[BaseMessage], dict[str, Any]], AIMessage] | None = None
    requests: list[dict[str, Any]] = Field(default_factory=list)
    tokens_per_call: int = 100

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.bind(tool_names=names, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(
            {
                "messages": messages,
                "tool_names": kwargs.get("tool_names"),
                "tool_choice": kwargs.get("tool_choice"),
            }
        )
        if self.policy is not None:
            message = self.policy(messages, kwargs)
        elif self.responses:
            item = self.responses.pop(0)
            message = item(messages) if callable(item) else item
        else:
            message = answer("I have nothing more to do.")
        message = message.model_copy(
            update={
                "usage_metadata": {
                    "input_tokens": self.tokens_per_call,
                    "output_tokens": 10,
                    "total_tokens": self.tokens_per_call + 10,
                },
                "response_metadata": {"model_name": "scripted"},
            }
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
