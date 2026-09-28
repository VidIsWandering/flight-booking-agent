"""Adapter that plugs the harness into LangChain's ``create_agent`` loop.

``create_agent`` owns the ReAct loop (model -> tools -> model ...). This middleware
hooks into it so that every stage that is *not* the model runs harness code:

* ``before_model``    - budget check; end the loop when the harness has stopped;
* ``wrap_model_call`` - context building: pin requirements and harness status;
* ``wrap_tool_call``  - permission gate, execution, recording, post-observation checks;
* ``after_model``     - the model wants to finish: verify before trusting it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, hook_config
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from flight_agent.harness.runtime import Harness


class HarnessMiddleware(AgentMiddleware):
    def __init__(self, harness: Harness) -> None:
        super().__init__()
        self.harness = harness

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self.harness.before_model_call():
            return {"jump_to": "end"}
        return None

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        base = request.system_prompt or ""
        pinned = self.harness.pinned_context()
        return handler(request.override(system_message=SystemMessage(f"{base}\n\n{pinned}")))

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        pushback = self.harness.review_finish(last.text)
        if pushback is None:
            return None
        return {"messages": [HumanMessage(pushback)], "jump_to": "model"}

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        outcome = self.harness.run_tool(
            call["name"], call.get("args") or {}, execute=lambda: handler(request)
        )
        return ToolMessage(
            content=outcome.content,
            tool_call_id=call["id"],
            name=call["name"],
            status="success" if outcome.ok else "error",
        )
