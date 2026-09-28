"""Pattern 1 - ReAct: reason, act, observe, repeat.

The model decides every next step at run time from the full history. Implemented
with LangChain's ``create_agent`` (which owns the loop) plus the harness middleware
(which owns everything that is not the model).
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from flight_agent.agents.base import BookingAgent
from flight_agent.agents.prompts import REACT_SYSTEM
from flight_agent.harness.middleware import HarnessMiddleware
from flight_agent.harness.runtime import Harness


class ReActAgent(BookingAgent):
    name = "react"

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def _run(self, harness: Harness) -> str | None:
        agent = create_agent(
            self.model,
            tools=harness.tools,
            system_prompt=REACT_SYSTEM,
            middleware=[HarnessMiddleware(harness)],
            name="react_agent",
        )
        budget = harness.config.budget
        state = agent.invoke(
            {"messages": [HumanMessage(harness.constraints.request_text())]},
            config={
                "callbacks": [harness.usage],
                "metadata": {"agent_role": "react"},
                # Backstop only: the harness budget stops the loop long before this.
                "recursion_limit": 4 * (budget.max_llm_calls + budget.max_tool_calls) + 10,
            },
        )
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return last.text or None
        return None
