"""Common interface of the reasoning patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
from google.genai import errors as genai_errors
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.errors import GraphRecursionError

from flight_agent.harness.handoff import StopReason
from flight_agent.harness.result import RunResult
from flight_agent.harness.runtime import Harness

CONFIGURATION_ERRORS = (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)
"""Errors caused by the request or the settings; retrying cannot fix them."""

PROVIDER_ERRORS = (ModelError, ChatGoogleGenerativeAIError, genai_errors.APIError, httpx.HTTPError)
"""Rate limits, outages, timeouts: the run is lost to infrastructure, not to the agent."""


class BookingAgent(ABC):
    """A reasoning pattern. All patterns run inside the same harness."""

    name: str

    def run(self, harness: Harness, *, label: str = "") -> RunResult:
        harness.start(pattern=self.name, label=label)
        final_message: str | None = None
        try:
            final_message = self._run(harness)
        except GraphRecursionError:
            harness.stop(StopReason.BUDGET_EXHAUSTED, "graph recursion limit reached")
        except CONFIGURATION_ERRORS:
            raise
        except PROVIDER_ERRORS as error:
            harness.stop(StopReason.INFRA_ERROR, f"{type(error).__name__}: {error}"[:300])
        return harness.finalize(pattern=self.name, final_message=final_message)

    @abstractmethod
    def _run(self, harness: Harness) -> str | None:
        """Drive the pattern until it ends; return the model's final message, if any."""
