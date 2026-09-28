"""Observation contract shared by tools and the harness.

Every observation the model receives is a JSON object with an explicit ``status``.
An empty string or free text leaves the model guessing whether a call found
nothing, was malformed or failed; an explicit status tells it which of those
happened and what to do next.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "ok"
    """The call worked. The payload may still be empty (``count: 0``)."""
    NOT_FOUND = "not_found"
    """The referenced entity does not exist."""
    INVALID_PARAM = "invalid_param"
    """The call was malformed; ``param`` and ``allowed``/``expected`` say how."""
    REJECTED = "rejected"
    """A business rule refused the request (``code`` says which)."""
    ERROR = "error"
    """A technical failure; ``retryable`` says whether trying again can help."""
    DENIED = "denied"
    """The harness (policy or human approver) refused to execute the call."""
    SKIPPED = "skipped"
    """The harness has stopped the run; the call was not executed."""


def dump(payload: dict[str, Any]) -> str:
    """Serialize an observation to the JSON string the model receives."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, default=str)


def load(content: Any) -> dict[str, Any] | None:
    """Parse an observation back into a dict; ``None`` if it is not a JSON object."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list):  # LangChain content blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def status_of(payload: dict[str, Any] | None) -> Status | None:
    if not payload:
        return None
    try:
        return Status(payload.get("status"))
    except ValueError:
        return None


def envelope(status: Status, **fields: Any) -> dict[str, Any]:
    """Build an observation, dropping ``None`` fields to keep it compact."""
    return {"status": status.value, **{k: v for k, v in fields.items() if v is not None}}
