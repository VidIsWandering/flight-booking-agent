"""Human approval: requests, decisions and approver implementations.

An approval request is a small handoff: it says where the run is, what the agent
intends to do and why the harness is asking, so a person can answer quickly.
"""

from __future__ import annotations

import sys
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ApprovalRequest(BaseModel):
    kind: Literal["action", "plan"]
    where: str = Field(description="Where the run is now.")
    action: str = Field(description="What the agent intends to do.")
    why: str = Field(description="Why the harness asks instead of deciding.")
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    amount: int | None = None
    refundable: bool | None = None
    booking_code: str | None = None

    def render(self) -> str:
        return (
            f"Where we are : {self.where}\nIntended     : {self.action}\nWhy we ask   : {self.why}"
        )


class ApprovalDecision(BaseModel):
    approved: bool
    approver: str
    note: str = ""


class Approver(Protocol):
    """Anything that can answer an approval request (a person, a policy, a test)."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


class AutoApprover:
    """Approves everything. Useful for demos; never for real payments."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, approver="auto", note="auto-approved")


class DenyAllApprover:
    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=False, approver="deny-all", note="auto-denied")


class ConsoleApprover:
    """Asks a person on the terminal."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        print("\n--- approval needed " + "-" * 40, file=sys.stderr)
        print(request.render(), file=sys.stderr)
        answer = input("Approve? [y/N] (optional note after a space): ").strip()
        approved = answer[:1].lower() == "y"
        note = answer[1:].strip() if len(answer) > 1 else ""
        return ApprovalDecision(approved=approved, approver="console", note=note)
