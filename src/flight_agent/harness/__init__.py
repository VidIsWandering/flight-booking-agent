"""The harness: every part of the agent loop that is code rather than model.

Layers (each in its own module):

========================  =================================================
constraints               requirements as typed data, pinned into context
permissions / approval    gate before execution: allow / deny / ask a human
completion                "done" is verified by code, never by the model
detectors                 loop and stall detection
budget                    hard limits on calls, tokens, time and money
grounding                 final-answer facts must come from observations
handoff                   stop reasons and a handoff a person can act on
trace                     JSONL trace of every decision
runtime                   the per-call checklist that ties it all together
middleware                adapter for LangChain ``create_agent``
========================  =================================================
"""

from flight_agent.harness.approval import (
    ApprovalDecision,
    ApprovalRequest,
    Approver,
    AutoApprover,
    ConsoleApprover,
    DenyAllApprover,
)
from flight_agent.harness.budget import Budget, Pricing, UsageMeter
from flight_agent.harness.completion import CompletionReport, verify_completion
from flight_agent.harness.constraints import TripConstraints, Violation
from flight_agent.harness.detectors import LoopConfig, LoopDetector, StallConfig, StallDetector
from flight_agent.harness.grounding import GroundingReport, check_grounding
from flight_agent.harness.handoff import Handoff, StopReason
from flight_agent.harness.permissions import Decision, PermissionGate, PermissionPolicy
from flight_agent.harness.result import RunMetrics, RunResult
from flight_agent.harness.runtime import Harness, HarnessConfig, ToolOutcome
from flight_agent.harness.trace import ConsolePrinter, Tracer

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "Budget",
    "CompletionReport",
    "ConsoleApprover",
    "ConsolePrinter",
    "Decision",
    "DenyAllApprover",
    "GroundingReport",
    "Handoff",
    "Harness",
    "HarnessConfig",
    "LoopConfig",
    "LoopDetector",
    "PermissionGate",
    "PermissionPolicy",
    "Pricing",
    "RunMetrics",
    "RunResult",
    "StallConfig",
    "StallDetector",
    "StopReason",
    "ToolOutcome",
    "Tracer",
    "TripConstraints",
    "UsageMeter",
    "Violation",
    "check_grounding",
    "verify_completion",
]
