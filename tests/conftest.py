from __future__ import annotations

from typing import Any

import pytest

from flight_agent.evaluation.scenarios import Scenario, load_scenarios
from flight_agent.harness.approval import Approver
from flight_agent.harness.runtime import Harness, HarnessConfig


@pytest.fixture(scope="session")
def scenarios() -> dict[str, Scenario]:
    return {s.id: s for s in load_scenarios()}


@pytest.fixture
def make_harness(scenarios: dict[str, Scenario]):
    def make(
        scenario_id: str = "baseline",
        *,
        approver: Approver | None = None,
        config: HarnessConfig | None = None,
        **policy: Any,
    ) -> Harness:
        scenario = scenarios[scenario_id]
        return Harness(
            constraints=scenario.constraints,
            backend=scenario.build_backend(seed=1),
            policy=scenario.policy.model_copy(update=policy),
            approver=approver or scenario.approver(),
            config=config,
        )

    return make
