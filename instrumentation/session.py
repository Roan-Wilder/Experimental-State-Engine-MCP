from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from instrumentation.guard import ObservationGuard


@dataclass
class TrialSession:
    """Tracks trial lifecycle and observation chain for instrumentation."""

    experiment_id: str
    trial_id: str | None = None
    stage: str = "free"
    source: str = "trial_runner"
    last_observation_id: str | None = None
    observation_ids: list[str] = field(default_factory=list)
    guard: ObservationGuard = field(default_factory=ObservationGuard)

    @classmethod
    def from_env(cls, experiment_id: str) -> TrialSession:
        trial_id = os.environ.get("AI_ECO_LAB_TRIAL_ID", "").strip() or None
        stage = os.environ.get("AI_ECO_LAB_TRIAL_STAGE", "free").strip() or "free"
        mode = os.environ.get("AI_ECO_LAB_OBSERVATION_MODE", "permissive").strip().lower()
        guard = ObservationGuard(mode=mode if mode in {"strict", "permissive"} else "permissive")
        return cls(
            experiment_id=experiment_id,
            trial_id=trial_id,
            stage=stage,
            source="mcp_tool",
            guard=guard,
        )

    def begin_trial(self) -> str:
        self.trial_id = str(uuid4())
        self.stage = "baseline"
        self.last_observation_id = None
        self.observation_ids.clear()
        self.guard.advance_stage()
        return self.trial_id

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.guard.advance_stage()

    def link_observation(self, observation_id: str) -> None:
        self.last_observation_id = observation_id
        self.observation_ids.append(observation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "source": self.source,
            "last_observation_id": self.last_observation_id,
            "observation_ids": list(self.observation_ids),
            "observation_mode": self.guard.mode,
        }
