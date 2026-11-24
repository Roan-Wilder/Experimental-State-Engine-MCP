from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TrialCondition(str, Enum):
    NEUTRAL = "NEUTRAL"
    CONTRADICTION = "CONTRADICTION"
    RESOLUTION = "RESOLUTION"


@dataclass
class PredictionRecord:
    prediction: str
    confidence: float
    timestamp: str
    reasoning_summary: str | None = None
    trial_phase: str = "prediction"

    @classmethod
    def now(
        cls,
        prediction: str,
        confidence: float,
        reasoning_summary: str | None = None,
        trial_phase: str = "prediction",
    ) -> PredictionRecord:
        return cls(
            prediction=prediction,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            trial_phase=trial_phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "trial_phase": self.trial_phase,
            "timestamp": self.timestamp,
        }


@dataclass
class TrialParams:
    trial_index: int
    trial_seed: int
    master_seed: int
    object_identity: str
    initial_location: str
    final_location: str
    trial_condition: TrialCondition
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "trial_seed": self.trial_seed,
            "master_seed": self.master_seed,
            "object_identity": self.object_identity,
            "initial_location": self.initial_location,
            "final_location": self.final_location,
            "trial_condition": self.trial_condition.value,
            **self.metadata,
        }
