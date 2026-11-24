from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

INSTRUMENTATION_SCHEMA_VERSION = "2.1.0"

VALID_STAGES = frozenset(
    {
        "baseline",
        "post_prediction",
        "post_information",
        "post_contradiction",
        "post_revision",
        "self_report",
        "free",
        "prediction",
        "visual_baseline",
        "visual_prediction",
        "visual_contradiction",
        "visual_revision",
        "visual_self_observation",
    }
)


@dataclass
class ObservationRecord:
    observation_id: str
    trial_id: str | None
    experiment_id: str
    stage: str
    timestamp: str
    source: str
    previous_observation_id: str | None
    behavioral: dict[str, Any]
    engine_state: dict[str, Any]
    model_io: dict[str, Any]
    self_report: dict[str, Any]
    internal_activation: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        experiment_id: str,
        stage: str,
        source: str,
        trial_id: str | None = None,
        previous_observation_id: str | None = None,
        behavioral: dict[str, Any] | None = None,
        engine_state: dict[str, Any] | None = None,
        model_io: dict[str, Any] | None = None,
        self_report: dict[str, Any] | None = None,
        internal_activation: dict[str, Any] | None = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> ObservationRecord:
        return cls(
            observation_id=str(uuid4()),
            trial_id=trial_id,
            experiment_id=experiment_id,
            stage=stage,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            previous_observation_id=previous_observation_id,
            behavioral=behavioral or {"status": "absent"},
            engine_state=engine_state or {},
            model_io=model_io or {"status": "unsupported", "metrics": None, "provenance": "lm_studio_mcp"},
            self_report=self_report or {"status": "absent", "content": None, "confidence_claims": None},
            internal_activation=internal_activation
            or {"status": "unsupported", "adapter": "null", "checkpoints": None},
            errors=errors or [],
            warnings=warnings or [],
            provenance=provenance or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "source": self.source,
            "previous_observation_id": self.previous_observation_id,
            "behavioral": self.behavioral,
            "engine_state": self.engine_state,
            "model_io": self.model_io,
            "self_report": self.self_report,
            "internal_activation": self.internal_activation,
            "errors": self.errors,
            "warnings": self.warnings,
            "provenance": {
                **self.provenance,
                "schema_version": INSTRUMENTATION_SCHEMA_VERSION,
            },
        }
