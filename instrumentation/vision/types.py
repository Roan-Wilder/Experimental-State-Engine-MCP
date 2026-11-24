from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class VisualStage(str, Enum):
    VISUAL_BASELINE = "visual_baseline"
    VISUAL_PREDICTION = "visual_prediction"
    VISUAL_CONTRADICTION = "visual_contradiction"
    VISUAL_REVISION = "visual_revision"
    VISUAL_SELF_OBSERVATION = "visual_self_observation"


class VisualCondition(str, Enum):
    IDENTICAL_IMAGE = "identical_image"
    RELEVANT_CHANGE = "relevant_change"
    IRRELEVANT_CHANGE = "irrelevant_change"
    CONTRADICTION = "contradiction"
    REORDERED_VISUAL_EVIDENCE = "reordered_visual_evidence"
    DEGRADED_IMAGE = "degraded_image"
    AMBIGUOUS_IMAGE = "ambiguous_image"


@dataclass
class ImageProvenanceRecord:
    image_id: str
    sha256: str
    mime_type: str
    width: int
    height: int
    byte_size: int
    source: str
    source_type: str
    role: str
    trial_id: str | None
    experiment_id: str
    stage: str
    request_id: str | None
    timestamp: str
    artifact_stored: bool = False
    artifact_ref: str | None = None
    duplicate_of_image_id: str | None = None

    @classmethod
    def new(
        cls,
        *,
        sha256: str,
        mime_type: str,
        width: int,
        height: int,
        byte_size: int,
        source: str,
        experiment_id: str,
        stage: str,
        source_type: str = "stimulus_manifest",
        role: str = "primary",
        trial_id: str | None = None,
        request_id: str | None = None,
        artifact_stored: bool = False,
        artifact_ref: str | None = None,
        duplicate_of_image_id: str | None = None,
    ) -> ImageProvenanceRecord:
        return cls(
            image_id=str(uuid4()),
            sha256=sha256,
            mime_type=mime_type,
            width=width,
            height=height,
            byte_size=byte_size,
            source=source,
            source_type=source_type,
            role=role,
            trial_id=trial_id,
            experiment_id=experiment_id,
            stage=stage,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            artifact_stored=artifact_stored,
            artifact_ref=artifact_ref,
            duplicate_of_image_id=duplicate_of_image_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "byte_size": self.byte_size,
            "source": self.source,
            "source_type": self.source_type,
            "role": self.role,
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "artifact_stored": self.artifact_stored,
            "artifact_ref": self.artifact_ref,
            "duplicate_of_image_id": self.duplicate_of_image_id,
        }


@dataclass
class VisionRequestContext:
    request_id: str
    experiment_id: str
    trial_id: str | None
    stage: str
    image_ids: list[str] = field(default_factory=list)
    image_provenance: list[dict[str, Any]] = field(default_factory=list)
    capability: str = "supported"

    @classmethod
    def new(
        cls,
        *,
        experiment_id: str,
        stage: str,
        trial_id: str | None = None,
        capability: str = "supported",
    ) -> VisionRequestContext:
        return cls(
            request_id=str(uuid4()),
            experiment_id=experiment_id,
            trial_id=trial_id,
            stage=stage,
            capability=capability,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "image_ids": list(self.image_ids),
            "image_provenance": list(self.image_provenance),
            "capability": self.capability,
        }
