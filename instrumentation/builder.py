from __future__ import annotations

from typing import TYPE_CHECKING, Any

from instrumentation.adapters.internal_activation import NullInternalActivationAdapter
from instrumentation.adapters.model_io import ModelIOAdapter, NullModelIOAdapter
from instrumentation.vision.types import VisionRequestContext
from instrumentation.session import TrialSession
from instrumentation.types import ObservationRecord

if TYPE_CHECKING:
    from state.experimental_provider import ExperimentalStateProvider


def build_engine_state_channel(provider: "ExperimentalStateProvider", observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "core_state": dict(observation.get("state", {})),
        "affective_state": dict(observation.get("affective_state", {})),
        "core_delta": dict(observation.get("delta", {})),
        "affective_delta": dict(observation.get("affective_delta", {})),
        "recent_event": observation.get("recent_event"),
        "observation_count": observation.get("observation_count"),
        "interpretation": observation.get("interpretation"),
        "source": observation.get("source", "experimental_state_provider"),
    }


def build_behavioral_channel(behavioral: dict[str, Any] | None = None) -> dict[str, Any]:
    if not behavioral:
        return {"status": "absent"}
    return {"status": "present", **behavioral}


def build_observation_record(
    provider: "ExperimentalStateProvider",
    session: TrialSession,
    *,
    stage: str | None = None,
    source: str | None = None,
    behavioral: dict[str, Any] | None = None,
    self_report: dict[str, Any] | None = None,
    model_io_adapter: ModelIOAdapter | None = None,
    internal_activation_adapter: NullInternalActivationAdapter | None = None,
    tool_name: str = "capture_observation",
    warnings: list[str] | None = None,
    vision_context: VisionRequestContext | None = None,
    capability: str = "unsupported",
) -> ObservationRecord:
    observation = provider.observe()
    resolved_stage = stage or session.stage
    model_io = (model_io_adapter or NullModelIOAdapter()).capture(
        checkpoint=resolved_stage,
        context={
            "trial_id": session.trial_id,
            "vision_context": vision_context,
            "capability": capability,
        },
    )
    internal_activation = (internal_activation_adapter or NullInternalActivationAdapter()).capture_checkpoint(
        checkpoint=resolved_stage,
    )

    record = ObservationRecord.new(
        experiment_id=session.experiment_id,
        trial_id=session.trial_id,
        stage=resolved_stage,
        source=source or session.source,
        previous_observation_id=session.last_observation_id,
        behavioral=build_behavioral_channel(behavioral),
        engine_state=build_engine_state_channel(provider, observation),
        model_io=model_io,
        self_report=self_report or {"status": "absent", "content": None, "confidence_claims": None},
        internal_activation=internal_activation,
        warnings=warnings or [],
        provenance={
            "tool_name": tool_name,
            "session_id": provider.condition_context.session_id,
            "provider_observation_id": observation.get("observation_id"),
        },
    )
    return record
