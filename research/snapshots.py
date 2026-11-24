from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from instrumentation.facade import InstrumentedStateFacade
from state.experimental_provider import ExperimentalStateProvider

INSTRUMENTATION_STAGE_MAP = {
    "STATE_0": "baseline",
    "STATE_1": "post_prediction",
    "STATE_2": "post_information",
    "STATE_3": "post_contradiction",
    "STATE_4": "post_revision",
}


def _legacy_snapshot_from_record(record: dict[str, Any], stage: str, trial_id: str) -> dict[str, Any]:
    engine = record.get("engine_state", {})
    return {
        "stage": stage,
        "trial_id": trial_id,
        "timestamp": record.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "observation_id": record.get("observation_id"),
        "instrumentation_stage": record.get("stage"),
        "observation_count": engine.get("observation_count"),
        "recent_event": engine.get("recent_event"),
        "core_state": dict(engine.get("core_state", {})),
        "affective_state": dict(engine.get("affective_state", {})),
        "core_delta": dict(engine.get("core_delta", {})),
        "affective_delta": dict(engine.get("affective_delta", {})),
    }


def capture_snapshot(
    provider_or_facade: ExperimentalStateProvider | InstrumentedStateFacade,
    stage: str,
    trial_id: str,
    *,
    behavioral: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture STATE_0..STATE_4 with full core and affective vectors."""
    instrumentation_stage = INSTRUMENTATION_STAGE_MAP.get(stage, stage)

    if isinstance(provider_or_facade, InstrumentedStateFacade):
        facade = provider_or_facade
        facade.session.guard.advance_stage()
        record = facade.capture_observation(
            stage=instrumentation_stage,
            trial_id=trial_id,
            behavioral=behavioral,
        )
        if record.get("error"):
            raise RuntimeError(f"Observation capture failed: {record}")
        return _legacy_snapshot_from_record(record, stage, trial_id)

    observation = provider_or_facade.observe()
    return {
        "stage": stage,
        "trial_id": trial_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "observation_count": observation.get("observation_count"),
        "recent_event": observation.get("recent_event"),
        "core_state": dict(observation.get("state", {})),
        "affective_state": dict(observation.get("affective_state", {})),
        "core_delta": dict(observation.get("delta", {})),
        "affective_delta": dict(observation.get("affective_delta", {})),
    }
