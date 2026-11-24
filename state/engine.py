from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class TransitionResult:
    transition_id: str
    timestamp: str
    event: str
    source: str
    previous_state: dict[str, float]
    new_state: dict[str, float]
    delta: dict[str, float]
    metadata: dict[str, Any]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalize_state(state: dict[str, float], dimensions: list[str]) -> dict[str, float]:
    return {dimension: float(state.get(dimension, 0.0)) for dimension in dimensions}


def apply_transition(
    current_state: dict[str, float],
    event_type: str,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    dimensions: list[str] | None = None,
) -> TransitionResult:
    dimensions = dimensions or config["dimensions"]
    bounds = config["bounds"]
    minimum = float(bounds["min"])
    maximum = float(bounds["max"])
    transitions: dict[str, dict[str, float]] = config.get("transitions", {})
    event_deltas = transitions.get(event_type, {})

    previous_state = normalize_state(current_state, dimensions)
    delta: dict[str, float] = {}
    new_state: dict[str, float] = {}

    for dimension in dimensions:
        change = float(event_deltas.get(dimension, 0.0))
        delta[dimension] = change
        new_state[dimension] = clamp(previous_state[dimension] + change, minimum, maximum)

    return TransitionResult(
        transition_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        event=event_type,
        source="experimental_state_engine",
        previous_state=previous_state,
        new_state=new_state,
        delta=delta,
        metadata=metadata or {},
    )


def replay_transitions(
    initial_core: dict[str, float],
    initial_affective: dict[str, float],
    event_types: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    """Deterministically replay a sequence of events on core and affective vectors."""
    core_state = dict(initial_core)
    affective_state = dict(initial_affective)
    affective_dimensions = config.get("affective_dimensions", [])

    for event_type in event_types:
        core_result = apply_transition(core_state, event_type, config, dimensions=config["dimensions"])
        core_state = dict(core_result.new_state)
        if affective_dimensions:
            affective_result = apply_transition(
                affective_state,
                event_type,
                config,
                dimensions=affective_dimensions,
            )
            affective_state = dict(affective_result.new_state)

    return core_state, affective_state
