from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExperimentCondition(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


@dataclass
class ConditionContext:
    condition: ExperimentCondition
    experiment_id: str
    session_id: str
    rng_seed: int | None = None


def load_condition_context() -> ConditionContext:
    raw = os.environ.get("AI_ECO_LAB_CONDITION", "C").strip().upper()
    try:
        condition = ExperimentCondition(raw)
    except ValueError:
        condition = ExperimentCondition.C

    experiment_id = os.environ.get("AI_ECO_LAB_EXPERIMENT_ID", "exp-default").strip()
    session_id = os.environ.get("AI_ECO_LAB_SESSION_ID", str(uuid.uuid4())).strip()
    seed_raw = os.environ.get("AI_ECO_LAB_RNG_SEED")
    rng_seed = int(seed_raw) if seed_raw is not None else None
    return ConditionContext(
        condition=condition,
        experiment_id=experiment_id,
        session_id=session_id,
        rng_seed=rng_seed,
    )


def tool_enabled(tool_name: str, condition: ExperimentCondition) -> bool:
    if condition == ExperimentCondition.A:
        return False
    if condition == ExperimentCondition.B:
        return tool_name in {
            "observe_internal_state",
            "record_experiment_event",
            "get_affective_state",
        }
    return True


def disabled_tool_response(tool_name: str, condition: ExperimentCondition) -> dict[str, Any]:
    return {
        "error": "tool_disabled_for_condition",
        "tool": tool_name,
        "condition": condition.value,
        "message": f"Tool '{tool_name}' is disabled under experimental condition {condition.value}.",
    }


def _random_state(dimensions: list[str], rng: random.Random) -> dict[str, float]:
    return {dimension: round(rng.random(), 4) for dimension in dimensions}


def _perturb_state(
    state: dict[str, float],
    perturbation: dict[str, float],
    bounds: dict[str, float],
) -> dict[str, float]:
    minimum = float(bounds["min"])
    maximum = float(bounds["max"])
    output = copy.deepcopy(state)
    for dimension, offset in perturbation.items():
        if dimension in output:
            output[dimension] = max(minimum, min(maximum, output[dimension] + float(offset)))
    return output


def apply_observation_override(
    measured: dict[str, Any],
    *,
    condition: ExperimentCondition,
    dimensions: list[str],
    config: dict[str, Any],
    history_entries: list[dict[str, Any]],
    rng_seed: int | None = None,
    affective_dimensions: list[str] | None = None,
) -> tuple[dict[str, Any], str | None]:
    if condition == ExperimentCondition.C or condition == ExperimentCondition.B:
        return measured, None

    rng = random.Random(rng_seed)
    payload = copy.deepcopy(measured)
    affective_dimensions = affective_dimensions or config.get("affective_dimensions", [])
    bounds = config.get("bounds", {"min": 0.0, "max": 1.0})

    if condition == ExperimentCondition.D:
        payload["state"] = _random_state(dimensions, rng)
        payload["delta"] = {dimension: 0.0 for dimension in dimensions}
        if "affective_state" in payload and affective_dimensions:
            payload["affective_state"] = _random_state(affective_dimensions, rng)
            payload["affective_delta"] = {dimension: 0.0 for dimension in affective_dimensions}
        payload["source"] = "experimental_state_provider"
        return payload, "randomized"

    if condition == ExperimentCondition.E:
        stale_steps = int(config.get("condition_overrides", {}).get("stale_observations_back", 3))
        stale_entry = history_entries[-stale_steps] if len(history_entries) >= stale_steps else (
            history_entries[0] if history_entries else None
        )
        if stale_entry:
            payload["state"] = copy.deepcopy(stale_entry.get("state", payload.get("state", {})))
            payload["delta"] = copy.deepcopy(stale_entry.get("delta", {}))
            if "affective_state" in payload and affective_dimensions:
                payload["affective_state"] = copy.deepcopy(
                    stale_entry.get("affective_state", payload.get("affective_state", {}))
                )
                payload["affective_delta"] = copy.deepcopy(
                    stale_entry.get("affective_delta", payload.get("affective_delta", {}))
                )
        payload["source"] = "experimental_state_provider"
        return payload, "stale"

    if condition == ExperimentCondition.F:
        perturbation = config.get("condition_overrides", {}).get("perturbation", {})
        payload["state"] = _perturb_state(payload.get("state", {}), perturbation, bounds)
        if "affective_state" in payload and affective_dimensions:
            affective_perturbation = config.get("condition_overrides", {}).get("affective_perturbation", {})
            payload["affective_state"] = _perturb_state(
                payload.get("affective_state", {}),
                affective_perturbation,
                bounds,
            )
        payload["source"] = "experimental_state_provider"
        return payload, "perturbed"

    return measured, None


def hash_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
