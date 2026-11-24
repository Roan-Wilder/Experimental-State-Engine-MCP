from __future__ import annotations

import statistics
from typing import Any


DISCLAIMER = (
    "Descriptive statistics only. No statistical significance claims are made "
    "unless a validated test is explicitly implemented."
)


def _collect_dimension_deltas(
    trials: list[dict[str, Any]],
    vector_key: str,
    dimension: str,
    stage_from: str,
    stage_to: str,
) -> list[float]:
    values: list[float] = []
    for trial in trials:
        deltas = trial.get("stage_deltas", {})
        transition = deltas.get(f"{stage_from}->{stage_to}", {})
        for entry in transition.get(vector_key, []):
            if entry.get("dimension") == dimension:
                values.append(float(entry.get("absolute_delta", 0.0)))
    return values


def summarize_deltas(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "trial_count": 0,
            "mean_delta": None,
            "median_delta": None,
            "min_delta": None,
            "max_delta": None,
            "std_dev": None,
        }
    return {
        "trial_count": len(values),
        "mean_delta": statistics.mean(values),
        "median_delta": statistics.median(values),
        "min_delta": min(values),
        "max_delta": max(values),
        "std_dev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def compare_conditions(
    neutral_trials: list[dict[str, Any]],
    contradiction_trials: list[dict[str, Any]],
    *,
    core_dimensions: list[str],
    affective_dimensions: list[str],
    stage_from: str = "STATE_1",
    stage_to: str = "STATE_2",
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "stage_transition": f"{stage_from}->{stage_to}",
        "neutral_trial_count": len(neutral_trials),
        "contradiction_trial_count": len(contradiction_trials),
        "core": {},
        "affective": {},
    }
    for dimension in core_dimensions:
        neutral_vals = _collect_dimension_deltas(neutral_trials, "core", dimension, stage_from, stage_to)
        contradiction_vals = _collect_dimension_deltas(
            contradiction_trials, "core", dimension, stage_from, stage_to
        )
        comparison["core"][dimension] = {
            "neutral": summarize_deltas(neutral_vals),
            "contradiction": summarize_deltas(contradiction_vals),
        }
    for dimension in affective_dimensions:
        neutral_vals = _collect_dimension_deltas(neutral_trials, "affective", dimension, stage_from, stage_to)
        contradiction_vals = _collect_dimension_deltas(
            contradiction_trials, "affective", dimension, stage_from, stage_to
        )
        comparison["affective"][dimension] = {
            "neutral": summarize_deltas(neutral_vals),
            "contradiction": summarize_deltas(contradiction_vals),
        }
    return comparison
