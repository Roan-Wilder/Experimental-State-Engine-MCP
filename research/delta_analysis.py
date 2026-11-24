from __future__ import annotations

from typing import Any


def compute_dimension_delta(
    dimension: str,
    before: float,
    after: float,
) -> dict[str, Any]:
    absolute_delta = after - before
    percent_delta: float | None = None
    if before != 0.0:
        percent_delta = (absolute_delta / before) * 100.0
    return {
        "dimension": dimension,
        "before": before,
        "after": after,
        "absolute_delta": absolute_delta,
        "percent_delta": percent_delta,
    }


def compute_state_deltas(
    before_state: dict[str, float],
    after_state: dict[str, float],
    dimensions: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dimension in dimensions:
        before = float(before_state.get(dimension, 0.0))
        after = float(after_state.get(dimension, 0.0))
        results.append(compute_dimension_delta(dimension, before, after))
    return results


def compute_snapshot_transition_deltas(
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    core_dimensions: list[str],
    affective_dimensions: list[str],
) -> dict[str, Any]:
    return {
        "core": compute_state_deltas(
            before_snapshot.get("core_state", {}),
            after_snapshot.get("core_state", {}),
            core_dimensions,
        ),
        "affective": compute_state_deltas(
            before_snapshot.get("affective_state", {}),
            after_snapshot.get("affective_state", {}),
            affective_dimensions,
        ),
    }
