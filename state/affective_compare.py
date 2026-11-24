from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DISCLAIMER = (
    "Analytical comparison of experimental affective observation variables only. "
    "This does not infer subjective emotional experience or neural activation changes."
)


def diff_affective_states(
    before: dict[str, float],
    after: dict[str, float],
    *,
    dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two affective observation vectors without causal inference."""
    all_dimensions = dimensions or sorted(set(before) | set(after))
    changes: dict[str, dict[str, float | str]] = {}
    unchanged: list[str] = []

    for dimension in all_dimensions:
        before_value = float(before.get(dimension, 0.0))
        after_value = float(after.get(dimension, 0.0))
        delta = after_value - before_value
        if abs(delta) < 1e-9:
            unchanged.append(dimension)
            continue
        direction = "increased" if delta > 0 else "decreased"
        changes[dimension] = {
            "before": before_value,
            "after": after_value,
            "delta": delta,
            "direction": direction,
            "summary": (
                f"The affective observation variable '{dimension}' {direction} by {abs(delta):.4f}."
            ),
        }

    return {
        "changes": changes,
        "unchanged": unchanged,
        "disclaimer": DISCLAIMER,
    }


def find_log_record(log_path: Path, reference: str) -> dict[str, Any] | None:
    """Find a JSONL record by transition_id or state_observation_id."""
    if not log_path.exists():
        return None

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("state_observation_id") == reference:
                return record
            metadata = record.get("metadata") or {}
            if metadata.get("transition_id") == reference:
                return record
            if record.get("event") and reference == record.get("timestamp"):
                return record
    return None


def extract_affective_from_record(record: dict[str, Any]) -> dict[str, float] | None:
    """Extract affective state from a log record or history entry."""
    if record.get("new_affective_state"):
        return dict(record["new_affective_state"])
    if record.get("affective_state"):
        return dict(record["affective_state"])
    metadata = record.get("metadata") or {}
    if metadata.get("new_affective_state"):
        return dict(metadata["new_affective_state"])
    return None
