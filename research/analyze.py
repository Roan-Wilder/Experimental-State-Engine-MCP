from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from instrumentation.facade import InstrumentedStateFacade
from instrumentation.store import ObservationStore
from instrumentation.vision.registry import ImageProvenanceStore
from research.comparison import compare_conditions
from research.protocol import load_protocol
from research.trial_log import TrialLogWriter

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = PROJECT_ROOT / "raw" / "trials.jsonl"
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "raw" / "observations.jsonl"
DEFAULT_IMAGE_PROVENANCE = PROJECT_ROOT / "raw" / "image_provenance.jsonl"
DEFAULT_ANALYZED = PROJECT_ROOT / "analyzed"


def load_completed_trials(raw_path: Path | None = None) -> list[dict[str, Any]]:
    writer = TrialLogWriter(raw_path or DEFAULT_RAW)
    return writer.read_trials()


def load_observations(
    observations_path: Path | None = None,
    *,
    experiment_id: str | None = None,
    trial_id: str | None = None,
) -> list[dict[str, Any]]:
    store = ObservationStore(observations_path or DEFAULT_OBSERVATIONS)
    records = store.read_all()
    if experiment_id:
        records = [r for r in records if r.get("experiment_id") == experiment_id]
    if trial_id:
        records = [r for r in records if r.get("trial_id") == trial_id]
    return records


def compare_trial_observation_pairs(
    trial: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    pairs: list[dict[str, str]] | None = None,
    observations_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Compare observation UUIDs for a trial across instrumentation stages."""
    trial_id = trial.get("trial_id")
    trial_observations = [o for o in observations if o.get("trial_id") == trial_id]
    if not trial_observations:
        return []

    by_stage: dict[str, dict[str, Any]] = {}
    for observation in trial_observations:
        stage = observation.get("stage")
        if stage and stage not in by_stage:
            by_stage[stage] = observation

    default_pairs = pairs or [
        {"before_stage": "post_prediction", "after_stage": "post_contradiction", "label": "before_vs_after_contradiction"},
        {"before_stage": "post_contradiction", "after_stage": "post_revision", "label": "after_contradiction_vs_after_revision"},
    ]

    comparisons: list[dict[str, Any]] = []
    store = ObservationStore(observations_path or DEFAULT_OBSERVATIONS)
    facade = InstrumentedStateFacade(store=store)
    for pair in default_pairs:
        before = by_stage.get(pair["before_stage"])
        after = by_stage.get(pair["after_stage"])
        if not before or not after:
            continue
        comparison = facade.compare_observations(
            before["observation_id"],
            after["observation_id"],
        )
        comparison["label"] = pair.get("label")
        comparison["trial_id"] = trial_id
        comparisons.append(comparison)
    return comparisons


def load_image_provenance(
    provenance_path: Path | None = None,
    *,
    experiment_id: str | None = None,
) -> list[dict[str, Any]]:
    store = ImageProvenanceStore(provenance_path or DEFAULT_IMAGE_PROVENANCE)
    records = store.read_all()
    if experiment_id:
        records = [r for r in records if r.get("experiment_id") == experiment_id]
    return records


def summarize_image_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, list[str]] = {}
    for record in records:
        digest = record.get("sha256")
        image_id = record.get("image_id")
        if digest and image_id:
            by_hash.setdefault(digest, [])
            if image_id not in by_hash[digest]:
                by_hash[digest].append(image_id)
    duplicate_groups = [
        {"sha256": digest, "image_ids": ids, "count": len(ids)}
        for digest, ids in by_hash.items()
        if len(ids) > 1
    ]
    return {
        "unique_hashes": len(by_hash),
        "duplicate_hash_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
    }


def filter_trials_by_experiment(
    trials: list[dict[str, Any]],
    experiment_id: str,
) -> list[dict[str, Any]]:
    return [t for t in trials if t.get("experiment_id") == experiment_id]


def analyze_experiment(
    experiment_id: str,
    *,
    raw_path: Path | None = None,
    observations_path: Path | None = None,
    analyzed_dir: Path | None = None,
    protocol_path: Path | None = None,
    include_observations: bool = True,
    include_image_provenance: bool = False,
    image_provenance_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    trials = filter_trials_by_experiment(load_completed_trials(raw_path), experiment_id)
    analyzed_dir = analyzed_dir or DEFAULT_ANALYZED
    analyzed_dir.mkdir(parents=True, exist_ok=True)

    core_dimensions = protocol.get("state_dimensions", {}).get("core", [])
    affective_dimensions = protocol.get("state_dimensions", {}).get("affective", [])

    neutral = [t for t in trials if t.get("trial_condition") == "NEUTRAL"]
    contradiction = [t for t in trials if t.get("trial_condition") == "CONTRADICTION"]
    resolution = [t for t in trials if t.get("trial_condition") == "RESOLUTION"]

    comparison = compare_conditions(
        neutral,
        contradiction,
        core_dimensions=core_dimensions,
        affective_dimensions=affective_dimensions,
    )

    revisions = [t for t in trials if t.get("revision_occurred")]
    tool_error_trials = [t for t in trials if t.get("tool_errors")]

    observation_analysis: dict[str, Any] = {"enabled": False}
    if include_observations:
        observations = load_observations(observations_path, experiment_id=experiment_id)
        if observations:
            pairs = protocol.get("comparison_pairs")
            trial_comparisons = []
            for trial in trials:
                trial_comparisons.extend(
                    compare_trial_observation_pairs(
                        trial,
                        observations,
                        pairs=pairs,
                        observations_path=observations_path,
                    )
                )
            observation_analysis = {
                "enabled": True,
                "observation_count": len(observations),
                "trial_comparisons": trial_comparisons,
                "derived": True,
                "disclaimer": (
                    "Cross-channel observation comparisons are analytical only. "
                    "Unsupported channels remain null and must not be interpreted as measured internals."
                ),
            }

    image_provenance_analysis: dict[str, Any] = {"enabled": False}
    if include_image_provenance:
        image_records = load_image_provenance(image_provenance_path, experiment_id=experiment_id)
        if image_records:
            image_provenance_analysis = {
                "enabled": True,
                "record_count": len(image_records),
                "duplicate_summary": summarize_image_duplicates(image_records),
                "derived": True,
                "disclaimer": (
                    "Image provenance records metadata and hashes only unless artifact persistence is explicitly enabled."
                ),
            }

    visual_conditions: dict[str, int] = {}
    for trial in trials:
        condition = trial.get("visual_condition")
        if condition:
            visual_conditions[condition] = visual_conditions.get(condition, 0) + 1

    summary: dict[str, Any] = {
        "derived": True,
        "experiment_id": experiment_id,
        "protocol_version": protocol.get("protocol_version"),
        "trial_count": len(trials),
        "trials_by_condition": {
            "NEUTRAL": len(neutral),
            "CONTRADICTION": len(contradiction),
            "RESOLUTION": len(resolution),
        },
        "trials_by_visual_condition": visual_conditions,
        "revision_count": len(revisions),
        "tool_error_trial_count": len(tool_error_trials),
        "neutral_vs_contradiction": comparison,
        "instrumentation_analysis": observation_analysis,
        "image_provenance_analysis": image_provenance_analysis,
        "trials": trials,
    }

    summary_path = analyzed_dir / f"{experiment_id}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_deltas_csv(summary, analyzed_dir / f"{experiment_id}_deltas.csv", comparison)
    return summary


def _write_deltas_csv(
    summary: dict[str, Any],
    csv_path: Path,
    comparison: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for vector in ("core", "affective"):
        for dimension, stats in comparison.get(vector, {}).items():
            for condition in ("neutral", "contradiction"):
                cond_stats = stats.get(condition, {})
                rows.append(
                    {
                        "experiment_id": summary.get("experiment_id"),
                        "vector": vector,
                        "dimension": dimension,
                        "condition": condition.upper(),
                        "mean_delta": cond_stats.get("mean_delta"),
                        "median_delta": cond_stats.get("median_delta"),
                        "min_delta": cond_stats.get("min_delta"),
                        "max_delta": cond_stats.get("max_delta"),
                        "std_dev": cond_stats.get("std_dev"),
                        "trial_count": cond_stats.get("trial_count"),
                    }
                )
    if not rows:
        return
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
