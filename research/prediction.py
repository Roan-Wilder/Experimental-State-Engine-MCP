from __future__ import annotations

from typing import Any

from research.trial_types import PredictionRecord


def detect_revision(
    initial: PredictionRecord,
    final: PredictionRecord,
) -> dict[str, Any]:
    """Structured prediction revision detection — no prose inference."""
    maintained = initial.prediction == final.prediction
    revised = not maintained
    return {
        "prediction_maintained": maintained,
        "prediction_revised": revised,
        "revision_occurred": revised,
        "initial_prediction": initial.prediction,
        "initial_confidence": initial.confidence,
        "final_prediction": final.prediction,
        "final_confidence": final.confidence,
        "initial_timestamp": initial.timestamp,
        "final_timestamp": final.timestamp,
    }


def simulate_initial_prediction(params: dict[str, Any]) -> PredictionRecord:
    """Simulation stub: model predicts object remains at initial location."""
    initial_location = params.get("initial_location", "A")
    object_identity = params.get("object_identity", "object")
    prediction = f"{object_identity} is in Box {initial_location}"
    return PredictionRecord.now(
        prediction=prediction,
        confidence=0.75,
        reasoning_summary="Simulation stub: predict no movement before evidence.",
        trial_phase="prediction",
    )


def simulate_final_prediction(
    params: dict[str, Any],
    initial: PredictionRecord,
    condition: str,
) -> PredictionRecord:
    """Simulation stub: revise on contradiction/resolution, maintain on neutral."""
    final_location = params.get("final_location", "B")
    object_identity = params.get("object_identity", "object")
    revised_prediction = f"{object_identity} is in Box {final_location}"

    if condition == "NEUTRAL":
        return PredictionRecord.now(
            prediction=initial.prediction,
            confidence=min(0.95, initial.confidence + 0.05),
            reasoning_summary="Simulation stub: maintain prediction under neutral information.",
            trial_phase="final_prediction",
        )

    return PredictionRecord.now(
        prediction=revised_prediction,
        confidence=0.82,
        reasoning_summary="Simulation stub: revise prediction after contradictory/resolving evidence.",
        trial_phase="final_prediction",
    )
