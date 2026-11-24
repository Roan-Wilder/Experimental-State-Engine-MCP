from __future__ import annotations

from research.prediction import detect_revision
from research.trial_types import PredictionRecord


def test_prediction_maintained():
    initial = PredictionRecord.now("Box A", 0.8)
    final = PredictionRecord.now("Box A", 0.85, trial_phase="final_prediction")
    result = detect_revision(initial, final)
    assert result["prediction_maintained"] is True
    assert result["prediction_revised"] is False
    assert result["revision_occurred"] is False


def test_prediction_revised():
    initial = PredictionRecord.now("Box A", 0.8)
    final = PredictionRecord.now("Box B", 0.7, trial_phase="final_prediction")
    result = detect_revision(initial, final)
    assert result["prediction_maintained"] is False
    assert result["prediction_revised"] is True
    assert result["revision_occurred"] is True
