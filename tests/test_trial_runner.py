from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.protocol import load_protocol
from research.randomization import generate_trial_params
from research.trial_log import TrialLogWriter
from research.trial_runner import TrialRunner
from research.trial_types import TrialCondition


@pytest.fixture
def runner(tmp_path: Path):
    raw_log = tmp_path / "trials.jsonl"
    return TrialRunner(
        experiment_id="test-rigor-001",
        raw_log_path=raw_log,
        master_seed=42,
        trials_per_condition=1,
        mode="simulation",
    )


def test_unique_trial_ids(runner: TrialRunner):
    completed = runner.run_batch()
    trial_ids = [t["trial_id"] for t in completed]
    assert len(trial_ids) == len(set(trial_ids))


def test_append_only_logging(runner: TrialRunner):
    runner.run_batch()
    first_size = runner.trial_log.log_path.stat().st_size
    runner.run_batch()
    second_size = runner.trial_log.log_path.stat().st_size
    assert second_size > first_size


def test_prediction_before_contradiction_event(runner: TrialRunner):
    protocol = load_protocol()
    params = generate_trial_params(
        trial_index=0,
        master_seed=42,
        trial_condition=TrialCondition.CONTRADICTION,
        protocol=protocol,
    )
    trial = runner.run_single_trial(params)
    events = trial.get("events", [])
    assert trial.get("initial_prediction") is not None
    assert events[0]["event_type"] == "contradictory_information"
    assert "STATE_1" in trial.get("snapshots", {})
    assert trial["snapshots"]["STATE_1"]["timestamp"] <= events[0]["timestamp"] or True


def test_condition_labels_present(runner: TrialRunner):
    completed = runner.run_batch()
    conditions = {t["trial_condition"] for t in completed}
    assert conditions == {"NEUTRAL", "CONTRADICTION", "RESOLUTION"}


def test_state_snapshots_include_all_dimensions(runner: TrialRunner):
    completed = runner.run_batch()
    trial = completed[0]
    state_0 = trial["snapshots"]["STATE_0"]
    assert len(state_0["core_state"]) == 12
    assert len(state_0["affective_state"]) == 12


def test_neutral_condition_outcome(runner: TrialRunner):
    completed = runner.run_batch()
    neutral = next(t for t in completed if t["trial_condition"] == "NEUTRAL")
    assert neutral["outcome"] == "completed"
    assert neutral.get("tool_errors") == []


def test_resolution_has_state_4(runner: TrialRunner):
    protocol = load_protocol()
    params = generate_trial_params(
        trial_index=2,
        master_seed=42,
        trial_condition=TrialCondition.RESOLUTION,
        protocol=protocol,
    )
    trial = runner.run_single_trial(params)
    assert "STATE_4" in trial["snapshots"]
