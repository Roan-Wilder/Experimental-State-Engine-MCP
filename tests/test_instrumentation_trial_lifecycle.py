from __future__ import annotations

from pathlib import Path

import pytest

from research.trial_runner import TrialRunner


def test_trial_emits_observation_id_chain(tmp_path: Path):
    runner = TrialRunner(
        experiment_id="lifecycle-test",
        raw_log_path=tmp_path / "trials.jsonl",
        observations_path=tmp_path / "observations.jsonl",
        master_seed=7,
        trials_per_condition=1,
    )
    completed = runner.run_batch()
    assert len(completed) == 3
    for trial in completed:
        observation_ids = trial.get("observation_ids", [])
        assert observation_ids
        assert len(observation_ids) == len(set(observation_ids))


def test_instrumentation_pilot_protocol_extends_base(tmp_path: Path):
    protocol_path = Path(__file__).resolve().parents[1] / "research" / "protocols" / "instrumentation_pilot_v1.yaml"
    runner = TrialRunner(
        experiment_id="pilot-test",
        protocol_path=protocol_path,
        raw_log_path=tmp_path / "trials.jsonl",
        observations_path=tmp_path / "observations.jsonl",
        master_seed=42,
        trials_per_condition=2,
    )
    completed = runner.run_batch()
    assert len(completed) == 6
    assert completed[0].get("instrumentation_schema_version") == "2.0.0"
