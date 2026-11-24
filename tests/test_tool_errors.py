from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from research.protocol import load_protocol
from research.randomization import generate_trial_params
from research.trial_runner import TrialRunner
from research.trial_types import TrialCondition


def test_invalid_event_aborts_trial(tmp_path: Path):
    runner = TrialRunner(
        experiment_id="test-tool-errors",
        raw_log_path=tmp_path / "trials.jsonl",
        master_seed=42,
        trials_per_condition=1,
        mode="simulation",
    )
    protocol = load_protocol()
    params = generate_trial_params(
        trial_index=0,
        master_seed=42,
        trial_condition=TrialCondition.NEUTRAL,
        protocol=protocol,
    )

    facade = runner._new_facade("trial-tool-error-test")
    provider = facade.provider

    def fail_on_neutral(event_type, metadata=None):
        if event_type == "neutral_information":
            return {
                "error": "unknown_event_type",
                "event_type": event_type,
                "allowed_event_types": [],
            }
        return provider.record_event(event_type, metadata)

    with patch.object(runner, "_new_facade", return_value=facade):
        with patch.object(facade, "record_experiment_event", side_effect=fail_on_neutral):
            trial = runner.run_single_trial(params)

    assert trial["outcome"] == "aborted_invalid_event"
    assert len(trial["tool_errors"]) == 1
    assert trial["tool_errors"][0]["error_type"] == "unknown_event_type"

    content = (tmp_path / "trials.jsonl").read_text(encoding="utf-8")
    assert "tool_error" in content
