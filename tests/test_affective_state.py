from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.conditions import load_condition_context
from experiments.logger import ExperimentLogger
from state.engine import apply_transition, load_config, normalize_state, replay_transitions
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "exp-002-test")
    monkeypatch.setenv("AI_ECO_LAB_SESSION_ID", "exp-002-session")
    log_dir = tmp_path / "experiments"
    context = load_condition_context()
    return ExperimentalStateProvider(
        condition_context=context,
        snapshots=SnapshotStore(tmp_path / "snapshots"),
        logger=ExperimentLogger(context, log_dir=log_dir),
    )


def test_initial_affective_state(config):
    affective_dims = config["affective_dimensions"]
    assert len(affective_dims) == 12
    initial = normalize_state(config["affective_initial_values"], affective_dims)
    assert all(initial[dimension] == 0.0 for dimension in affective_dims)


def test_affective_range_enforcement(config):
    affective_dims = config["affective_dimensions"]
    current = {dimension: 0.99 for dimension in affective_dims}
    result = apply_transition(
        current,
        "threatening_information",
        config,
        dimensions=affective_dims,
    )
    for value in result.new_state.values():
        assert 0.0 <= value <= 1.0


def test_positive_affective_delta(provider: ExperimentalStateProvider):
    result = provider.record_event("positive_social_feedback")
    assert result["affective_delta"]["affection"] == pytest.approx(0.15)
    assert result["new_affective_state"]["affection"] > result["previous_affective_state"]["affection"]


def test_negative_affective_delta(config):
    affective_dims = config["affective_dimensions"]
    current = normalize_state(config["affective_initial_values"], affective_dims)
    current["frustration"] = 0.30
    result = apply_transition(current, "resolution", config, dimensions=affective_dims)
    assert result.delta["frustration"] == pytest.approx(-0.10)
    assert result.new_state["frustration"] < current["frustration"]


def test_multiple_simultaneous_affective_deltas(provider: ExperimentalStateProvider):
    result = provider.record_event("contradictory_information")
    assert result["affective_delta"]["frustration"] == pytest.approx(0.20)
    assert result["affective_delta"]["curiosity"] == pytest.approx(0.10)
    assert result["delta"]["uncertainty"] == pytest.approx(0.15)


def test_affective_clamping(config):
    affective_dims = config["affective_dimensions"]
    current = {dimension: 0.95 for dimension in affective_dims}
    result = apply_transition(current, "threatening_information", config, dimensions=affective_dims)
    for value in result.new_state.values():
        assert value <= 1.0


def test_deterministic_replay(config):
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "exp-002-stimulus-sequence.yaml"
    with fixture_path.open("r", encoding="utf-8") as handle:
        fixture = yaml.safe_load(handle)
    events = fixture["events"]
    initial_core = dict(config["initial_values"])
    initial_affective = dict(config["affective_initial_values"])

    first_core, first_affective = replay_transitions(initial_core, initial_affective, events, config)
    second_core, second_affective = replay_transitions(initial_core, initial_affective, events, config)
    assert first_core == second_core
    assert first_affective == second_affective


def test_jsonl_logging_includes_affective_fields(provider: ExperimentalStateProvider):
    provider.record_event("positive_social_feedback")
    content = provider.logger.log_path.read_text(encoding="utf-8")
    assert "new_affective_state" in content
    assert "affective_delta" in content
    assert "record_experiment_event" in content


def test_observe_includes_affective_state_without_altering_core(provider: ExperimentalStateProvider, config):
    provider.record_event("novel_information")
    payload = provider.observe()
    assert "affective_state" in payload
    assert payload["state"]["novelty"] == pytest.approx(0.45)
    assert len(payload["state"]) == len(config["dimensions"])


def test_core_only_event_leaves_affective_unchanged(provider: ExperimentalStateProvider):
    before = dict(provider.affective_state)
    provider.record_event("tool_call")
    after = dict(provider.affective_state)
    assert before == after


def test_neutral_information_control_condition(provider: ExperimentalStateProvider):
    before = dict(provider.affective_state)
    result = provider.record_event("neutral_information")
    assert result["affective_delta"] == {dimension: 0.0 for dimension in provider.affective_dimensions}
    assert dict(provider.affective_state) == before


def test_compare_affective_states_output_format(provider: ExperimentalStateProvider):
    provider.record_event("positive_social_feedback")
    after_ref = "current"
    before_ref = "initial"
    comparison = provider.compare_affective_states(before_ref, after_ref)
    assert "changes" in comparison
    assert "unchanged" in comparison
    assert "disclaimer" in comparison
    assert "subjective emotional" in comparison["disclaimer"]
    if comparison["changes"]:
        for entry in comparison["changes"].values():
            assert "summary" in entry
            assert "became" not in entry["summary"].lower()
            assert "direction" in entry
