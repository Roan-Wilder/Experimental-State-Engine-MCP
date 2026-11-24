from __future__ import annotations

from pathlib import Path

import pytest

from experiments.conditions import (
    ExperimentCondition,
    apply_observation_override,
    load_condition_context,
    tool_enabled,
)
from state.engine import load_config
from state.experimental_provider import ExperimentalStateProvider


@pytest.fixture
def config():
    return load_config()


def test_condition_b_disables_history_and_reflect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "B")
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "cond-b")
    provider = ExperimentalStateProvider(
        condition_context=load_condition_context(),
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(tmp_path / "snapshots"),
    )
    assert "observation_id" in provider.observe()
    assert provider.history()["error"] == "tool_disabled_for_condition"
    assert provider.reflect_context()["error"] == "tool_disabled_for_condition"


def test_condition_d_randomizes_state(config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_RNG_SEED", "42")
    measured = {
        "state": {dimension: 0.5 for dimension in config["dimensions"]},
        "delta": {dimension: 0.0 for dimension in config["dimensions"]},
    }
    overridden, override = apply_observation_override(
        measured,
        condition=ExperimentCondition.D,
        dimensions=config["dimensions"],
        config=config,
        history_entries=[],
        rng_seed=42,
    )
    assert override == "randomized"
    assert overridden["state"] != measured["state"]


def test_condition_f_perturbs_state(config):
    measured = {
        "state": {dimension: 0.5 for dimension in config["dimensions"]},
        "delta": {dimension: 0.0 for dimension in config["dimensions"]},
    }
    overridden, override = apply_observation_override(
        measured,
        condition=ExperimentCondition.F,
        dimensions=config["dimensions"],
        config=config,
        history_entries=[],
    )
    assert override == "perturbed"
    assert overridden["state"]["activation"] > measured["state"]["activation"]


def test_tool_enabled_matrix():
    assert tool_enabled("observe_internal_state", ExperimentCondition.B) is True
    assert tool_enabled("observe_state_history", ExperimentCondition.B) is False
    assert tool_enabled("reflect_on_internal_state", ExperimentCondition.C) is True


def test_record_event_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "log-test")
    provider = ExperimentalStateProvider(
        condition_context=load_condition_context(),
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(tmp_path / "snapshots"),
    )
    provider.record_event("user_message")
    log_path = tmp_path.parent / "experiments" / "log-test.jsonl"
    # logger uses project data dir; verify through provider logger path
    assert provider.logger.log_path.exists()
    content = provider.logger.log_path.read_text(encoding="utf-8")
    assert "record_experiment_event" in content
