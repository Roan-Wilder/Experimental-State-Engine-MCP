from __future__ import annotations

from pathlib import Path

import pytest

from experiments.conditions import ExperimentCondition, load_condition_context
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "test-exp")
    monkeypatch.setenv("AI_ECO_LAB_SESSION_ID", "test-session")
    context = load_condition_context()
    return ExperimentalStateProvider(
        condition_context=context,
        snapshots=SnapshotStore(tmp_path / "snapshots"),
    )


def test_snapshot_round_trip(provider: ExperimentalStateProvider, tmp_path: Path):
    provider.record_event("novel_information")
    saved = provider.save_snapshot("baseline")
    provider.record_event("contradiction")
    assert provider.current_state != saved["state"]

    restored = provider.restore_snapshot("baseline")
    assert restored["snapshot_id"] == "baseline"
    assert provider.current_state == saved["state"]


def test_snapshot_contains_schema_version(provider: ExperimentalStateProvider):
    payload = provider.save_snapshot("v1")
    assert payload["schema_version"] == "1.1.0"
    assert payload["experiment_id"] == "test-exp"
