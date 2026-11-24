from __future__ import annotations

from pathlib import Path

import pytest

from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


def test_invalid_metadata_rejected_without_state_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(snapshots=SnapshotStore(tmp_path / "snapshots"))
    before = dict(provider.current_state)
    result = provider.record_event(
        "neutral_information",
        metadata={"model_response": "<tool_call>{\"name\": \"observe\"}</tool_call>"},
    )
    assert result["error"] == "invalid_metadata"
    assert dict(provider.current_state) == before
