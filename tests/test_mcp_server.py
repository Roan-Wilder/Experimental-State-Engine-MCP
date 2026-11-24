from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from state.experimental_provider import ExperimentalStateProvider


def test_provider_observe_returns_null_interpretation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(tmp_path / "snapshots"),
    )
    payload = provider.observe()
    assert payload["interpretation"] is None
    assert "state" in payload
    assert "activation" in payload["state"]


def test_reflect_keeps_engine_interpretation_null(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(tmp_path / "snapshots"),
    )
    payload = provider.reflect_context()
    assert payload["engine_interpretation"] is None
    assert payload["interpretation_source"] == "model"


def test_malformed_event_type_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(tmp_path / "snapshots"),
    )
    result = provider.record_event("definitely_not_a_real_event")
    assert result["error"] == "unknown_event_type"


def test_importing_mcp_module_keeps_stdout_clean(monkeypatch: pytest.MonkeyPatch):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "mcp" / "internal_state_server.py"
        spec = importlib.util.spec_from_file_location("internal_state_server", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "observe_internal_state")
    assert stdout.getvalue() == ""
