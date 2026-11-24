from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from research.subject_protocol import (
    PROTOCOL_ID,
    load_research_subject_protocol,
)


def _load_mcp_module(name: str):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "mcp"
        / "internal_state_server.py"
    )
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_protocol_missing_is_reported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", raising=False)

    with pytest.raises(FileNotFoundError):
        load_research_subject_protocol()


def test_loader_reads_external_protocol_and_returns_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protocol = "External research protocol for testing.\n"
    protocol_file = tmp_path / "private_protocol.txt"
    protocol_file.write_text(protocol, encoding="utf-8")
    monkeypatch.setenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", str(protocol_file))

    payload = load_research_subject_protocol()

    assert payload["ok"] is True
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["text"] == protocol
    assert payload["sha256"] == hashlib.sha256(
        protocol.encode("utf-8")
    ).hexdigest()
    assert payload["path"] == "external_protocol"


def test_loader_sha256_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protocol_file = tmp_path / "private_protocol.txt"
    protocol_file.write_text("Stable test protocol.\n", encoding="utf-8")
    monkeypatch.setenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", str(protocol_file))

    first = load_research_subject_protocol()["sha256"]
    second = load_research_subject_protocol()["sha256"]

    assert first == second


def test_empty_protocol_is_rejected(tmp_path: Path):
    protocol_file = tmp_path / "empty_protocol.txt"
    protocol_file.write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_research_subject_protocol(protocol_file)


def test_mcp_tool_reports_unconfigured_protocol_safely(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", raising=False)
    module = _load_mcp_module("internal_state_server_protocol_missing")

    payload = module.get_research_subject_protocol()

    assert payload["ok"] is False
    assert payload["error"] == "research_subject_protocol_unavailable"
    assert "AI_ECO_LAB_RESEARCH_PROTOCOL_PATH" in payload["message"]
    assert "C:\\" not in payload["message"]


def test_mcp_tool_returns_configured_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protocol_file = tmp_path / "private_protocol.txt"
    protocol_file.write_text("Configured protocol.\n", encoding="utf-8")
    monkeypatch.setenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", str(protocol_file))

    module = _load_mcp_module("internal_state_server_protocol_configured")
    payload = module.get_research_subject_protocol()

    assert payload["ok"] is True
    assert payload["text"] == "Configured protocol.\n"
    assert payload["sha256"]


def test_mcp_tool_available_when_voice_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    protocol_file = tmp_path / "private_protocol.txt"
    protocol_file.write_text("Voice-disabled test protocol.\n", encoding="utf-8")

    monkeypatch.setenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH", str(protocol_file))
    monkeypatch.setenv("TTS_ENABLED", "0")
    monkeypatch.setenv("STT_ENABLED", "0")

    module = _load_mcp_module("internal_state_server_protocol_voice_off")

    assert hasattr(module, "get_research_subject_protocol")
    assert not hasattr(module, "speak_text")
    assert module.get_research_subject_protocol()["ok"] is True


def test_protocol_loader_no_state_imports():
    source = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "subject_protocol.py"
    ).read_text(encoding="utf-8")

    forbidden = (
        "instrumentation.facade",
        "record_experiment_event",
        "state.engine",
    )
    for token in forbidden:
        assert token not in source