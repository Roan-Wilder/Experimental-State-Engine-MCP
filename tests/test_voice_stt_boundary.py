from __future__ import annotations

from pathlib import Path

import pytest

from voice.stt_adapter import NullSTTAdapter


def test_null_stt_not_implemented():
    adapter = NullSTTAdapter()
    adapter.initialize()
    with pytest.raises(NotImplementedError):
        adapter.transcribe(b"abc", mime_type="audio/wav")
    adapter.shutdown()
    status = adapter.status()
    assert status["implemented"] is False
    assert status["adapter"] == "null"


def test_stt_module_has_no_state_engine_imports():
    text = Path(__file__).resolve().parents[1].joinpath("voice", "stt_adapter.py").read_text(encoding="utf-8")
    assert "instrumentation.facade" not in text
    assert "state.experimental_provider" not in text
    assert "record_experiment_event" not in text
