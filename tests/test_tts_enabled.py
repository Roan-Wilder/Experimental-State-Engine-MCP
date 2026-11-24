from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from voice.config import VoiceConfig, tts_enabled_from_env
from voice.service import get_voice_service, reset_voice_service_for_tests


def _load_mcp_module(name: str):
    module_path = Path(__file__).resolve().parents[1] / "mcp" / "internal_state_server.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_voice_singleton():
    reset_voice_service_for_tests()
    yield
    reset_voice_service_for_tests()


def test_tts_enabled_defaults_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TTS_ENABLED", raising=False)
    monkeypatch.delenv("QWEN_TTS_ENABLED", raising=False)
    assert tts_enabled_from_env() is False


def test_tts_enabled_master_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "1")
    monkeypatch.setenv("QWEN_TTS_ENABLED", "0")
    assert tts_enabled_from_env() is True
    monkeypatch.setenv("TTS_ENABLED", "0")
    assert tts_enabled_from_env() is False


def test_qwen_tts_enabled_legacy_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TTS_ENABLED", raising=False)
    monkeypatch.setenv("QWEN_TTS_ENABLED", "1")
    assert tts_enabled_from_env() is True


def test_mcp_does_not_register_speak_text_when_tts_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    monkeypatch.setenv("STT_ENABLED", "0")
    module = _load_mcp_module("internal_state_server_tts_off")
    assert not hasattr(module, "speak_text")
    assert not hasattr(module, "get_voice_status")
    assert hasattr(module, "observe_internal_state")


def test_mcp_research_tools_available_when_tts_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    module = _load_mcp_module("internal_state_server_research")
    assert callable(module.observe_internal_state)
    assert callable(module.capture_observation)
    assert callable(module.record_experiment_event)
    result = module.observe_internal_state()
    assert "state" in result or "ok" in result or isinstance(result, dict)


def test_get_voice_service_raises_when_tts_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    with pytest.raises(RuntimeError, match="TTS is disabled"):
        get_voice_service()


def test_mcp_does_not_initialize_voice_service_when_tts_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    with patch("voice.service.VoiceService") as mock_service:
        _load_mcp_module("internal_state_server_no_voice_init")
        mock_service.assert_not_called()


def test_mcp_does_not_start_sidecar_when_tts_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    with patch("voice.process_manager.VoiceProcessManager.ensure_sidecar_http") as mock_start:
        _load_mcp_module("internal_state_server_no_sidecar")
        mock_start.assert_not_called()


def test_stt_independent_when_tts_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("TTS_ENABLED", "0")
    monkeypatch.setenv("STT_ENABLED", "1")
    monkeypatch.setenv("STT_BACKEND", "mock")
    monkeypatch.setenv("STT_EVENTS_PATH", str(tmp_path / "stt_events.jsonl"))
    monkeypatch.setenv("STT_TEMP_DIR", str(tmp_path / "tmp_stt"))
    module = _load_mcp_module("internal_state_server_stt_only")
    assert not hasattr(module, "speak_text")
    assert hasattr(module, "get_stt_status")
    assert hasattr(module, "record_voice_input")


def test_mcp_registers_tts_tools_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("TTS_ENABLED", "1")
    monkeypatch.setenv("QWEN_TTS_BACKEND", "mock")
    monkeypatch.setenv("QWEN_TTS_MODEL_PATH", str(tmp_path / "talker.gguf"))
    monkeypatch.setenv("QWEN_TTS_CODEC_PATH", str(tmp_path / "codec.gguf"))
    monkeypatch.setenv("QWEN_TTS_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("QWEN_TTS_EVENTS_PATH", str(tmp_path / "voice_events.jsonl"))
    (tmp_path / "talker.gguf").write_bytes(b"t")
    (tmp_path / "codec.gguf").write_bytes(b"c")
    module = _load_mcp_module("internal_state_server_tts_on")
    assert hasattr(module, "speak_text")
    payload = module.speak_text(text="Hello")
    assert isinstance(payload, list)


def test_voice_config_enabled_follows_tts_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_ENABLED", "0")
    assert VoiceConfig.from_env().enabled is False
    monkeypatch.setenv("TTS_ENABLED", "1")
    assert VoiceConfig.from_env().enabled is True
