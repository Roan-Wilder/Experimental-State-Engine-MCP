from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from voice.config import VoiceConfig
from voice.playback_adapter import LocalPlaybackAdapter, NullPlaybackAdapter, build_playback_adapter
from voice.service import VoiceService, reset_voice_service_for_tests
from voice.types import OutputMode


def _voice_config(tmp_path: Path, **overrides) -> VoiceConfig:
    model = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    model.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    base = VoiceConfig(
        enabled=True,
        backend="mock",
        model_path=model,
        codec_path=codec,
        binary_path=None,
        sidecar_url="http://127.0.0.1:8765",
        auto_start_sidecar=False,
        language="English",
        default_instruction="adult, neutral, clear",
        output_format="wav",
        max_text_length=500,
        timeout_seconds=30.0,
        persist_artifacts=True,
        artifact_root=tmp_path / "artifacts",
        events_path=tmp_path / "voice_events.jsonl",
        personas_path=tmp_path / "voice_personas.jsonl",
        benchmarks_path=tmp_path / "benchmarks.jsonl",
        persona_reference_text="Reference calibration sample.",
        output_mode=OutputMode.GENERATE_AND_RETURN.value,
        playback_enabled=False,
        ggml_backend=None,
        max_wav_bytes=50 * 1024 * 1024,
    )
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


@pytest.fixture(autouse=True)
def _reset_voice_singleton():
    reset_voice_service_for_tests()
    yield
    reset_voice_service_for_tests()


def test_null_playback_is_default():
    adapter = build_playback_adapter(enabled=False)
    assert isinstance(adapter, NullPlaybackAdapter)
    adapter.play_wav(b"", sample_rate=24000)


def test_generate_only_skips_audio_in_mcp_response(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="metadata only", output_mode=OutputMode.GENERATE_ONLY.value)
    assert result["ok"]
    assert result["include_audio"] is False
    formatted = service.format_mcp_response(result, audio_cls=object)
    assert isinstance(formatted, dict)


def test_generate_and_play_requires_playback_enabled(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path, playback_enabled=False))
    result = service.synthesize(
        text="play attempt",
        output_mode=OutputMode.GENERATE_AND_PLAY.value,
        playback=True,
    )
    assert result["error"] == "playback_disabled"


def test_local_playback_adapter_can_be_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    adapter = LocalPlaybackAdapter()
    called = {"count": 0}

    def _fake_play(wav_bytes: bytes, *, sample_rate: int) -> None:
        called["count"] += 1

    monkeypatch.setattr(adapter, "play_wav", _fake_play)
    service = VoiceService(_voice_config(tmp_path, playback_enabled=True))
    service._playback = adapter
    result = service.synthesize(
        text="play ok",
        output_mode=OutputMode.GENERATE_AND_PLAY.value,
        playback=True,
    )
    assert result["ok"]
    assert called["count"] == 1


def test_mcp_generate_only_returns_metadata_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("TTS_ENABLED", "1")
    monkeypatch.setenv("QWEN_TTS_ENABLED", "1")
    monkeypatch.setenv("QWEN_TTS_BACKEND", "mock")
    monkeypatch.setenv("QWEN_TTS_MODEL_PATH", str(tmp_path / "talker.gguf"))
    monkeypatch.setenv("QWEN_TTS_CODEC_PATH", str(tmp_path / "codec.gguf"))
    monkeypatch.setenv("QWEN_TTS_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("QWEN_TTS_EVENTS_PATH", str(tmp_path / "voice_events.jsonl"))
    (tmp_path / "talker.gguf").write_bytes(b"t")
    (tmp_path / "codec.gguf").write_bytes(b"c")
    module_path = Path(__file__).resolve().parents[1] / "mcp" / "internal_state_server.py"
    spec = importlib.util.spec_from_file_location("internal_state_server_playback", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.speak_text(text="Hello", output_mode="generate_only")
    assert isinstance(payload, dict)
    assert payload["ok"] is True
