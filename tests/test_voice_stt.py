from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from state.experimental_provider import ExperimentalStateProvider
from voice.audio_capture import AudioCaptureAdapter, NullAudioCaptureAdapter
from voice.stt_adapter import NullSTTAdapter, build_stt_adapter
from voice.stt_config import STTConfig, faster_whisper_available
from voice.stt_mock_adapter import MockSTTAdapter
from voice.stt_types import STT_DISCLAIMER
from voice.stt_wav import pcm16_to_wav_bytes, validate_wav_bytes
from voice.voice_input_service import VoiceInputService, reset_voice_input_service_for_tests


def _stt_config(tmp_path: Path, **overrides) -> STTConfig:
    base = STTConfig(
        enabled=True,
        backend="mock",
        model="base",
        model_path=None,
        device="cpu",
        compute_type="int8",
        language="en",
        sample_rate=16000,
        max_record_seconds=10.0,
        events_path=tmp_path / "stt_events.jsonl",
        temp_dir=tmp_path / "tmp_stt",
        persist_audio=False,
        allow_model_download=False,
        hardware_test=False,
    )
    for key, value in overrides.items():
        object.__setattr__(base, key, value)
    return base


def _fixture_wav(*, duration_seconds: float = 0.25, sample_rate: int = 16000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    pcm = struct.pack(f"<{frames}h", *([1000] * frames))
    return pcm16_to_wav_bytes(pcm, sample_rate=sample_rate, channels=1)


class MockCaptureAdapter(AudioCaptureAdapter):
    def __init__(self, wav_bytes: bytes, *, max_seconds: float = 10.0) -> None:
        self.wav_bytes = wav_bytes
        self.max_seconds = max_seconds
        self.shutdown_called = False

    def list_devices(self) -> list[dict[str, Any]]:
        return [{"index": 0, "name": "mock-mic", "max_input_channels": 1}]

    def capture(self, *, duration_seconds: float, device: int | None = None) -> bytes:
        if duration_seconds > self.max_seconds:
            raise ValueError("duration exceeds max")
        return self.wav_bytes

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def _reset_stt_singleton():
    reset_voice_input_service_for_tests()
    yield
    reset_voice_input_service_for_tests()


def test_null_stt_raises_not_implemented():
    adapter = NullSTTAdapter()
    adapter.initialize()
    with pytest.raises(NotImplementedError):
        adapter.transcribe(_fixture_wav(), mime_type="audio/wav")
    adapter.shutdown()


def test_stt_config_invalid_backend():
    config = STTConfig.from_env().resolve_paths()
    object.__setattr__(config, "backend", "unknown")
    error = config.validate()
    assert error is not None
    assert error["error"] == "invalid_backend"


def test_stt_config_missing_faster_whisper_dependency(monkeypatch):
    monkeypatch.setattr("voice.stt_config.faster_whisper_available", lambda: False)
    config = _stt_config(Path("."), backend="faster_whisper")
    error = config.validate()
    assert error is not None
    assert error["error"] == "missing_stt_dependencies"


def test_build_mock_stt_adapter():
    config = _stt_config(Path("."))
    adapter = build_stt_adapter(config)
    assert isinstance(adapter, MockSTTAdapter)


def test_mock_stt_transcription_result_fields(tmp_path: Path):
    config = _stt_config(tmp_path)
    adapter = MockSTTAdapter(config)
    adapter.initialize()
    wav = _fixture_wav()
    result = adapter.transcribe(wav, mime_type="audio/wav")
    assert result.text
    assert result.backend == "mock"
    assert result.audio_hash
    assert result.disclaimer == STT_DISCLAIMER
    forbidden = ("emotion_detected", "intent_detected", "consciousness_detected")
    payload = result.to_dict()
    for field in forbidden:
        assert field not in payload


def test_validate_wav_rejects_empty():
    with pytest.raises(ValueError):
        validate_wav_bytes(b"", expected_sample_rate=16000)


def test_validate_wav_accepts_fixture():
    props = validate_wav_bytes(_fixture_wav(), expected_sample_rate=16000)
    assert props["channels"] == 1
    assert props["duration_seconds"] > 0


def test_voice_input_service_transcribe_bytes_mock(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "exp-stt-001")
    config = _stt_config(tmp_path)
    service = VoiceInputService(config)
    result = service.transcribe_bytes(_fixture_wav(), source="file_bytes")
    assert result["ok"] is True
    transcription = result["transcription"]
    assert transcription["text"]
    assert transcription["experiment_id"] == "exp-stt-001"
    assert transcription["backend"] == "mock"
    events = (tmp_path / "stt_events.jsonl").read_text(encoding="utf-8").strip()
    event = json.loads(events)
    assert event["stt_backend"] == "mock"
    assert event["text_hash"]
    assert not list(tmp_path.glob("tmp_stt/*.wav"))


def test_voice_input_service_temp_cleanup_when_not_persisting(tmp_path: Path):
    config = _stt_config(tmp_path, persist_audio=False)
    service = VoiceInputService(config)
    service.transcribe_bytes(_fixture_wav(), source="file_bytes")
    assert not list((tmp_path / "tmp_stt").glob("*.wav"))


def test_voice_input_service_persist_audio_when_enabled(tmp_path: Path):
    config = _stt_config(tmp_path, persist_audio=True)
    service = VoiceInputService(config)
    service.transcribe_bytes(_fixture_wav(), source="file_bytes")
    assert list((tmp_path / "tmp_stt").glob("*.wav"))


def test_record_and_transcribe_uses_capture_adapter(tmp_path: Path):
    wav = _fixture_wav()
    capture = MockCaptureAdapter(wav)
    config = _stt_config(tmp_path)
    service = VoiceInputService(config, capture_adapter=capture)
    result = service.record_and_transcribe(duration_seconds=1.0)
    assert result["ok"] is True
    assert result["transcription"]["source"] == "microphone"


def test_capture_max_duration_enforced():
    capture = MockCaptureAdapter(_fixture_wav(), max_seconds=2.0)
    config = _stt_config(Path("."), max_record_seconds=2.0)
    adapter = capture
    with pytest.raises(ValueError):
        adapter.capture(duration_seconds=5.0, device=None)


def test_null_capture_adapter_lists_empty():
    adapter = NullAudioCaptureAdapter()
    assert adapter.list_devices() == []
    with pytest.raises(RuntimeError):
        adapter.capture(duration_seconds=1.0)


def test_stt_does_not_mutate_experimental_state(tmp_path: Path):
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(
            tmp_path / "snapshots"
        ),
    )
    before = provider.observe()
    config = _stt_config(tmp_path)
    service = VoiceInputService(config)
    service.transcribe_bytes(_fixture_wav(), source="file_bytes")
    after = provider.observe()
    assert before["state"] == after["state"]


def test_stt_modules_do_not_import_state_facade():
    voice_root = Path(__file__).resolve().parents[1] / "voice"
    forbidden = ("instrumentation.facade", "record_experiment_event", "observe_internal_state")
    for path in voice_root.rglob("*.py"):
        if path.name in {"service.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} must not reference {token}"


def test_clamp_duration_respects_max():
    config = _stt_config(Path("."), max_record_seconds=5.0)
    assert config.clamp_duration(10.0) == 5.0


def test_faster_whisper_backend_validation_skips_when_installed():
    config = _stt_config(Path("."), backend="faster_whisper")
    if faster_whisper_available():
        assert config.validate() is None
    else:
        assert config.validate() is not None
