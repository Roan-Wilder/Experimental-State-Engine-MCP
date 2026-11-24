from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voice.stt_config import STTConfig, faster_whisper_available, sounddevice_available
from voice.voice_input_service import VoiceInputService, reset_voice_input_service_for_tests


def _hardware_enabled() -> bool:
    return os.environ.get("STT_HARDWARE_TEST", "0") == "1"


def _require_hardware() -> STTConfig:
    if not _hardware_enabled():
        pytest.skip("Set STT_HARDWARE_TEST=1 to run STT hardware integration tests.")
    config = STTConfig.from_env().resolve_paths()
    if config.backend == "mock":
        pytest.fail("STT hardware tests must not use STT_BACKEND=mock.")
    if config.backend != "faster_whisper":
        pytest.fail(f"Unsupported STT hardware backend: {config.backend}")
    if not faster_whisper_available():
        pytest.fail('faster-whisper not installed. Install with: pip install -e ".[stt]"')
    if not sounddevice_available():
        pytest.fail('sounddevice not installed. Install with: pip install -e ".[stt]"')
    validation = config.validate()
    if validation is not None:
        pytest.fail(validation["message"])
    return config


@pytest.mark.integration
def test_hardware_microphone_enumeration():
    config = _require_hardware()
    service = VoiceInputService(config)
    devices = service.list_microphones()
    if not devices:
        pytest.fail("No microphone devices detected for STT hardware test.")
    assert devices[0]["index"] is not None


@pytest.mark.integration
def test_hardware_record_and_transcribe():
    reset_voice_input_service_for_tests()
    config = _require_hardware()
    artifact_root = Path(
        os.environ.get("STT_HARDWARE_ARTIFACT_ROOT", "tmp_stt/hardware_test")
    )
    if not artifact_root.is_absolute():
        artifact_root = (Path(__file__).resolve().parents[2] / artifact_root).resolve()
    events_path = artifact_root / "stt_events_hardware.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    config = STTConfig(
        **{
            **config.__dict__,
            "events_path": events_path,
            "temp_dir": artifact_root / "temp",
            "persist_audio": False,
        }
    )
    service = VoiceInputService(config)
    duration = min(5.0, config.max_record_seconds)
    result = service.record_and_transcribe(duration_seconds=duration, source="microphone")
    assert result.get("ok"), result
    transcription = result["transcription"]
    assert transcription["backend"] == "faster_whisper"
    assert transcription["processing_time_seconds"] > 0
    text = (transcription["text"] or "").strip()
    if not text:
        pytest.fail(
            "STT hardware test produced empty transcription. "
            "Speak clearly during the bounded recording window and re-run."
        )
    assert transcription["audio_hash"]
    assert transcription["device"] == config.device
    assert transcription["compute_type"] == config.compute_type
    event = json.loads(events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert event["stt_backend"] == "faster_whisper"
    assert not list((artifact_root / "temp").glob("*.wav")) if (artifact_root / "temp").exists() else True
    service.shutdown()
