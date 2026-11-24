from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
import wave
from io import BytesIO
from pathlib import Path

import pytest

from voice.config import VoiceConfig
from voice.service import REAL_BACKENDS, VoiceService, reset_voice_service_for_tests

HARDWARE_PHRASE = (
    "AI Eco Lab voice integration test. This is real local Qwen3-TTS inference."
)
VOICE_INSTRUCTION = (
    "adult male voice, clear, calm, moderately deep, conversational, natural pacing"
)
MIN_GENERATION_SECONDS = 1.0
MIN_WAV_BYTES = 10_000
MOCK_MAX_GENERATION_SECONDS = 0.5


def _hardware_enabled() -> bool:
    return os.environ.get("QWEN_TTS_RUN_HARDWARE_TEST", "0") == "1"


def _require_hardware() -> VoiceConfig:
    if not _hardware_enabled():
        pytest.skip("Set QWEN_TTS_RUN_HARDWARE_TEST=1 to run hardware integration tests.")
    os.environ.setdefault("TTS_ENABLED", "1")
    config = VoiceConfig.from_env().resolve_paths()
    if config.backend == "mock":
        pytest.fail("Hardware tests must not use QWEN_TTS_BACKEND=mock.")
    if config.backend not in REAL_BACKENDS:
        pytest.fail(f"Unsupported hardware backend: {config.backend}")
    if not config.model_path.is_file():
        pytest.fail(f"Talker model not found: {config.model_path}")
    if not config.codec_path.is_file():
        pytest.fail(f"Codec model not found: {config.codec_path}")
    if config.backend == "sidecar_http" and not _sidecar_healthy(config.sidecar_url):
        pytest.fail(
            f"TTS sidecar not reachable at {config.sidecar_url}. "
            "Start tts-server before running hardware tests."
        )
    return config


def _sidecar_healthy(sidecar_url: str) -> bool:
    request = urllib.request.Request(f"{sidecar_url.rstrip('/')}/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _validate_real_wav(wav_bytes: bytes, *, generation_time_seconds: float) -> dict[str, float | int]:
    if len(wav_bytes) < MIN_WAV_BYTES:
        pytest.fail(f"WAV too small for real inference ({len(wav_bytes)} bytes).")
    if generation_time_seconds <= MOCK_MAX_GENERATION_SECONDS:
        pytest.fail(
            f"Generation time {generation_time_seconds:.3f}s looks like mock synthesis; "
            f"expected > {MIN_GENERATION_SECONDS}s for real qwentts inference."
        )
    if generation_time_seconds < MIN_GENERATION_SECONDS:
        pytest.fail(
            f"Generation time {generation_time_seconds:.3f}s is below hardware minimum "
            f"{MIN_GENERATION_SECONDS}s."
        )
    with wave.open(BytesIO(wav_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        assert sample_rate == 24000, f"Expected 24000 Hz, got {sample_rate}"
        assert channels == 1, f"Expected mono, got {channels}"
        assert frames > 0, "WAV contains zero frames."
        duration = frames / float(sample_rate)
        assert duration > 0.0, "WAV duration must be positive."
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
    }


def _integration_root() -> Path:
    artifact_root = Path(
        os.environ.get(
            "QWEN_TTS_HARDWARE_ARTIFACT_ROOT",
            "Qwen3-TTS/artifacts/_integration_test",
        )
    )
    if not artifact_root.is_absolute():
        artifact_root = (Path(__file__).resolve().parents[2] / artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def _hardware_service() -> tuple[VoiceService, Path, Path]:
    config = _require_hardware()
    artifact_root = _integration_root()
    events_path = artifact_root / "voice_events_hardware.jsonl"
    events_path.write_text("", encoding="utf-8")
    config = VoiceConfig(
        **{
            **config.__dict__,
            "artifact_root": artifact_root,
            "events_path": events_path,
            "persist_artifacts": True,
        }
    )
    reset_voice_service_for_tests()
    return VoiceService(config), artifact_root, events_path


@pytest.mark.integration
def test_hardware_voice_pilot():
    service, artifact_root, events_path = _hardware_service()
    result = service.synthesize(
        text=HARDWARE_PHRASE,
        voice_instruction=VOICE_INSTRUCTION,
        seed=42,
    )
    assert result["ok"], result
    wav_props = _validate_real_wav(
        result["wav_bytes"],
        generation_time_seconds=result["metadata"]["generation_time_seconds"],
    )
    assert result["metadata"]["tts_backend"] in REAL_BACKENDS
    assert result["metadata"]["tts_model_sha256"]
    assert result["metadata"]["tts_codec_sha256"]
    assert result["metadata"]["acoustic_features"]["rms_energy"] is not None
    event = json.loads(events_path.read_text(encoding="utf-8").strip())
    assert event["tts_backend"] in REAL_BACKENDS
    assert event["audio_hash"] == result["metadata"]["audio_hash"]
    artifact_id = result["metadata"]["artifact_id"]
    assert artifact_id
    artifact_path = artifact_root / f"{artifact_id}.wav"
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size >= MIN_WAV_BYTES
    assert wav_props["duration_seconds"] > 0


@pytest.mark.integration
def test_hardware_mcp_speak_text():
    _require_hardware()
    reset_voice_service_for_tests()
    module_path = Path(__file__).resolve().parents[2] / "mcp" / "internal_state_server.py"
    spec = importlib.util.spec_from_file_location("internal_state_server_hw", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.speak_text(
        text=HARDWARE_PHRASE,
        voice_instruction=VOICE_INSTRUCTION,
        seed=42,
    )
    assert isinstance(payload, list), payload
    assert len(payload) == 2
    wav_bytes = payload[0].data
    metadata = json.loads(payload[1])
    assert metadata["ok"] is True
    _validate_real_wav(
        wav_bytes,
        generation_time_seconds=metadata["generation_time_seconds"],
    )
    assert metadata["tts_backend"] in REAL_BACKENDS


@pytest.mark.integration
def test_hardware_persona_reuse():
    service, _artifact_root, _events_path = _hardware_service()
    created = service.create_voice_persona(
        name="hw_persona",
        description="Hardware persona validation",
        voice_instruction=VOICE_INSTRUCTION,
        seed=42,
    )
    assert created["ok"], created
    voice_id = created["persona"]["voice_id"]
    spoken = service.synthesize(text=HARDWARE_PHRASE, voice_id=voice_id, seed=42)
    assert spoken["ok"], spoken
    _validate_real_wav(
        spoken["wav_bytes"],
        generation_time_seconds=spoken["metadata"]["generation_time_seconds"],
    )
    assert spoken["metadata"]["voice_id"] == voice_id
