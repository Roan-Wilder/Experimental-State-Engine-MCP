from __future__ import annotations

import json
from pathlib import Path

import pytest

from state.experimental_provider import ExperimentalStateProvider
from voice.config import VoiceConfig
from voice.persona import PERSONA_DISCLAIMER
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


def test_create_voice_persona_stage_a(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.create_voice_persona(
        name="lab_neutral",
        description="Neutral experimental voice",
        voice_instruction="adult, neutral, clear",
        seed=42,
    )
    assert result["ok"]
    persona = result["persona"]
    assert persona["version"] == 1
    assert persona["status"] == "active"
    assert persona["reference_audio_hash"]
    assert persona["persona_disclaimer"] == PERSONA_DISCLAIMER
    assert (tmp_path / "artifacts" / "personas" / persona["voice_id"] / "v1.wav").is_file()


def test_reuse_persona_via_voice_id_stage_b(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    created = service.create_voice_persona(
        name="reuse_me",
        description="Reuse test",
        voice_instruction="adult, calm, clear",
        seed=7,
    )
    voice_id = created["persona"]["voice_id"]
    spoken = service.synthesize(text="Consistency check.", voice_id=voice_id)
    assert spoken["ok"]
    assert spoken["metadata"]["voice_id"] == voice_id
    assert spoken["metadata"]["persona_version"] == 1


def test_regenerate_persona_supersedes_previous_version(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    created = service.create_voice_persona(
        name="versioned",
        description="Version test",
        voice_instruction="adult, neutral",
        seed=1,
    )
    voice_id = created["persona"]["voice_id"]
    regenerated = service.regenerate_voice_persona(
        voice_id,
        voice_instruction="adult, energetic, clear",
        seed=2,
    )
    assert regenerated["ok"]
    assert regenerated["persona"]["version"] == 2
    listed = service.list_voice_personas()
    versions = [p for p in listed["personas"] if p["voice_id"] == voice_id]
    assert any(p["status"] == "superseded" for p in versions)
    assert any(p["status"] == "active" and p["version"] == 2 for p in versions)


def test_get_and_list_voice_personas(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    service.create_voice_persona(name="one", description="d", voice_instruction="calm")
    listed = service.list_voice_personas(status="active")
    assert listed["count"] == 1
    voice_id = listed["personas"][0]["voice_id"]
    fetched = service.get_voice_persona(voice_id)
    assert fetched["ok"]
    assert fetched["persona"]["voice_id"] == voice_id


def test_persona_creation_logs_voice_event(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    service.create_voice_persona(name="evented", description="d", voice_instruction="calm")
    lines = (tmp_path / "voice_events.jsonl").read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[-1])
    assert event["type"] == "voice_persona_creation"
    assert event["voice_id"]


def test_persona_does_not_mutate_state_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(
            tmp_path / "snapshots"
        ),
    )
    before = provider.observe()
    service = VoiceService(_voice_config(tmp_path))
    created = service.create_voice_persona(name="iso", description="d", voice_instruction="calm")
    service.synthesize(text="hello", voice_id=created["persona"]["voice_id"])
    after = provider.observe()
    assert before["state"] == after["state"]
