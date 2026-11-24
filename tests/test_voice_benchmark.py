from __future__ import annotations

import json
from pathlib import Path

from voice.benchmark import run_benchmark
from voice.config import VoiceConfig
from voice.types import OutputMode


def _voice_config(tmp_path: Path) -> VoiceConfig:
    model = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    model.write_bytes(b"talker")
    codec.write_bytes(b"codec")
    return VoiceConfig(
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
        benchmarks_path=tmp_path / "benchmarks" / "voice_tts.jsonl",
        persona_reference_text="Reference calibration sample.",
        output_mode=OutputMode.GENERATE_AND_RETURN.value,
        playback_enabled=False,
        ggml_backend=None,
        max_wav_bytes=50 * 1024 * 1024,
    )


def test_benchmark_writes_mock_backend_row(tmp_path: Path):
    config = _voice_config(tmp_path)
    results = run_benchmark(config=config, backends=["mock"], seed=42)
    assert len(results) == 1
    assert results[0]["backend"] == "mock"
    assert results[0]["synthesis_seconds"] is not None
    lines = config.benchmarks_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[-1])
    assert row["backend"] == "mock"
    assert row["seed"] == 42


def test_benchmark_does_not_write_observations(tmp_path: Path):
    config = _voice_config(tmp_path)
    observations = tmp_path / "observations.jsonl"
    observations.write_text('{"existing": true}\n', encoding="utf-8")
    run_benchmark(config=config, backends=["mock"])
    assert observations.read_text(encoding="utf-8") == '{"existing": true}\n'
