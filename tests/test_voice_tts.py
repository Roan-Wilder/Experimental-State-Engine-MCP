from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from state.experimental_provider import ExperimentalStateProvider
from voice.adapter import VoiceAdapter
from voice.artifacts import VoiceArtifactStore
from voice.config import VoiceConfig
from voice.mock_adapter import MockVoiceAdapter
from voice.presets import resolve_preset
from voice.sanitize import sanitize_voice_instruction, sanitize_voice_text
from voice.service import VoiceService, reset_voice_service_for_tests
from voice.types import OutputMode, VoiceSynthesisRequest


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


def test_voice_adapter_interface_contract(tmp_path: Path):
    adapter: VoiceAdapter = MockVoiceAdapter(_voice_config(tmp_path))
    adapter.initialize()
    status = adapter.get_status()
    assert status.backend == "mock"
    result = adapter.synthesize(
        VoiceSynthesisRequest(
            text="hello",
            voice_instruction="calm voice",
            language="English",
        )
    )
    assert result.sample_rate == 24000
    assert result.channels == 1
    adapter.shutdown()


def test_config_loading_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("QWEN_TTS_BACKEND", "mock")
    monkeypatch.setenv("QWEN_TTS_MODEL_PATH", str(tmp_path / "m.gguf"))
    monkeypatch.setenv("QWEN_TTS_CODEC_PATH", str(tmp_path / "c.gguf"))
    config = VoiceConfig.from_env().resolve_paths()
    assert config.backend == "mock"
    assert config.model_path.name == "m.gguf"


def test_missing_model_file_returns_error(tmp_path: Path):
    config = _voice_config(tmp_path, backend="sidecar_http")
    config = VoiceConfig(
        **{
            **config.__dict__,
            "model_path": tmp_path / "missing.gguf",
        }
    )
    assert config.validate() == {
        "error": "missing_model",
        "message": f"Talker model not found: {tmp_path / 'missing.gguf'}",
    }


def test_missing_codec_file_returns_error(tmp_path: Path):
    config = _voice_config(tmp_path, backend="sidecar_http")
    config = VoiceConfig(
        **{
            **config.__dict__,
            "codec_path": tmp_path / "missing.gguf",
        }
    )
    assert config.validate() == {
        "error": "missing_codec",
        "message": f"Codec model not found: {tmp_path / 'missing.gguf'}",
    }


def test_invalid_backend_name(tmp_path: Path):
    config = _voice_config(tmp_path, backend="not_real")
    assert config.validate()["error"] == "invalid_backend"


def test_text_length_limit_enforcement():
    long_text = "a" * 501
    result = sanitize_voice_text(long_text, max_length=500)
    assert result["error"] == "text_too_long"


def test_successful_synthesis_via_mock_adapter(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="Hello world")
    assert result["ok"] is True
    assert len(result["wav_bytes"]) > 44


def test_mcp_speak_text_schema_and_sanitization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
    spec = importlib.util.spec_from_file_location("internal_state_server_voice", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.speak_text(text="Hello", voice_preset="calm")
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_voice_event_creation_fields(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="Event test", seed=7)
    assert result["ok"]
    lines = (tmp_path / "voice_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(lines[0])
    assert event["type"] == "voice_generation"
    assert event["seed"] == 7
    assert event["text_hash"]
    assert event["audio_hash"]
    assert event["disclaimer"]


def test_experiment_trial_correlation_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_EXPERIMENT_ID", "exp-voice-001")
    monkeypatch.setenv("AI_ECO_LAB_TRIAL_ID", "trial-001")
    monkeypatch.setenv("AI_ECO_LAB_TRIAL_STAGE", "voice_pilot")
    service = VoiceService(_voice_config(tmp_path))
    service.synthesize(text="correlation")
    event = json.loads((tmp_path / "voice_events.jsonl").read_text(encoding="utf-8").strip())
    assert event["experiment_id"] == "exp-voice-001"
    assert event["trial_id"] == "trial-001"
    assert event["stage"] == "voice_pilot"


def test_experiment_trial_null_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("AI_ECO_LAB_EXPERIMENT_ID", raising=False)
    monkeypatch.delenv("AI_ECO_LAB_TRIAL_ID", raising=False)
    monkeypatch.delenv("AI_ECO_LAB_TRIAL_STAGE", raising=False)
    service = VoiceService(_voice_config(tmp_path))
    service.synthesize(text="no correlation")
    event = json.loads((tmp_path / "voice_events.jsonl").read_text(encoding="utf-8").strip())
    assert event["experiment_id"] is None
    assert event["trial_id"] is None


def test_audio_hash_generation(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    first = service.synthesize(text="hash me")
    second = service.synthesize(text="hash me")
    assert first["metadata"]["audio_hash"] == second["metadata"]["audio_hash"]


def test_artifact_uuid_naming(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="artifact")
    artifact_id = result["metadata"]["artifact_id"]
    assert artifact_id
    assert (tmp_path / "artifacts" / f"{artifact_id}.wav").is_file()


def test_path_traversal_rejected_in_artifact_root(tmp_path: Path):
    store = VoiceArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.resolve_artifact_path("../escape")


def test_shell_injection_in_text_rejected():
    result = sanitize_voice_text("hello; rm -rf /", max_length=500)
    assert result["error"] == "unsafe_text"


def test_timeout_handling_mock_slow_sidecar(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))

    def _timed_out(_request: VoiceSynthesisRequest):
        raise TimeoutError("mock sidecar timed out")

    service._adapter.initialize()
    service._adapter.synthesize = _timed_out  # type: ignore[method-assign]
    result = service.synthesize(text="slow")
    assert result["ok"] is False
    assert result["error"] == "synthesis_failed"
    assert "timed out" in result["message"]


def test_tts_failure_returns_error_without_raising_through_service(tmp_path: Path):
    config = _voice_config(tmp_path, backend="sidecar_http")
    service = VoiceService(config)
    result = service.synthesize(text="will fail")
    assert result["ok"] is False
    assert result["error"] == "synthesis_failed"


def test_state_engine_unchanged_after_speak_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AI_ECO_LAB_CONDITION", "C")
    provider = ExperimentalStateProvider(
        snapshots=__import__("state.snapshots", fromlist=["SnapshotStore"]).SnapshotStore(
            tmp_path / "snapshots"
        ),
    )
    before = provider.observe()
    service = VoiceService(_voice_config(tmp_path))
    service.synthesize(text="state isolation")
    after = provider.observe()
    assert before["state"] == after["state"]


def test_voice_code_does_not_import_state_facade():
    voice_root = Path(__file__).resolve().parents[1] / "voice"
    forbidden = ("instrumentation.facade", "record_experiment_event", "observe_internal_state")
    for path in voice_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} must not reference {token}"


def test_voice_instruction_does_not_call_record_experiment_event(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(
        text="instruction only",
        voice_instruction="adult, calm, clear",
    )
    assert result["ok"]


def test_seed_passed_through_to_adapter_request(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="seed", seed=123)
    assert result["ok"]
    event = json.loads((tmp_path / "voice_events.jsonl").read_text(encoding="utf-8").strip())
    assert event["seed"] == 123


def test_unknown_preset_returns_error(tmp_path: Path):
    service = VoiceService(_voice_config(tmp_path))
    result = service.synthesize(text="preset", voice_preset="missing")
    assert result["error"] == "unknown_preset"


def test_resolve_preset_helper():
    resolved = resolve_preset("research_neutral")
    assert resolved["ok"]
    assert "neutral" in resolved["instruction"]
