from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from voice.types import OutputMode, VALID_OUTPUT_MODES

VALID_BACKENDS = frozenset({"sidecar_http", "sidecar_rpc", "cli_oneshot", "mock"})

DEFAULT_PERSONA_REFERENCE_TEXT = (
    "This is a reference sample for voice persona calibration."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "True", "yes", "YES"}


def tts_enabled_from_env() -> bool:
    """Master TTS switch. Defaults OFF. Legacy QWEN_TTS_ENABLED used only when TTS_ENABLED unset."""
    if "TTS_ENABLED" in os.environ:
        return _env_bool("TTS_ENABLED", False)
    return _env_bool("QWEN_TTS_ENABLED", False)


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    backend: str
    model_path: Path
    codec_path: Path
    binary_path: Path | None
    sidecar_url: str
    auto_start_sidecar: bool
    language: str
    default_instruction: str
    output_format: str
    max_text_length: int
    timeout_seconds: float
    persist_artifacts: bool
    artifact_root: Path
    events_path: Path
    personas_path: Path
    benchmarks_path: Path
    persona_reference_text: str
    output_mode: str
    playback_enabled: bool
    ggml_backend: str | None
    max_wav_bytes: int

    @classmethod
    def from_env(cls, *, repo_root: Path | None = None) -> VoiceConfig:
        root = repo_root or _repo_root()
        backend = os.environ.get("QWEN_TTS_BACKEND", "sidecar_http").strip()
        binary_raw = os.environ.get("QWEN_TTS_BINARY", "").strip()
        binary_path = Path(binary_raw) if binary_raw else None
        output_mode = os.environ.get(
            "QWEN_TTS_OUTPUT_MODE",
            OutputMode.GENERATE_AND_RETURN.value,
        ).strip()

        return cls(
            enabled=tts_enabled_from_env(),
            backend=backend,
            model_path=Path(
                os.environ.get(
                    "QWEN_TTS_MODEL_PATH",
                    "models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf",
                )
            ),
            codec_path=Path(
                os.environ.get(
                    "QWEN_TTS_CODEC_PATH",
                    "models/qwen-tokenizer-12hz-Q4_K_M.gguf",
                )
            ),
            binary_path=binary_path,
            sidecar_url=os.environ.get("QWEN_TTS_SIDECAR_URL", "http://127.0.0.1:8765").strip(),
            auto_start_sidecar=_env_bool("QWEN_TTS_AUTO_START_SIDECAR", False),
            language=os.environ.get("QWEN_TTS_LANGUAGE", "English").strip(),
            default_instruction=os.environ.get(
                "QWEN_TTS_DEFAULT_INSTRUCTION",
                "adult, neutral, clear, moderate pitch, conversational",
            ).strip(),
            output_format=os.environ.get("QWEN_TTS_OUTPUT_FORMAT", "wav").strip(),
            max_text_length=int(os.environ.get("QWEN_TTS_MAX_TEXT_LENGTH", "500")),
            timeout_seconds=float(os.environ.get("QWEN_TTS_TIMEOUT_SECONDS", "120")),
            persist_artifacts=_env_bool("QWEN_TTS_PERSIST_ARTIFACTS", True),
            artifact_root=Path(
                os.environ.get("QWEN_TTS_ARTIFACT_ROOT", "research/audio_artifacts")
            ),
            events_path=Path(
                os.environ.get(
                    "QWEN_TTS_EVENTS_PATH",
                    "research/raw/voice_events.jsonl",
                )
            ),
            personas_path=Path(
                os.environ.get(
                    "QWEN_TTS_PERSONAS_PATH",
                    "research/raw/voice_personas.jsonl",
                )
            ),
            benchmarks_path=Path(
                os.environ.get(
                    "QWEN_TTS_BENCHMARKS_PATH",
                    "research/benchmarks/voice_tts.jsonl",
                )
            ),
            persona_reference_text=os.environ.get(
                "QWEN_TTS_PERSONA_REFERENCE_TEXT",
                DEFAULT_PERSONA_REFERENCE_TEXT,
            ).strip(),
            output_mode=output_mode,
            playback_enabled=_env_bool("QWEN_TTS_PLAYBACK_ENABLED", False),
            ggml_backend=os.environ.get("GGML_BACKEND", "").strip() or None,
            max_wav_bytes=int(os.environ.get("QWEN_TTS_MAX_WAV_BYTES", str(50 * 1024 * 1024))),
        )

    def resolve_paths(self, repo_root: Path | None = None) -> VoiceConfig:
        root = repo_root or _repo_root()

        def _resolve(path: Path) -> Path:
            return (root / path).resolve() if not path.is_absolute() else path.resolve()

        return VoiceConfig(
            enabled=self.enabled,
            backend=self.backend,
            model_path=_resolve(self.model_path),
            codec_path=_resolve(self.codec_path),
            binary_path=self.binary_path,
            sidecar_url=self.sidecar_url,
            auto_start_sidecar=self.auto_start_sidecar,
            language=self.language,
            default_instruction=self.default_instruction,
            output_format=self.output_format,
            max_text_length=self.max_text_length,
            timeout_seconds=self.timeout_seconds,
            persist_artifacts=self.persist_artifacts,
            artifact_root=_resolve(self.artifact_root),
            events_path=_resolve(self.events_path),
            personas_path=_resolve(self.personas_path),
            benchmarks_path=_resolve(self.benchmarks_path),
            persona_reference_text=self.persona_reference_text,
            output_mode=self.output_mode,
            playback_enabled=self.playback_enabled,
            ggml_backend=self.ggml_backend,
            max_wav_bytes=self.max_wav_bytes,
        )

    def validate(self) -> dict[str, str] | None:
        if self.backend not in VALID_BACKENDS:
            return {
                "error": "invalid_backend",
                "message": f"QWEN_TTS_BACKEND must be one of {sorted(VALID_BACKENDS)}.",
            }
        if self.output_mode not in VALID_OUTPUT_MODES:
            return {
                "error": "invalid_output_mode",
                "message": f"QWEN_TTS_OUTPUT_MODE must be one of {sorted(VALID_OUTPUT_MODES)}.",
            }
        if self.backend != "mock":
            if not self.model_path.is_file():
                return {
                    "error": "missing_model",
                    "message": f"Talker model not found: {self.model_path}",
                }
            if not self.codec_path.is_file():
                return {
                    "error": "missing_codec",
                    "message": f"Codec model not found: {self.codec_path}",
                }
        if self.backend in {"sidecar_http", "sidecar_rpc", "cli_oneshot"}:
            binary = self.resolve_binary()
            if binary is None:
                return {
                    "error": "missing_binary",
                    "message": "QWEN TTS binary not found. Set QWEN_TTS_BINARY or build qwentts.cpp.",
                }
        return None

    def resolve_binary(self) -> Path | None:
        if self.binary_path and self.binary_path.is_file():
            return self.binary_path.resolve()
        for name in ("tts-server", "tts-server.exe", "qwen-tts", "qwen-tts.exe"):
            found = shutil.which(name)
            if found:
                return Path(found)
        return None

    def model_basename(self) -> str:
        return self.model_path.name

    def codec_basename(self) -> str:
        return self.codec_path.name

    def with_backend(self, backend: str) -> VoiceConfig:
        return VoiceConfig(
            enabled=self.enabled,
            backend=backend,
            model_path=self.model_path,
            codec_path=self.codec_path,
            binary_path=self.binary_path,
            sidecar_url=self.sidecar_url,
            auto_start_sidecar=self.auto_start_sidecar,
            language=self.language,
            default_instruction=self.default_instruction,
            output_format=self.output_format,
            max_text_length=self.max_text_length,
            timeout_seconds=self.timeout_seconds,
            persist_artifacts=self.persist_artifacts,
            artifact_root=self.artifact_root,
            events_path=self.events_path,
            personas_path=self.personas_path,
            benchmarks_path=self.benchmarks_path,
            persona_reference_text=self.persona_reference_text,
            output_mode=self.output_mode,
            playback_enabled=self.playback_enabled,
            ggml_backend=self.ggml_backend,
            max_wav_bytes=self.max_wav_bytes,
        )
