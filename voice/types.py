from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class OutputMode(StrEnum):
    GENERATE_ONLY = "generate_only"
    GENERATE_AND_SAVE = "generate_and_save"
    GENERATE_AND_RETURN = "generate_and_return"
    GENERATE_AND_PLAY = "generate_and_play"


VALID_OUTPUT_MODES = frozenset(mode.value for mode in OutputMode)


@dataclass(frozen=True)
class VoiceSynthesisRequest:
    text: str
    voice_instruction: str
    language: str
    voice_preset: str | None = None
    voice_id: str | None = None
    speed: float | None = None
    seed: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None


@dataclass(frozen=True)
class VoiceSynthesisResult:
    wav_bytes: bytes
    sample_rate: int
    channels: int
    duration_seconds: float
    generation_time_seconds: float
    backend: str
    seed_used: int | None = None


@dataclass
class VoiceStatus:
    initialized: bool = False
    backend: str = "mock"
    model_path: str | None = None
    codec_path: str | None = None
    model_sha256: str | None = None
    codec_sha256: str | None = None
    language: str = "English"
    available: bool = False
    loaded: bool = False
    ggml_backend: str | None = None
    last_error: str | None = None
    sidecar_pid: int | None = None
    sidecar_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initialized": self.initialized,
            "backend": self.backend,
            "model_path": self.model_path,
            "codec_path": self.codec_path,
            "model_sha256": self.model_sha256,
            "codec_sha256": self.codec_sha256,
            "language": self.language,
            "available": self.available,
            "loaded": self.loaded,
            "ggml_backend": self.ggml_backend,
            "last_error": self.last_error,
            "sidecar_pid": self.sidecar_pid,
            "sidecar_url": self.sidecar_url,
        }


@dataclass(frozen=True)
class VoiceEvent:
    event_id: str
    type: str
    timestamp: str
    experiment_id: str | None
    trial_id: str | None
    stage: str | None
    tts_model: str
    tts_model_sha256: str | None
    tts_codec: str
    tts_codec_sha256: str | None
    tts_backend: str
    language: str
    voice_instruction: str
    voice_preset: str | None
    text_hash: str
    audio_hash: str
    sample_rate: int
    channels: int
    duration_seconds: float
    seed: int | None
    generation_time_seconds: float
    artifact_id: str | None
    acoustic_features: dict[str, Any]
    acoustic_observation: dict[str, Any] | None
    disclaimer: str
    voice_id: str | None = None
    persona_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "tts_model": self.tts_model,
            "tts_model_sha256": self.tts_model_sha256,
            "tts_codec": self.tts_codec,
            "tts_codec_sha256": self.tts_codec_sha256,
            "tts_backend": self.tts_backend,
            "language": self.language,
            "voice_instruction": self.voice_instruction,
            "voice_preset": self.voice_preset,
            "text_hash": self.text_hash,
            "audio_hash": self.audio_hash,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "generation_time_seconds": self.generation_time_seconds,
            "artifact_id": self.artifact_id,
            "acoustic_features": self.acoustic_features,
            "disclaimer": self.disclaimer,
        }
        if self.acoustic_observation is not None:
            payload["acoustic_observation"] = self.acoustic_observation
        if self.voice_id is not None:
            payload["voice_id"] = self.voice_id
        if self.persona_version is not None:
            payload["persona_version"] = self.persona_version
        return payload


VOICE_DISCLAIMER = (
    "Voice output is acoustic data, not evidence of consciousness or affective state."
)
