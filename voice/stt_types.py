from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STT_DISCLAIMER = (
    "Speech recognition output is transcribed audio data, not evidence of "
    "consciousness, emotion, intent, or internal state."
)


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class TranscriptionResult:
    transcription_id: str
    timestamp: str
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float
    backend: str
    model: str
    device: str
    compute_type: str
    processing_time_seconds: float
    segments: tuple[TranscriptionSegment, ...]
    audio_hash: str
    source: str
    experiment_id: str | None
    trial_id: str | None
    stage: str | None
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcription_id": self.transcription_id,
            "timestamp": self.timestamp,
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration_seconds": self.duration_seconds,
            "backend": self.backend,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "processing_time_seconds": self.processing_time_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
            "audio_hash": self.audio_hash,
            "source": self.source,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class STTEvent:
    transcription_id: str
    type: str
    timestamp: str
    experiment_id: str | None
    trial_id: str | None
    stage: str | None
    stt_backend: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    language: str | None
    language_probability: float | None
    text: str
    text_hash: str
    audio_hash: str
    duration_seconds: float
    processing_time_seconds: float
    source: str
    segments: tuple[TranscriptionSegment, ...]
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcription_id": self.transcription_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "trial_id": self.trial_id,
            "stage": self.stage,
            "stt_backend": self.stt_backend,
            "stt_model": self.stt_model,
            "stt_device": self.stt_device,
            "stt_compute_type": self.stt_compute_type,
            "language": self.language,
            "language_probability": self.language_probability,
            "text": self.text,
            "text_hash": self.text_hash,
            "audio_hash": self.audio_hash,
            "duration_seconds": self.duration_seconds,
            "processing_time_seconds": self.processing_time_seconds,
            "source": self.source,
            "segments": [segment.to_dict() for segment in self.segments],
            "disclaimer": self.disclaimer,
        }
