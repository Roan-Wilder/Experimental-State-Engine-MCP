from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone

from voice.stt_adapter import STTAdapter
from voice.stt_config import STTConfig
from voice.stt_types import STT_DISCLAIMER, TranscriptionResult, TranscriptionSegment
from voice.stt_wav import wav_duration_seconds


class MockSTTAdapter(STTAdapter):
    """Deterministic STT adapter for unit tests only (STT_BACKEND=mock)."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def transcribe(self, audio_bytes: bytes, *, mime_type: str) -> TranscriptionResult:
        if not self._initialized:
            self.initialize()
        started = time.perf_counter()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        duration = wav_duration_seconds(audio_bytes)
        text = f"mock transcription {audio_hash[:12]}"
        processing_time = time.perf_counter() - started
        return TranscriptionResult(
            transcription_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            text=text,
            language=self.config.language,
            language_probability=None,
            duration_seconds=duration,
            backend="mock",
            model=self.config.model_label(),
            device=self.config.device,
            compute_type=self.config.compute_type,
            processing_time_seconds=processing_time,
            segments=(
                TranscriptionSegment(start=0.0, end=max(duration, 0.01), text=text),
            ),
            audio_hash=audio_hash,
            source="mock",
            experiment_id=None,
            trial_id=None,
            stage=None,
            disclaimer=STT_DISCLAIMER,
        )

    def shutdown(self) -> None:
        self._initialized = False

    def status(self) -> dict[str, object]:
        return {
            "implemented": True,
            "adapter": "mock",
            "initialized": self._initialized,
            "model": self.config.model_label(),
            "device": self.config.device,
            "compute_type": self.config.compute_type,
        }
