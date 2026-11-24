from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voice.stt_adapter import STTAdapter
from voice.stt_config import STTConfig, faster_whisper_available
from voice.stt_types import STT_DISCLAIMER, TranscriptionResult, TranscriptionSegment
from voice.stt_wav import validate_wav_bytes, wav_to_float32_mono


class FasterWhisperSTTAdapter(STTAdapter):
    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model: Any = None
        self._initialized = False
        self._last_error: str | None = None

    def initialize(self) -> None:
        if self._initialized:
            return
        if not faster_whisper_available():
            raise RuntimeError(
                'faster-whisper is not installed. Install STT extras: pip install -e ".[stt]"'
            )
        from faster_whisper import WhisperModel

        model_ref = self._resolve_model_reference()
        try:
            self._model = WhisperModel(
                model_ref,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        except Exception as exc:
            self._last_error = str(exc)
            raise RuntimeError(f"Failed to initialize faster-whisper model: {exc}") from exc
        self._initialized = True

    def _resolve_model_reference(self) -> str:
        if self.config.model_path is not None:
            return str(self.config.model_path)
        if self.config.allow_model_download:
            return self.config.model
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_dir = cache_root / "hub"
        if hub_dir.is_dir():
            pattern = f"models--Systran--faster-whisper-{self.config.model}"
            for candidate in hub_dir.glob(pattern):
                if candidate.is_dir():
                    return self.config.model
        raise RuntimeError(
            "Whisper model is not available locally. Set STT_MODEL_PATH to a local model, "
            "or set STT_ALLOW_MODEL_DOWNLOAD=1 to download on initialization."
        )

    def transcribe(self, audio_bytes: bytes, *, mime_type: str) -> TranscriptionResult:
        if mime_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
            raise ValueError(f"Unsupported mime_type for STT: {mime_type}")
        if not self._initialized:
            self.initialize()
        assert self._model is not None

        props = validate_wav_bytes(
            audio_bytes,
            expected_sample_rate=self.config.sample_rate,
            expected_channels=1,
        )
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        floats, sample_rate = wav_to_float32_mono(audio_bytes)
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Unexpected sample rate {sample_rate}; expected {self.config.sample_rate}."
            )

        import numpy as np

        audio = np.asarray(floats, dtype=np.float32)
        started = time.perf_counter()
        segments_iter, info = self._model.transcribe(
            audio,
            language=self.config.language or None,
            beam_size=1,
        )
        segments: list[TranscriptionSegment] = []
        text_parts: list[str] = []
        for segment in segments_iter:
            segment_text = segment.text.strip()
            if not segment_text:
                continue
            text_parts.append(segment_text)
            segments.append(
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment_text,
                )
            )
        processing_time = time.perf_counter() - started
        text = " ".join(text_parts).strip()
        language_probability = getattr(info, "language_probability", None)
        if language_probability is not None:
            language_probability = float(language_probability)

        return TranscriptionResult(
            transcription_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            text=text,
            language=getattr(info, "language", None) or self.config.language,
            language_probability=language_probability,
            duration_seconds=float(props["duration_seconds"]),
            backend="faster_whisper",
            model=self.config.model_label(),
            device=self.config.device,
            compute_type=self.config.compute_type,
            processing_time_seconds=processing_time,
            segments=tuple(segments),
            audio_hash=audio_hash,
            source="bytes",
            experiment_id=None,
            trial_id=None,
            stage=None,
            disclaimer=STT_DISCLAIMER,
        )

    def shutdown(self) -> None:
        self._model = None
        self._initialized = False

    def status(self) -> dict[str, Any]:
        return {
            "implemented": True,
            "adapter": "faster_whisper",
            "initialized": self._initialized,
            "model": self.config.model_label(),
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "allow_model_download": self.config.allow_model_download,
            "last_error": self._last_error,
        }
