from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voice.audio_capture import AudioCaptureAdapter, build_audio_capture_adapter
from voice.stt_adapter import STTAdapter, build_stt_adapter
from voice.stt_config import STTConfig
from voice.stt_events import STTEventStore
from voice.stt_types import STTEvent, STT_DISCLAIMER, TranscriptionResult
from voice.stt_wav import validate_wav_bytes


class VoiceInputService:
    def __init__(
        self,
        config: STTConfig | None = None,
        *,
        stt_adapter: STTAdapter | None = None,
        capture_adapter: AudioCaptureAdapter | None = None,
    ) -> None:
        self.config = (config or STTConfig.from_env()).resolve_paths()
        self._stt_adapter: STTAdapter = stt_adapter or build_stt_adapter(self.config)
        self._capture_adapter: AudioCaptureAdapter = (
            capture_adapter or build_audio_capture_adapter(self.config)
        )
        self._event_store = STTEventStore(self.config.events_path)
        self._initialized = False
        self._last_error: str | None = None

    def initialize(self) -> None:
        validation = self.config.validate()
        if validation is not None:
            raise RuntimeError(validation["message"])
        if self.config.backend == "null":
            raise RuntimeError("STT backend is null; enable STT and configure a real backend.")
        self._stt_adapter.initialize()
        self._initialized = True

    def get_status(self) -> dict[str, Any]:
        validation = self.config.validate()
        payload: dict[str, Any] = {
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "backend": self.config.backend,
            "model": self.config.model_label(),
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "language": self.config.language,
            "sample_rate": self.config.sample_rate,
            "max_record_seconds": self.config.max_record_seconds,
            "persist_audio": self.config.persist_audio,
            "allow_model_download": self.config.allow_model_download,
            "last_error": self._last_error,
            "disclaimer": STT_DISCLAIMER,
        }
        if validation is not None:
            payload["available"] = False
            payload["validation_error"] = validation
            return payload
        payload["available"] = True
        payload["stt"] = self._stt_adapter.status()
        try:
            payload["microphones"] = self.list_microphones()
            payload["microphone_count"] = len(payload["microphones"])
        except Exception as exc:
            payload["microphones"] = []
            payload["microphone_count"] = 0
            payload["microphone_error"] = str(exc)
        return payload

    def list_microphones(self) -> list[dict[str, Any]]:
        return self._capture_adapter.list_devices()

    def record_and_transcribe(
        self,
        *,
        duration_seconds: float,
        device_index: int | None = None,
        source: str = "microphone",
    ) -> dict[str, Any]:
        try:
            clamped = self.config.clamp_duration(duration_seconds)
            wav_bytes = self._capture_adapter.capture(
                duration_seconds=clamped,
                device=device_index,
            )
            return self.transcribe_bytes(wav_bytes, source=source)
        except Exception as exc:
            self._last_error = str(exc)
            return {
                "ok": False,
                "error": "record_and_transcribe_failed",
                "message": str(exc),
            }

    def transcribe_bytes(self, wav_bytes: bytes, *, source: str = "file_bytes") -> dict[str, Any]:
        temp_path: Path | None = None
        try:
            if not self._initialized:
                self.initialize()
            props = validate_wav_bytes(
                wav_bytes,
                expected_sample_rate=self.config.sample_rate,
                expected_channels=1,
            )
            audio_hash = hashlib.sha256(wav_bytes).hexdigest()

            if self.config.persist_audio:
                self.config.temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = self.config.temp_dir / f"{uuid.uuid4()}.wav"
                temp_path.write_bytes(wav_bytes)
            else:
                temp_path = self._write_temp_wav(wav_bytes)

            result = self._stt_adapter.transcribe(wav_bytes, mime_type="audio/wav")
            enriched = self._enrich_result(result, source=source)
            self._record_event(enriched, props=props)
            return {
                "ok": True,
                "transcription": enriched.to_dict(),
                "metadata": enriched.to_dict(),
            }
        except Exception as exc:
            self._last_error = str(exc)
            return {
                "ok": False,
                "error": "transcribe_failed",
                "message": str(exc),
            }
        finally:
            if temp_path is not None and not self.config.persist_audio and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _write_temp_wav(self, wav_bytes: bytes) -> Path:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.config.temp_dir / f"{uuid.uuid4()}.wav"
        temp_path.write_bytes(wav_bytes)
        return temp_path

    def _enrich_result(self, result: TranscriptionResult, *, source: str) -> TranscriptionResult:
        return TranscriptionResult(
            transcription_id=result.transcription_id,
            timestamp=result.timestamp,
            text=result.text,
            language=result.language,
            language_probability=result.language_probability,
            duration_seconds=result.duration_seconds,
            backend=result.backend,
            model=result.model,
            device=result.device,
            compute_type=result.compute_type,
            processing_time_seconds=result.processing_time_seconds,
            segments=result.segments,
            audio_hash=result.audio_hash,
            source=source,
            experiment_id=os.environ.get("AI_ECO_LAB_EXPERIMENT_ID") or None,
            trial_id=os.environ.get("AI_ECO_LAB_TRIAL_ID") or None,
            stage=os.environ.get("AI_ECO_LAB_TRIAL_STAGE") or None,
            disclaimer=result.disclaimer,
        )

    def _record_event(self, result: TranscriptionResult, *, props: dict[str, int | float]) -> None:
        text_hash = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        event = STTEvent(
            transcription_id=result.transcription_id,
            type="speech_transcription",
            timestamp=result.timestamp or datetime.now(timezone.utc).isoformat(),
            experiment_id=result.experiment_id,
            trial_id=result.trial_id,
            stage=result.stage,
            stt_backend=result.backend,
            stt_model=result.model,
            stt_device=result.device,
            stt_compute_type=result.compute_type,
            language=result.language,
            language_probability=result.language_probability,
            text=result.text,
            text_hash=text_hash,
            audio_hash=result.audio_hash,
            duration_seconds=float(props["duration_seconds"]),
            processing_time_seconds=result.processing_time_seconds,
            source=result.source,
            segments=result.segments,
            disclaimer=result.disclaimer,
        )
        self._event_store.append(event)

    def shutdown(self) -> None:
        self._capture_adapter.shutdown()
        self._stt_adapter.shutdown()
        self._initialized = False


_singleton: VoiceInputService | None = None


def get_voice_input_service() -> VoiceInputService:
    global _singleton
    if _singleton is None:
        _singleton = VoiceInputService()
    return _singleton


def reset_voice_input_service_for_tests() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.shutdown()
    _singleton = None
