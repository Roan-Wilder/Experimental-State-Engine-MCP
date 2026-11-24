from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from voice.stt_config import STTConfig, sounddevice_available
from voice.stt_wav import pcm16_to_wav_bytes


class AudioCaptureAdapter(ABC):
    @abstractmethod
    def list_devices(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def capture(self, *, duration_seconds: float, device: int | None = None) -> bytes: ...

    @abstractmethod
    def shutdown(self) -> None: ...


class NullAudioCaptureAdapter(AudioCaptureAdapter):
    def list_devices(self) -> list[dict[str, Any]]:
        return []

    def capture(self, *, duration_seconds: float, device: int | None = None) -> bytes:
        raise RuntimeError("Audio capture is not available (null adapter).")

    def shutdown(self) -> None:
        return None


class LocalMicrophoneAdapter(AudioCaptureAdapter):
    def __init__(self, config: STTConfig) -> None:
        self.config = config
        if not sounddevice_available():
            raise RuntimeError(
                "sounddevice is not installed. Install STT extras: pip install -e \".[stt]\""
            )

    def list_devices(self) -> list[dict[str, Any]]:
        import sounddevice as sd

        devices: list[dict[str, Any]] = []
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                devices.append(
                    {
                        "index": index,
                        "name": info.get("name"),
                        "max_input_channels": info.get("max_input_channels"),
                        "default_samplerate": info.get("default_samplerate"),
                    }
                )
        return devices

    def capture(self, *, duration_seconds: float, device: int | None = None) -> bytes:
        import numpy as np
        import sounddevice as sd

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if duration_seconds > self.config.max_record_seconds:
            raise ValueError(
                f"duration_seconds exceeds STT_MAX_RECORD_SECONDS ({self.config.max_record_seconds})."
            )
        devices = self.list_devices()
        if not devices:
            raise RuntimeError("No microphone input devices detected.")
        if device is not None and not any(entry["index"] == device for entry in devices):
            raise RuntimeError(f"Microphone device index not found: {device}")

        frames = int(duration_seconds * self.config.sample_rate)
        try:
            recording = sd.rec(
                frames,
                samplerate=self.config.sample_rate,
                channels=1,
                dtype="int16",
                device=device,
            )
            sd.wait()
        except Exception as exc:
            raise RuntimeError(f"Microphone capture failed: {exc}") from exc

        pcm = np.asarray(recording, dtype=np.int16).reshape(-1).tobytes()
        if not pcm:
            raise RuntimeError("Microphone capture returned empty audio.")
        return pcm16_to_wav_bytes(
            pcm,
            sample_rate=self.config.sample_rate,
            channels=1,
        )

    def shutdown(self) -> None:
        return None


def build_audio_capture_adapter(config: STTConfig) -> AudioCaptureAdapter:
    if config.backend == "null":
        return NullAudioCaptureAdapter()
    if not sounddevice_available():
        return NullAudioCaptureAdapter()
    return LocalMicrophoneAdapter(config)
