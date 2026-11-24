from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from voice.stt_config import STTConfig
from voice.stt_types import TranscriptionResult


class STTAdapter(ABC):
    """Speech-to-text boundary (input modality only)."""

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, *, mime_type: str) -> TranscriptionResult: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def status(self) -> dict[str, Any]:
        return {
            "implemented": True,
            "adapter": self.__class__.__name__,
        }


class NullSTTAdapter(STTAdapter):
    def initialize(self) -> None:
        return None

    def transcribe(self, audio_bytes: bytes, *, mime_type: str) -> TranscriptionResult:
        raise NotImplementedError(
            "STT is not implemented. Future pipeline: Microphone -> STTAdapter -> Gemma."
        )

    def shutdown(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "implemented": False,
            "adapter": "null",
            "message": (
                "Speech recognition is a future extension and does not affect "
                "experimental state."
            ),
        }


def build_stt_adapter(config: STTConfig) -> STTAdapter:
    if config.backend == "null":
        return NullSTTAdapter()
    if config.backend == "mock":
        from voice.stt_mock_adapter import MockSTTAdapter

        return MockSTTAdapter(config)
    if config.backend == "faster_whisper":
        from voice.local_stt_adapter import FasterWhisperSTTAdapter

        return FasterWhisperSTTAdapter(config)
    raise ValueError(f"Unsupported STT backend: {config.backend}")
