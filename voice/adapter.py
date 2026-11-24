from __future__ import annotations

from abc import ABC, abstractmethod

from voice.types import VoiceStatus, VoiceSynthesisRequest, VoiceSynthesisResult


class VoiceAdapter(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Load models or connect to sidecar."""

    @abstractmethod
    def get_status(self) -> VoiceStatus:
        """Return current adapter status."""

    @abstractmethod
    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        """Synthesize speech for the given request."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources."""

    def synthesize_stream(self, request: VoiceSynthesisRequest):  # pragma: no cover
        raise NotImplementedError("Streaming synthesis is not implemented in Phase 1.")
