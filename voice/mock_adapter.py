from __future__ import annotations

import hashlib
import time

from voice.acoustic import build_mock_wav, extract_acoustic_features, wav_format
from voice.adapter import VoiceAdapter
from voice.config import VoiceConfig
from voice.types import VoiceStatus, VoiceSynthesisRequest, VoiceSynthesisResult


class MockVoiceAdapter(VoiceAdapter):
    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._initialized = False
        self._last_error: str | None = None
        self._slow_seconds = 0.0

    def set_slow_seconds(self, seconds: float) -> None:
        self._slow_seconds = max(0.0, seconds)

    def initialize(self) -> None:
        self._initialized = True
        self._last_error = None

    def get_status(self) -> VoiceStatus:
        return VoiceStatus(
            initialized=self._initialized,
            backend="mock",
            model_path=str(self.config.model_path),
            codec_path=str(self.config.codec_path),
            model_sha256=self._file_sha256(self.config.model_path),
            codec_sha256=self._file_sha256(self.config.codec_path),
            language=self.config.language,
            available=True,
            loaded=self._initialized,
            ggml_backend=self.config.ggml_backend,
            last_error=self._last_error,
        )

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        if not self._initialized:
            self.initialize()
        started = time.perf_counter()
        if self._slow_seconds:
            time.sleep(self._slow_seconds)
        duration = min(2.0, max(0.2, len(request.text) / 120.0))
        wav_bytes = build_mock_wav(
            sample_rate=24000,
            channels=1,
            duration_seconds=duration,
            seed=request.seed,
        )
        sample_rate, channels = wav_format(wav_bytes)
        features = extract_acoustic_features(wav_bytes)
        elapsed = time.perf_counter() - started
        return VoiceSynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=features["duration_seconds"],
            generation_time_seconds=elapsed,
            backend="mock",
            seed_used=request.seed,
        )

    def shutdown(self) -> None:
        self._initialized = False

    @staticmethod
    def _file_sha256(path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
