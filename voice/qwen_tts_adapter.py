from __future__ import annotations

import hashlib
import logging
import sys

from voice.acoustic import extract_acoustic_features, wav_format
from voice.adapter import VoiceAdapter
from voice.config import VoiceConfig
from voice.process_manager import VoiceProcessManager
from voice.types import VoiceStatus, VoiceSynthesisRequest, VoiceSynthesisResult

_logger = logging.getLogger(__name__)


class QwenTTSAdapter(VoiceAdapter):
    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._initialized = False
        self._last_error: str | None = None
        self._model_sha256: str | None = None
        self._codec_sha256: str | None = None
        self._process_manager = VoiceProcessManager(
            sidecar_url=config.sidecar_url,
            binary_path=config.resolve_binary(),
            model_path=config.model_path,
            codec_path=config.codec_path,
            auto_start=config.auto_start_sidecar,
            ggml_backend=config.ggml_backend,
            timeout_seconds=config.timeout_seconds,
        )

    def initialize(self) -> None:
        validation = self.config.validate()
        if validation:
            self._last_error = validation["message"]
            raise RuntimeError(self._last_error)
        self._model_sha256 = self._sha256_file(self.config.model_path)
        self._codec_sha256 = self._sha256_file(self.config.codec_path)
        if self.config.backend == "cli_oneshot":
            message = (
                "QWEN_TTS_BACKEND=cli_oneshot reloads models on every call; "
                "use sidecar_http in production."
            )
            _logger.warning(message)
            print(message, file=sys.stderr)
        if self.config.backend == "sidecar_http":
            self._process_manager.ensure_sidecar_http()
        elif self.config.backend == "sidecar_rpc":
            self._process_manager.ensure_sidecar_rpc()
        self._initialized = True
        self._last_error = None

    def get_status(self) -> VoiceStatus:
        loaded = self._initialized
        available = False
        if self.config.backend == "sidecar_http":
            available = self._process_manager.health_check()
        elif self._initialized:
            available = True
        return VoiceStatus(
            initialized=self._initialized,
            backend=self.config.backend,
            model_path=str(self.config.model_path),
            codec_path=str(self.config.codec_path),
            model_sha256=self._model_sha256,
            codec_sha256=self._codec_sha256,
            language=self.config.language,
            available=available,
            loaded=loaded,
            ggml_backend=self.config.ggml_backend,
            last_error=self._last_error,
            sidecar_pid=self._process_manager.sidecar_pid,
            sidecar_url=self.config.sidecar_url,
        )

    def synthesize(self, request: VoiceSynthesisRequest) -> VoiceSynthesisResult:
        if not self._initialized:
            self.initialize()
        import time

        started = time.perf_counter()
        try:
            if self.config.backend == "sidecar_http":
                wav_bytes = self._process_manager.synthesize_http(
                    text=request.text,
                    instructions=request.voice_instruction,
                    language=request.language,
                    seed=request.seed,
                    speed=request.speed,
                )
            elif self.config.backend == "sidecar_rpc":
                wav_bytes = self._process_manager.synthesize_rpc(
                    text=request.text,
                    instructions=request.voice_instruction,
                    language=request.language,
                    seed=request.seed,
                    speed=request.speed,
                )
            else:
                wav_bytes = self._process_manager.synthesize_cli_oneshot(
                    text=request.text,
                    instructions=request.voice_instruction,
                    language=request.language,
                    seed=request.seed,
                )
        except Exception as exc:
            self._last_error = str(exc)
            self._initialized = False
            raise
        if len(wav_bytes) > self.config.max_wav_bytes:
            raise RuntimeError(
                f"Generated WAV exceeds max size ({self.config.max_wav_bytes} bytes)."
            )
        sample_rate, channels = wav_format(wav_bytes)
        features = extract_acoustic_features(wav_bytes)
        elapsed = time.perf_counter() - started
        self._last_error = None
        return VoiceSynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=features["duration_seconds"],
            generation_time_seconds=elapsed,
            backend=self.config.backend,
            seed_used=request.seed,
        )

    def shutdown(self) -> None:
        self._process_manager.shutdown()
        self._initialized = False

    @staticmethod
    def _sha256_file(path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
