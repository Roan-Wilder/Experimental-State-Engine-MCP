from __future__ import annotations

import io
import math
import struct
import wave
from abc import ABC, abstractmethod
from typing import Any

# Acoustic analyzers may emit measurable signal properties only.
# They must NOT map measurements to emotion, consciousness, sentience,
# or subjective affective state.

RESERVED_ACOUSTIC_KEYS = frozenset(
    {
        "duration_seconds",
        "rms_energy",
        "peak_amplitude",
        "f0_hz",
        "f0_mean_hz",
        "f0_std_hz",
        "speaking_rate",
        "speech_pause_timing",
        "spectral_centroid_hz",
        "spectral_measurements",
    }
)

FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "emotion",
        "affect",
        "consciousness",
        "sentience",
        "subjective_experience",
        "internal_affect",
    }
)


def wav_duration_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def wav_format(wav_bytes: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        return handle.getframerate(), handle.getnchannels()


def _read_pcm_samples(wav_bytes: bytes) -> tuple[list[float], int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
        sample_rate = handle.getframerate()

    if sample_width == 2:
        count = len(raw) // 2
        ints = struct.unpack(f"<{count}h", raw)
        samples = [value / 32768.0 for value in ints]
    elif sample_width == 1:
        samples = [(byte - 128) / 128.0 for byte in raw]
    elif sample_width == 4:
        count = len(raw) // 4
        ints = struct.unpack(f"<{count}i", raw)
        samples = [value / 2147483648.0 for value in ints]
    else:
        samples = []
    return samples, sample_rate


def _compute_rms_peak(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    return rms, peak


class AcousticAnalyzer(ABC):
    name: str = "base"
    version: str = "0.0.0"

    @abstractmethod
    def analyze(self, wav_bytes: bytes) -> dict[str, Any]:
        """Return measurable acoustic properties only."""


class BasicAcousticAnalyzer(AcousticAnalyzer):
    name = "basic"
    version = "1.0.0"

    def analyze(self, wav_bytes: bytes) -> dict[str, Any]:
        duration = wav_duration_seconds(wav_bytes)
        samples, _sample_rate = _read_pcm_samples(wav_bytes)
        rms, peak = _compute_rms_peak(samples)
        return {
            "duration_seconds": duration,
            "rms_energy": rms,
            "peak_amplitude": peak,
            "f0_hz": None,
            "f0_mean_hz": None,
            "f0_std_hz": None,
            "speaking_rate": None,
            "speech_pause_timing": None,
            "spectral_centroid_hz": None,
            "spectral_measurements": None,
        }


class AcousticAnalyzerRegistry:
    def __init__(self, analyzers: list[AcousticAnalyzer] | None = None) -> None:
        self._analyzers = analyzers or [BasicAcousticAnalyzer()]

    def run(self, wav_bytes: bytes) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        observations: list[dict[str, Any]] = []
        for analyzer in self._analyzers:
            features = analyzer.analyze(wav_bytes)
            for key in features:
                if key in FORBIDDEN_INFERENCE_KEYS:
                    raise ValueError(f"Acoustic analyzer emitted forbidden inference key: {key}")
            merged.update(features)
            observations.append(
                {
                    "analyzer": analyzer.name,
                    "version": analyzer.version,
                    "features": features,
                }
            )
        return {
            "features": merged,
            "acoustic_observation": {
                "analyzers": observations,
                "note": (
                    "Acoustic measurements are signal properties only; "
                    "not evidence of emotion, consciousness, or subjective experience."
                ),
            },
        }


_default_registry = AcousticAnalyzerRegistry()


def extract_acoustic_features(wav_bytes: bytes) -> dict[str, Any]:
    """Backward-compatible helper returning flat feature dict."""
    return _default_registry.run(wav_bytes)["features"]


def analyze_acoustic_observation(wav_bytes: bytes) -> dict[str, Any]:
    return _default_registry.run(wav_bytes)


def build_mock_wav(
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    duration_seconds: float = 0.25,
    seed: int | None = None,
) -> bytes:
    import random

    rng = random.Random(seed if seed is not None else 0)
    frame_count = max(1, int(sample_rate * duration_seconds))
    samples = []
    frequency = 220.0 + (rng.random() * 80.0)
    for index in range(frame_count):
        value = math.sin(2.0 * math.pi * frequency * (index / sample_rate))
        samples.append(int(value * 12000))
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()
