from __future__ import annotations

import struct
import wave
from io import BytesIO


MIN_WAV_BYTES = 44


def validate_wav_bytes(
    audio_bytes: bytes,
    *,
    expected_sample_rate: int | None = None,
    expected_channels: int = 1,
) -> dict[str, int | float]:
    if len(audio_bytes) < MIN_WAV_BYTES:
        raise ValueError(f"WAV too small ({len(audio_bytes)} bytes).")
    with wave.open(BytesIO(audio_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channel(s), got {channels}.")
        if expected_sample_rate is not None and sample_rate != expected_sample_rate:
            raise ValueError(
                f"Expected sample rate {expected_sample_rate}, got {sample_rate}."
            )
        if frames <= 0:
            raise ValueError("WAV contains zero frames.")
        duration = frames / float(sample_rate)
        if duration <= 0:
            raise ValueError("WAV duration must be positive.")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds": duration,
    }


def wav_duration_seconds(audio_bytes: bytes) -> float:
    return float(validate_wav_bytes(audio_bytes)["duration_seconds"])


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_bytes)
    return buffer.getvalue()


def wav_to_float32_mono(audio_bytes: bytes) -> tuple[list[float], int]:
    with wave.open(BytesIO(audio_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    count = len(frames) // 2
    samples = struct.unpack(f"<{count}h", frames)
    if channels == 1:
        floats = [sample / 32768.0 for sample in samples]
        return floats, sample_rate
    if channels == 2:
        left = samples[0::2]
        right = samples[1::2]
        floats = [(left[index] + right[index]) / (2 * 32768.0) for index in range(len(left))]
        return floats, sample_rate
    raise ValueError(f"Unsupported WAV channel count: {channels}")
