from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any

from voice.config import VALID_BACKENDS, VoiceConfig
from voice.mock_adapter import MockVoiceAdapter
from voice.qwen_tts_adapter import QwenTTSAdapter
from voice.types import VoiceSynthesisRequest

BENCHMARK_TEXT = "Hello. This is the AI Eco Lab experimental voice system."
DEFAULT_SEED = 42


def _sha256_file(path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_memory_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KB; macOS reports bytes.
        value = usage.ru_maxrss
        if sys.platform == "darwin":
            return int(value)
        return int(value * 1024)
    except Exception:
        return None


def _available_backends(config: VoiceConfig) -> list[str]:
    backends = ["mock"]
    if config.model_path.is_file() and config.codec_path.is_file() and config.resolve_binary():
        for backend in ("sidecar_http", "sidecar_rpc", "cli_oneshot"):
            if backend in VALID_BACKENDS:
                backends.append(backend)
    return backends


def _build_adapter(config: VoiceConfig):
    if config.backend == "mock":
        return MockVoiceAdapter(config)
    return QwenTTSAdapter(config)


def run_backend_benchmark(
    config: VoiceConfig,
    *,
    backend: str,
    text: str,
    seed: int,
) -> dict[str, Any]:
    backend_config = config.with_backend(backend)
    adapter = _build_adapter(backend_config)
    started = time.perf_counter()
    init_error = None
    try:
        adapter.initialize()
    except Exception as exc:
        init_error = str(exc)
    init_seconds = time.perf_counter() - started

    result_payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "text": text,
        "seed": seed,
        "init_seconds": init_seconds,
        "synthesis_seconds": None,
        "audio_duration_seconds": None,
        "real_time_factor": None,
        "peak_memory_bytes": _peak_memory_bytes(),
        "model_path": str(backend_config.model_path),
        "codec_path": str(backend_config.codec_path),
        "model_sha256": _sha256_file(backend_config.model_path),
        "codec_sha256": _sha256_file(backend_config.codec_path),
        "generation_parameters": {
            "language": backend_config.language,
            "voice_instruction": backend_config.default_instruction,
            "seed": seed,
        },
        "error": init_error,
    }
    if init_error:
        adapter.shutdown()
        return result_payload

    synth_started = time.perf_counter()
    try:
        result = adapter.synthesize(
            VoiceSynthesisRequest(
                text=text,
                voice_instruction=backend_config.default_instruction,
                language=backend_config.language,
                seed=seed,
            )
        )
    except Exception as exc:
        result_payload["error"] = str(exc)
        adapter.shutdown()
        result_payload["synthesis_seconds"] = time.perf_counter() - synth_started
        result_payload["peak_memory_bytes"] = _peak_memory_bytes()
        return result_payload

    synthesis_seconds = time.perf_counter() - synth_started
    duration = result.duration_seconds
    rtf = synthesis_seconds / duration if duration > 0 else None
    result_payload.update(
        {
            "synthesis_seconds": synthesis_seconds,
            "audio_duration_seconds": duration,
            "real_time_factor": rtf,
            "peak_memory_bytes": _peak_memory_bytes(),
            "error": None,
        }
    )
    adapter.shutdown()
    return result_payload


def run_benchmark(
    *,
    config: VoiceConfig | None = None,
    text: str = BENCHMARK_TEXT,
    seed: int = DEFAULT_SEED,
    backends: list[str] | None = None,
) -> list[dict[str, Any]]:
    resolved = (config or VoiceConfig.from_env()).resolve_paths()
    selected = backends or _available_backends(resolved)
    results = [run_backend_benchmark(resolved, backend=backend, text=text, seed=seed) for backend in selected]
    resolved.benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved.benchmarks_path.open("a", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen TTS backends for AI Eco Lab.")
    parser.add_argument("--text", default=BENCHMARK_TEXT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--backends",
        default="",
        help="Comma-separated backend list. Default: auto-detect available backends.",
    )
    args = parser.parse_args(argv)
    backends = [item.strip() for item in args.backends.split(",") if item.strip()] or None
    results = run_benchmark(text=args.text, seed=args.seed, backends=backends)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
