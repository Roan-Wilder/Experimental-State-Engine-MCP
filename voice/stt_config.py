from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

VALID_STT_BACKENDS = frozenset({"faster_whisper", "null", "mock"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "True", "yes", "YES"}


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class STTConfig:
    enabled: bool
    backend: str
    model: str
    model_path: Path | None
    device: str
    compute_type: str
    language: str
    sample_rate: int
    max_record_seconds: float
    events_path: Path
    temp_dir: Path
    persist_audio: bool
    allow_model_download: bool
    hardware_test: bool

    @classmethod
    def from_env(cls, *, repo_root: Path | None = None) -> STTConfig:
        root = repo_root or _repo_root()
        model_path_raw = os.environ.get("STT_MODEL_PATH", "").strip()
        return cls(
            enabled=_env_bool("STT_ENABLED", False),
            backend=os.environ.get("STT_BACKEND", "faster_whisper").strip(),
            model=os.environ.get("STT_MODEL", "base").strip(),
            model_path=Path(model_path_raw) if model_path_raw else None,
            device=os.environ.get("STT_DEVICE", "cpu").strip(),
            compute_type=os.environ.get("STT_COMPUTE_TYPE", "int8").strip(),
            language=os.environ.get("STT_LANGUAGE", "en").strip(),
            sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000")),
            max_record_seconds=float(os.environ.get("STT_MAX_RECORD_SECONDS", "10")),
            events_path=Path(os.environ.get("STT_EVENTS_PATH", "research/raw/stt_events.jsonl")),
            temp_dir=Path(os.environ.get("STT_TEMP_DIR", "tmp_stt")),
            persist_audio=_env_bool("STT_PERSIST_AUDIO", False),
            allow_model_download=_env_bool("STT_ALLOW_MODEL_DOWNLOAD", False),
            hardware_test=_env_bool("STT_HARDWARE_TEST", False),
        ).resolve_paths(root)

    def resolve_paths(self, repo_root: Path | None = None) -> STTConfig:
        root = repo_root or _repo_root()

        def _resolve(path: Path) -> Path:
            return (root / path).resolve() if not path.is_absolute() else path.resolve()

        model_path = self.model_path
        if model_path is not None:
            model_path = _resolve(model_path)

        return STTConfig(
            enabled=self.enabled,
            backend=self.backend,
            model=self.model,
            model_path=model_path,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            sample_rate=self.sample_rate,
            max_record_seconds=self.max_record_seconds,
            events_path=_resolve(self.events_path),
            temp_dir=_resolve(self.temp_dir),
            persist_audio=self.persist_audio,
            allow_model_download=self.allow_model_download,
            hardware_test=self.hardware_test,
        )

    def validate(self) -> dict[str, str] | None:
        if self.backend not in VALID_STT_BACKENDS:
            return {
                "error": "invalid_backend",
                "message": f"STT_BACKEND must be one of {sorted(VALID_STT_BACKENDS)}.",
            }
        if self.max_record_seconds <= 0:
            return {
                "error": "invalid_max_record_seconds",
                "message": "STT_MAX_RECORD_SECONDS must be positive.",
            }
        if self.sample_rate <= 0:
            return {
                "error": "invalid_sample_rate",
                "message": "STT_SAMPLE_RATE must be positive.",
            }
        if self.backend == "faster_whisper":
            if not faster_whisper_available():
                return {
                    "error": "missing_stt_dependencies",
                    "message": (
                        "faster-whisper is not installed. "
                        'Install STT extras: pip install -e ".[stt]"'
                    ),
                }
            if self.model_path is not None and not self.model_path.exists():
                return {
                    "error": "missing_model_path",
                    "message": f"STT model path not found: {self.model_path}",
                }
        if self.backend in {"faster_whisper", "mock"} and self.backend == "faster_whisper":
            pass
        return None

    def clamp_duration(self, duration_seconds: float) -> float:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        return min(duration_seconds, self.max_record_seconds)

    def model_label(self) -> str:
        if self.model_path is not None:
            return self.model_path.name
        return self.model
