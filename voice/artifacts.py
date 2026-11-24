from __future__ import annotations

import uuid
from pathlib import Path


class VoiceArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _validate_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"Artifact root is not a directory: {self.root}")

    def write_wav(self, wav_bytes: bytes) -> str:
        self._validate_root()
        artifact_id = str(uuid.uuid4())
        target = self.root / f"{artifact_id}.wav"
        resolved = target.resolve()
        if self.root not in resolved.parents:
            raise ValueError("Artifact path escapes allowlisted root.")
        target.write_bytes(wav_bytes)
        return artifact_id

    def resolve_artifact_path(self, artifact_id: str) -> Path:
        if not artifact_id or "/" in artifact_id or "\\" in artifact_id or ".." in artifact_id:
            raise ValueError("Invalid artifact id.")
        target = (self.root / f"{artifact_id}.wav").resolve()
        if self.root not in target.parents:
            raise ValueError("Artifact path escapes allowlisted root.")
        return target

    def write_persona_reference(self, voice_id: str, version: int, wav_bytes: bytes) -> str:
        if not voice_id or "/" in voice_id or "\\" in voice_id or ".." in voice_id:
            raise ValueError("Invalid voice id.")
        persona_dir = (self.root / "personas" / voice_id).resolve()
        if self.root not in persona_dir.parents:
            raise ValueError("Persona path escapes allowlisted root.")
        persona_dir.mkdir(parents=True, exist_ok=True)
        target = (persona_dir / f"v{version}.wav").resolve()
        if self.root not in target.parents:
            raise ValueError("Persona artifact path escapes allowlisted root.")
        target.write_bytes(wav_bytes)
        return f"personas/{voice_id}/v{version}.wav"
