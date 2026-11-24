from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "research" / "stimuli" / "_artifacts"


def persist_artifacts_enabled() -> bool:
    return os.environ.get("AI_ECO_LAB_PERSIST_IMAGE_ARTIFACTS", "0").strip() == "1"


class ImageArtifactStore:
    """Optional local blob store keyed by sha256 prefix."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_ARTIFACT_ROOT

    def store(self, sha256: str, data: bytes, mime_type: str) -> str | None:
        if not persist_artifacts_enabled():
            return None
        prefix = sha256[:2]
        target_dir = self.root / prefix
        target_dir.mkdir(parents=True, exist_ok=True)
        ext = mime_type.split("/")[-1]
        path = target_dir / f"{sha256}.{ext}"
        path.write_bytes(data)
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    def load(self, artifact_ref: str) -> bytes | None:
        path = PROJECT_ROOT / artifact_ref
        if not path.exists() or not path.is_file():
            return None
        return path.read_bytes()
