from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.conditions import stable_json
from instrumentation.vision.artifact_store import ImageArtifactStore
from instrumentation.vision.types import ImageProvenanceRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROVENANCE_PATH = PROJECT_ROOT / "research" / "raw" / "image_provenance.jsonl"


class ImageProvenanceStore:
    """Append-only store for image provenance records."""

    def __init__(
        self,
        log_path: Path | None = None,
        artifact_store: ImageArtifactStore | None = None,
    ) -> None:
        self.log_path = log_path or DEFAULT_PROVENANCE_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store or ImageArtifactStore()
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_hash: dict[str, list[str]] = {}

    def append(self, record: ImageProvenanceRecord, *, data: bytes | None = None) -> dict[str, Any]:
        existing_ids = self._by_hash.get(record.sha256, [])
        if existing_ids:
            record.duplicate_of_image_id = existing_ids[0]

        if data is not None:
            artifact_ref = self.artifact_store.store(record.sha256, data, record.mime_type)
            if artifact_ref:
                record.artifact_stored = True
                record.artifact_ref = artifact_ref

        payload = record.to_dict()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(payload) + "\n")

        self._by_id[record.image_id] = payload
        self._by_hash.setdefault(record.sha256, []).append(record.image_id)
        return payload

    def get(self, image_id: str) -> dict[str, Any] | None:
        if image_id in self._by_id:
            return self._by_id[image_id]
        self._load_index()
        return self._by_id.get(image_id)

    def find_by_hash(self, sha256: str) -> list[dict[str, Any]]:
        self._load_index()
        return [self._by_id[iid] for iid in self._by_hash.get(sha256, []) if iid in self._by_id]

    def read_all(self) -> list[dict[str, Any]]:
        self._load_index()
        return list(self._by_id.values())

    def _load_index(self) -> None:
        if not self.log_path.exists():
            return
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_id = record.get("image_id")
                sha256 = record.get("sha256")
                if image_id:
                    self._by_id[image_id] = record
                if image_id and sha256:
                    self._by_hash.setdefault(sha256, [])
                    if image_id not in self._by_hash[sha256]:
                        self._by_hash[sha256].append(image_id)
