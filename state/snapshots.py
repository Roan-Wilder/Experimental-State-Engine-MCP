from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"


class SnapshotStore:
    def __init__(self, snapshot_dir: Path | None = None) -> None:
        self.snapshot_dir = snapshot_dir or DEFAULT_SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, snapshot_id: str) -> Path:
        safe_id = snapshot_id.replace("\\", "_").replace("/", "_")
        return self.snapshot_dir / f"{safe_id}.json"

    def save(
        self,
        snapshot_id: str,
        *,
        state: dict[str, float],
        history: list[dict[str, Any]],
        experiment_id: str,
        condition: str,
        schema_version: str,
        affective_state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "snapshot_id": snapshot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "affective_state": affective_state or {},
            "history": history,
            "experiment_id": experiment_id,
            "condition": condition,
            "schema_version": schema_version,
        }
        path = self._path_for(snapshot_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def load(self, snapshot_id: str) -> dict[str, Any]:
        path = self._path_for(snapshot_id)
        if not path.exists():
            raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found.")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def exists(self, snapshot_id: str) -> bool:
        return self._path_for(snapshot_id).exists()
