from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.conditions import stable_json
from instrumentation.types import ObservationRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBSERVATIONS_PATH = PROJECT_ROOT / "research" / "raw" / "observations.jsonl"


class ObservationStore:
    """Append-only store for multi-channel observation records."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or DEFAULT_OBSERVATIONS_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}

    def append(self, record: ObservationRecord) -> dict[str, Any]:
        payload = record.to_dict()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(payload) + "\n")
        self._index[record.observation_id] = payload
        return payload

    def get(self, observation_id: str) -> dict[str, Any] | None:
        if observation_id in self._index:
            return self._index[observation_id]
        if not self.log_path.exists():
            return None
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                oid = record.get("observation_id")
                if oid:
                    self._index[oid] = record
                if oid == observation_id:
                    return record
        return None

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(record)
                oid = record.get("observation_id")
                if oid:
                    self._index[oid] = record
        return records
