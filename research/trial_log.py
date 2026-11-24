from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class TrialLogWriter:
    """Append-only trial JSONL writer. Never overwrites existing lines."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path or (DEFAULT_RAW_DIR / "trials.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(record) + "\n")
        return record

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
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def read_trials(self) -> list[dict[str, Any]]:
        return [r for r in self.read_all() if r.get("record_type") == "trial_complete"]
