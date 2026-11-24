from __future__ import annotations

import json
from pathlib import Path

from instrumentation.store import ObservationStore


def test_legacy_jsonl_without_observation_ids_still_readable(tmp_path: Path):
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp-001",
                "session_id": "sess-1",
                "condition": "C",
                "timestamp": "2026-09-01T16:00:00+00:00",
                "tool_called": "observe_internal_state",
                "metadata": {"legacy": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store = ObservationStore(tmp_path / "observations.jsonl")
    assert store.get("nonexistent-id") is None
    assert legacy_path.exists()
