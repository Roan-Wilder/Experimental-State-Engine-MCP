from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.conditions import ConditionContext, stable_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "experiments"


class ExperimentLogger:
    def __init__(self, context: ConditionContext, log_dir: Path | None = None) -> None:
        self.context = context
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{context.experiment_id}.jsonl"

    def log(
        self,
        *,
        tool_called: str | None = None,
        event: str | None = None,
        previous_state: dict[str, float] | None = None,
        new_state: dict[str, float] | None = None,
        delta: dict[str, float] | None = None,
        previous_affective_state: dict[str, float] | None = None,
        new_affective_state: dict[str, float] | None = None,
        affective_delta: dict[str, float] | None = None,
        state_observation_id: str | None = None,
        model_response_hash: str | None = None,
        condition_override: str | None = None,
        schema_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "experiment_id": self.context.experiment_id,
            "session_id": self.context.session_id,
            "condition": self.context.condition.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "previous_state": previous_state,
            "new_state": new_state,
            "delta": delta,
            "previous_affective_state": previous_affective_state,
            "new_affective_state": new_affective_state,
            "affective_delta": affective_delta,
            "tool_called": tool_called,
            "model_response_hash": model_response_hash,
            "state_observation_id": state_observation_id,
            "condition_override": condition_override,
            "schema_version": schema_version,
            "metadata": metadata or {},
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(stable_json(record) + "\n")
        return record
