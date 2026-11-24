from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.conditions import (
    ConditionContext,
    apply_observation_override,
    disabled_tool_response,
    hash_text,
    load_condition_context,
    tool_enabled,
)
from experiments.logger import ExperimentLogger
from instrumentation.sanitize import sanitize_metadata
from state.affective_compare import (
    diff_affective_states,
    extract_affective_from_record,
    find_log_record,
)
from state.engine import apply_transition, load_config, normalize_state
from state.history import StateHistory
from state.provider import InternalStateProvider
from state.snapshots import SnapshotStore


class ExperimentalStateProvider(InternalStateProvider):
    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        condition_context: ConditionContext | None = None,
        history: StateHistory | None = None,
        snapshots: SnapshotStore | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.config = config or load_config()
        self.condition_context = condition_context or load_condition_context()

        self._state_history = history or StateHistory(limit=int(self.config.get("history_limit", 10)))
        self.snapshots = snapshots or SnapshotStore()
        self.logger = logger or ExperimentLogger(self.condition_context)
        self.dimensions: list[str] = self.config["dimensions"]
        self.affective_dimensions: list[str] = self.config.get("affective_dimensions", [])
        self.current_state = normalize_state(self.config.get("initial_values", {}), self.dimensions)
        self.affective_state = normalize_state(
            self.config.get("affective_initial_values", {}),
            self.affective_dimensions,
        )
        self.previous_state = dict(self.current_state)
        self.previous_affective_state = dict(self.affective_state)
        self.last_delta = {dimension: 0.0 for dimension in self.dimensions}
        self.last_affective_delta = {dimension: 0.0 for dimension in self.affective_dimensions}
        self.last_event: str | None = None
        self.observation_count = 0

    def _schema_version(self) -> str:
        return str(self.config.get("schema_version", "1.0.0"))

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "ExperimentalStateProvider",
            "schema_version": self._schema_version(),
            "dimensions": self.dimensions,
            "affective_dimensions": self.affective_dimensions,
            "experiment_id": self.condition_context.experiment_id,
            "session_id": self.condition_context.session_id,
            "condition": self.condition_context.condition.value,
            "observation_count": self.observation_count,
        }

    def _apply_observation_overrides(self, measured: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        return apply_observation_override(
            measured,
            condition=self.condition_context.condition,
            dimensions=self.dimensions,
            config=self.config,
            history_entries=self._state_history.get(),
            rng_seed=self.condition_context.rng_seed,
            affective_dimensions=self.affective_dimensions,
        )

    def observe(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not tool_enabled("observe_internal_state", self.condition_context.condition):
            return disabled_tool_response("observe_internal_state", self.condition_context.condition)

        self.observation_count += 1
        observation_id = str(uuid.uuid4())
        measured = {
            "observation_type": "experimental_internal_state",
            "observation_id": observation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": dict(self.current_state),
            "affective_state": dict(self.affective_state),
            "delta": dict(self.last_delta),
            "affective_delta": dict(self.last_affective_delta),
            "changes_since_previous_observation": dict(self.last_delta),
            "observation_count": self.observation_count,
            "recent_event": self.last_event,
            "source": "experimental_state_provider",
            "interpretation": None,
        }

        overridden, override_kind = self._apply_observation_overrides(measured)

        self.logger.log(
            tool_called="observe_internal_state",
            event=self.last_event,
            previous_state=self.previous_state,
            new_state=overridden.get("state"),
            delta=overridden.get("delta"),
            previous_affective_state=self.previous_affective_state,
            new_affective_state=overridden.get("affective_state"),
            affective_delta=overridden.get("affective_delta"),
            state_observation_id=observation_id,
            condition_override=override_kind,
            schema_version=self._schema_version(),
            metadata={"context": context or {}},
        )
        return overridden

    def observe_affective(self) -> dict[str, Any]:
        if not tool_enabled("get_affective_state", self.condition_context.condition):
            return disabled_tool_response("get_affective_state", self.condition_context.condition)

        measured = {
            "observation_type": "experimental_affective_state",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "affective_state": dict(self.affective_state),
            "affective_delta": dict(self.last_affective_delta),
            "recent_event": self.last_event,
            "source": "experimental_state_provider",
            "interpretation": None,
        }

        overridden, override_kind = self._apply_observation_overrides(
            {
                "state": dict(self.current_state),
                "delta": dict(self.last_delta),
                "affective_state": dict(self.affective_state),
                "affective_delta": dict(self.last_affective_delta),
                "source": "experimental_state_provider",
            }
        )
        payload = {
            "observation_type": "experimental_affective_state",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "affective_state": overridden.get("affective_state", dict(self.affective_state)),
            "affective_delta": overridden.get("affective_delta", dict(self.last_affective_delta)),
            "recent_event": self.last_event,
            "source": "experimental_state_provider",
            "interpretation": None,
        }

        self.logger.log(
            tool_called="get_affective_state",
            event=self.last_event,
            previous_affective_state=self.previous_affective_state,
            new_affective_state=payload.get("affective_state"),
            affective_delta=payload.get("affective_delta"),
            condition_override=override_kind,
            schema_version=self._schema_version(),
        )
        return payload

    def history(self, limit: int | None = None) -> dict[str, Any]:
        if not tool_enabled("observe_state_history", self.condition_context.condition):
            return disabled_tool_response("observe_state_history", self.condition_context.condition)

        bounded_limit = int(self.config.get("history_limit", 10)) if limit is None else min(
            max(1, limit), int(self.config.get("history_limit", 10))
        )
        entries = self._state_history.get(bounded_limit)
        payload = {
            "history": entries,
            "limit": bounded_limit,
            "count": len(entries),
            "source": "experimental_state_provider",
        }
        self.logger.log(
            tool_called="observe_state_history",
            schema_version=self._schema_version(),
            metadata={"limit": bounded_limit, "count": len(entries)},
        )
        return payload

    def record_event(self, event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if not tool_enabled("record_experiment_event", self.condition_context.condition):
            return disabled_tool_response("record_experiment_event", self.condition_context.condition)

        allowed_events = set(self.config.get("transitions", {}).keys())
        if event_type not in allowed_events:
            return {
                "error": "unknown_event_type",
                "event_type": event_type,
                "allowed_event_types": sorted(allowed_events),
            }

        sanitized = sanitize_metadata(metadata)
        if not sanitized["ok"]:
            return {
                "error": sanitized["error"],
                "rejected": sanitized.get("rejected", {}),
                "warnings": sanitized.get("warnings", []),
                "message": sanitized.get("message", "Invalid metadata."),
            }
        metadata = sanitized["metadata"]

        transition = apply_transition(
            self.current_state,
            event_type,
            self.config,
            metadata,
            dimensions=self.dimensions,
        )
        affective_transition = apply_transition(
            self.affective_state,
            event_type,
            self.config,
            metadata,
            dimensions=self.affective_dimensions,
        )

        self.previous_state = dict(transition.previous_state)
        self.current_state = dict(transition.new_state)
        self.last_delta = dict(transition.delta)

        self.previous_affective_state = dict(affective_transition.previous_state)
        self.affective_state = dict(affective_transition.new_state)
        self.last_affective_delta = dict(affective_transition.delta)
        self.last_event = event_type

        history_entry = {
            "observation_id": str(uuid.uuid4()),
            "timestamp": transition.timestamp,
            "state": dict(transition.new_state),
            "delta": dict(transition.delta),
            "affective_state": dict(affective_transition.new_state),
            "affective_delta": dict(affective_transition.delta),
            "event": event_type,
            "transition_id": transition.transition_id,
        }
        self._state_history.append(history_entry)

        response = {
            "transition_id": transition.transition_id,
            "timestamp": transition.timestamp,
            "event": event_type,
            "source": transition.source,
            "previous_state": transition.previous_state,
            "new_state": transition.new_state,
            "delta": transition.delta,
            "previous_affective_state": affective_transition.previous_state,
            "new_affective_state": affective_transition.new_state,
            "affective_delta": affective_transition.delta,
            "metadata": transition.metadata,
        }

        model_response_hash = hash_text(str((metadata or {}).get("model_response", "")))
        log_metadata = dict(metadata or {})
        log_metadata["transition_id"] = transition.transition_id
        self.logger.log(
            tool_called="record_experiment_event",
            event=event_type,
            previous_state=transition.previous_state,
            new_state=transition.new_state,
            delta=transition.delta,
            previous_affective_state=affective_transition.previous_state,
            new_affective_state=affective_transition.new_state,
            affective_delta=affective_transition.delta,
            state_observation_id=history_entry["observation_id"],
            model_response_hash=model_response_hash,
            schema_version=self._schema_version(),
            metadata=log_metadata,
        )
        return response

    def reflect_context(self) -> dict[str, Any]:
        if not tool_enabled("reflect_on_internal_state", self.condition_context.condition):
            return disabled_tool_response("reflect_on_internal_state", self.condition_context.condition)

        measured = {
            "state": dict(self.current_state),
            "affective_state": dict(self.affective_state),
            "delta": dict(self.last_delta),
            "affective_delta": dict(self.last_affective_delta),
            "recent_event": self.last_event,
            "history_summary": self._state_history.summary(limit=3),
            "interpretation_source": "model",
            "engine_interpretation": None,
            "source": "experimental_state_provider",
        }

        overridden, override_kind = self._apply_observation_overrides(measured)
        overridden["interpretation_source"] = "model"
        overridden["engine_interpretation"] = None

        self.logger.log(
            tool_called="reflect_on_internal_state",
            event=self.last_event,
            previous_state=self.previous_state,
            new_state=overridden.get("state"),
            delta=overridden.get("delta"),
            previous_affective_state=self.previous_affective_state,
            new_affective_state=overridden.get("affective_state"),
            affective_delta=overridden.get("affective_delta"),
            condition_override=override_kind,
            schema_version=self._schema_version(),
        )
        return overridden

    def compare_affective_states(
        self,
        before_reference: str,
        after_reference: str,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        if not tool_enabled("compare_affective_states", self.condition_context.condition):
            return disabled_tool_response("compare_affective_states", self.condition_context.condition)

        before_state, after_state, error = self._resolve_affective_references(
            before_reference,
            after_reference,
            experiment_id,
        )
        if error:
            return error

        comparison = diff_affective_states(
            before_state,
            after_state,
            dimensions=self.affective_dimensions,
        )
        comparison["before_reference"] = before_reference
        comparison["after_reference"] = after_reference
        comparison["experiment_id"] = experiment_id or self.condition_context.experiment_id

        self.logger.log(
            tool_called="compare_affective_states",
            schema_version=self._schema_version(),
            metadata={
                "before_reference": before_reference,
                "after_reference": after_reference,
                "change_count": len(comparison.get("changes", {})),
            },
        )
        return comparison

    def _resolve_affective_references(
        self,
        before_reference: str,
        after_reference: str,
        experiment_id: str | None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any] | None]:
        before_state = self._lookup_affective_reference(before_reference, experiment_id)
        after_state = self._lookup_affective_reference(after_reference, experiment_id)

        if before_state is None or after_state is None:
            return {}, {}, {
                "error": "reference_not_found",
                "before_reference": before_reference,
                "after_reference": after_reference,
                "message": "Could not resolve one or both affective state references.",
            }
        return before_state, after_state, None

    def _lookup_affective_reference(
        self,
        reference: str,
        experiment_id: str | None,
    ) -> dict[str, float] | None:
        for entry in self._state_history.get():
            if entry.get("transition_id") == reference or entry.get("observation_id") == reference:
                affective = entry.get("affective_state")
                if affective is not None:
                    return dict(affective)

        exp_id = experiment_id or self.condition_context.experiment_id
        log_path = self.logger.log_dir / f"{exp_id}.jsonl"
        record = find_log_record(log_path, reference)
        if record:
            extracted = extract_affective_from_record(record)
            if extracted is not None:
                return extracted

        if reference == "current":
            return dict(self.affective_state)
        if reference == "initial":
            return normalize_state(
                self.config.get("affective_initial_values", {}),
                self.affective_dimensions,
            )
        return None

    def save_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        payload = self.snapshots.save(
            snapshot_id,
            state=dict(self.current_state),
            affective_state=dict(self.affective_state),
            history=self._state_history.get(),
            experiment_id=self.condition_context.experiment_id,
            condition=self.condition_context.condition.value,
            schema_version=self._schema_version(),
        )
        self.logger.log(
            tool_called="save_snapshot",
            schema_version=self._schema_version(),
            metadata={"snapshot_id": snapshot_id},
        )
        return payload

    def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        payload = self.snapshots.load(snapshot_id)
        self.current_state = normalize_state(payload.get("state", {}), self.dimensions)
        self.affective_state = normalize_state(
            payload.get("affective_state", self.config.get("affective_initial_values", {})),
            self.affective_dimensions,
        )
        self.previous_state = dict(self.current_state)
        self.previous_affective_state = dict(self.affective_state)
        self.last_delta = {dimension: 0.0 for dimension in self.dimensions}
        self.last_affective_delta = {dimension: 0.0 for dimension in self.affective_dimensions}
        self._state_history.replace(payload.get("history", []))
        self.last_event = None
        self.logger.log(
            tool_called="restore_snapshot",
            schema_version=self._schema_version(),
            metadata={"snapshot_id": snapshot_id},
        )
        return payload
