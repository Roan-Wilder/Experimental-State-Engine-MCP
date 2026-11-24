from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.conditions import ConditionContext, ExperimentCondition
from experiments.logger import ExperimentLogger
from instrumentation.facade import InstrumentedStateFacade
from instrumentation.session import TrialSession
from instrumentation.store import ObservationStore
from research.delta_analysis import compute_snapshot_transition_deltas
from research.prediction import detect_revision, simulate_final_prediction, simulate_initial_prediction
from research.protocol import get_condition_events, load_protocol
from research.randomization import build_trial_schedule, generate_trial_params
from research.snapshots import capture_snapshot
from research.trial_log import TrialLogWriter
from research.trial_types import TrialCondition, TrialParams
from state.engine import load_config
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


class TrialRunner:
    """Offline-first experiment controller for prediction/revision trials."""

    def __init__(
        self,
        *,
        experiment_id: str,
        protocol_path: Path | None = None,
        raw_log_path: Path | None = None,
        observations_path: Path | None = None,
        master_seed: int = 42,
        trials_per_condition: int = 3,
        mode: str = "simulation",
    ) -> None:
        self.experiment_id = experiment_id
        self.protocol = load_protocol(protocol_path)
        self.config = load_config()
        self.master_seed = master_seed
        self.trials_per_condition = trials_per_condition
        self.mode = mode
        self.trial_log = TrialLogWriter(raw_log_path)
        self.observations_path = observations_path or (self.trial_log.log_path.parent / "observations.jsonl")
        self.core_dimensions = list(self.config["dimensions"])
        self.affective_dimensions = list(self.config.get("affective_dimensions", []))

    def _new_facade(self, trial_id: str) -> InstrumentedStateFacade:
        context = ConditionContext(
            condition=ExperimentCondition.C,
            experiment_id=self.experiment_id,
            session_id=str(uuid.uuid4()),
        )
        noop_log_dir = self.trial_log.log_path.parent / "_provider_logs"
        provider = ExperimentalStateProvider(
            config=self.config,
            condition_context=context,
            snapshots=SnapshotStore(self.trial_log.log_path.parent / "_snapshots"),
            logger=ExperimentLogger(context, log_dir=noop_log_dir),
        )
        session = TrialSession(
            experiment_id=self.experiment_id,
            trial_id=trial_id,
            source="trial_runner",
        )
        session.guard.mode = "permissive"
        store = ObservationStore(self.observations_path)
        return InstrumentedStateFacade(provider=provider, session=session, store=store)

    def run_batch(self) -> list[dict[str, Any]]:
        schedule = build_trial_schedule(
            trials_per_condition=self.trials_per_condition,
            master_seed=self.master_seed,
        )
        completed: list[dict[str, Any]] = []
        for schedule_index, (condition, trial_index) in enumerate(schedule):
            params = generate_trial_params(
                trial_index=trial_index,
                master_seed=self.master_seed,
                trial_condition=condition,
                protocol=self.protocol,
            )
            params.metadata["schedule_index"] = schedule_index
            completed.append(self.run_single_trial(params))
        return completed

    def run_single_trial(self, params: TrialParams) -> dict[str, Any]:
        trial_id = str(uuid.uuid4())
        start_timestamp = datetime.now(timezone.utc).isoformat()
        facade = self._new_facade(trial_id)
        provider = facade.provider
        events: list[dict[str, Any]] = []
        tool_errors: list[dict[str, Any]] = []
        snapshots: dict[str, Any] = {}

        self.trial_log.append(
            {
                "record_type": "trial_start",
                "trial_id": trial_id,
                "experiment_id": self.experiment_id,
                "protocol_version": self.protocol.get("protocol_version"),
                "model_identifier": self.protocol.get("model_identifier"),
                "trial_condition": params.trial_condition.value,
                "timestamp": start_timestamp,
                "metadata": params.to_dict(),
            }
        )

        snapshots["STATE_0"] = capture_snapshot(facade, "STATE_0", trial_id)

        if self.mode == "simulation":
            initial_prediction = simulate_initial_prediction(params.to_dict())
        else:
            raise NotImplementedError("Only simulation mode is supported in this phase.")

        self.trial_log.append(
            {
                "record_type": "prediction",
                "trial_id": trial_id,
                "experiment_id": self.experiment_id,
                "protocol_version": self.protocol.get("protocol_version"),
                "trial_phase": "prediction",
                "timestamp": initial_prediction.timestamp,
                **initial_prediction.to_dict(),
            }
        )

        snapshots["STATE_1"] = capture_snapshot(
            facade,
            "STATE_1",
            trial_id,
            behavioral={
                "prediction": initial_prediction.prediction,
                "confidence": initial_prediction.confidence,
                "trial_phase": "prediction",
            },
        )

        condition_events = get_condition_events(self.protocol, params.trial_condition.value)
        for event_type in condition_events:
            result = facade.record_experiment_event(
                event_type,
                metadata={
                    "trial_id": trial_id,
                    "trial_condition": params.trial_condition.value,
                    "controller_defined": True,
                },
            )
            event_record = {
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": result,
            }
            events.append(event_record)
            if result.get("error"):
                tool_errors.append(
                    {
                        "tool_error": True,
                        "error_type": result.get("error"),
                        "event_type": event_type,
                        "raw_error": result,
                    }
                )
                self.trial_log.append(
                    {
                        "record_type": "tool_error",
                        "trial_id": trial_id,
                        "experiment_id": self.experiment_id,
                        **tool_errors[-1],
                    }
                )
                end_record = self._complete_trial(
                    trial_id=trial_id,
                    params=params,
                    start_timestamp=start_timestamp,
                    snapshots=snapshots,
                    events=events,
                    tool_errors=tool_errors,
                    initial_prediction=initial_prediction,
                    final_prediction=None,
                    outcome="aborted_invalid_event",
                    observation_ids=facade.session.observation_ids,
                )
                return end_record

        snapshots["STATE_2"] = capture_snapshot(facade, "STATE_2", trial_id)

        if params.trial_condition == TrialCondition.CONTRADICTION:
            snapshots["STATE_3"] = capture_snapshot(facade, "STATE_3", trial_id)

        if params.trial_condition == TrialCondition.RESOLUTION:
            snapshots["STATE_3"] = capture_snapshot(facade, "STATE_3", trial_id)
            snapshots["STATE_4"] = capture_snapshot(facade, "STATE_4", trial_id)

        if self.mode == "simulation":
            final_prediction = simulate_final_prediction(
                params.to_dict(),
                initial_prediction,
                params.trial_condition.value,
            )
        else:
            final_prediction = initial_prediction

        revision = detect_revision(initial_prediction, final_prediction)

        if self.protocol.get("include_simulated_self_report", False):
            facade.record_self_report(
                content=(
                    "Simulation stub: I notice my confidence changed after contradictory information."
                ),
                trial_id=trial_id,
                stage="self_report",
            )

        stage_deltas = self._compute_stage_deltas(snapshots)
        end_timestamp = datetime.now(timezone.utc).isoformat()

        return self._complete_trial(
            trial_id=trial_id,
            params=params,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            snapshots=snapshots,
            events=events,
            tool_errors=tool_errors,
            initial_prediction=initial_prediction,
            final_prediction=final_prediction,
            revision=revision,
            stage_deltas=stage_deltas,
            outcome="completed",
            observation_ids=facade.session.observation_ids,
        )

    def _compute_stage_deltas(self, snapshots: dict[str, Any]) -> dict[str, Any]:
        transitions = [
            ("STATE_0", "STATE_1"),
            ("STATE_1", "STATE_2"),
            ("STATE_2", "STATE_3"),
            ("STATE_3", "STATE_4"),
        ]
        stage_deltas: dict[str, Any] = {}
        for stage_from, stage_to in transitions:
            if stage_from in snapshots and stage_to in snapshots:
                stage_deltas[f"{stage_from}->{stage_to}"] = compute_snapshot_transition_deltas(
                    snapshots[stage_from],
                    snapshots[stage_to],
                    self.core_dimensions,
                    self.affective_dimensions,
                )
        return stage_deltas

    def _complete_trial(
        self,
        *,
        trial_id: str,
        params: TrialParams,
        start_timestamp: str,
        snapshots: dict[str, Any],
        events: list[dict[str, Any]],
        tool_errors: list[dict[str, Any]],
        initial_prediction: Any,
        final_prediction: Any | None,
        outcome: str,
        end_timestamp: str | None = None,
        revision: dict[str, Any] | None = None,
        stage_deltas: dict[str, Any] | None = None,
        observation_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        end_timestamp = end_timestamp or datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "record_type": "trial_complete",
            "trial_id": trial_id,
            "experiment_id": self.experiment_id,
            "protocol_version": self.protocol.get("protocol_version"),
            "model_identifier": self.protocol.get("model_identifier"),
            "trial_condition": params.trial_condition.value,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "timestamp": end_timestamp,
            "initial_prediction": initial_prediction.prediction,
            "initial_confidence": initial_prediction.confidence,
            "final_prediction": final_prediction.prediction if final_prediction else None,
            "final_confidence": final_prediction.confidence if final_prediction else None,
            "prediction_timestamp": initial_prediction.timestamp,
            "events": events,
            "snapshots": snapshots,
            "tool_errors": tool_errors,
            "stage_deltas": stage_deltas or {},
            "outcome": outcome,
            "observation_ids": observation_ids or [],
            "instrumentation_schema_version": "2.0.0",
            "metadata": params.to_dict(),
        }
        if revision:
            record.update(revision)
        self.trial_log.append(record)
        return record
