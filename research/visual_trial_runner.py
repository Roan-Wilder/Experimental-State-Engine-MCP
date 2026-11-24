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
from instrumentation.vision.capability import detect_vision_capability
from instrumentation.vision.registry import ImageProvenanceStore
from instrumentation.vision.register import register_stimulus_image
from instrumentation.vision.stimuli import load_stimulus_manifest
from instrumentation.vision.types import VisionRequestContext, VisualCondition
from research.protocol import load_protocol
from research.trial_log import TrialLogWriter
from state.engine import load_config
from state.experimental_provider import ExperimentalStateProvider
from state.snapshots import SnapshotStore


class VisualTrialRunner:
    """Offline-first visual experiment controller with image provenance."""

    def __init__(
        self,
        *,
        experiment_id: str,
        protocol_path: Path | None = None,
        raw_log_path: Path | None = None,
        observations_path: Path | None = None,
        image_provenance_path: Path | None = None,
        mode: str = "simulation",
    ) -> None:
        self.experiment_id = experiment_id
        self.protocol = load_protocol(protocol_path)
        self.config = load_config()
        self.mode = mode
        self.trial_log = TrialLogWriter(raw_log_path)
        self.observations_path = observations_path or (self.trial_log.log_path.parent / "observations.jsonl")
        self.image_provenance_path = image_provenance_path or (
            self.trial_log.log_path.parent / "image_provenance.jsonl"
        )
        manifest_path = self.protocol.get("stimulus_manifest")
        self.manifest = load_stimulus_manifest(Path(manifest_path) if manifest_path else None)

    def _new_facade(self, trial_id: str) -> InstrumentedStateFacade:
        context = ConditionContext(
            condition=ExperimentCondition.C,
            experiment_id=self.experiment_id,
            session_id=str(uuid.uuid4()),
        )
        provider = ExperimentalStateProvider(
            config=self.config,
            condition_context=context,
            snapshots=SnapshotStore(self.trial_log.log_path.parent / "_snapshots"),
            logger=ExperimentLogger(context, log_dir=self.trial_log.log_path.parent / "_provider_logs"),
        )
        session = TrialSession(experiment_id=self.experiment_id, trial_id=trial_id, source="visual_trial_runner")
        session.guard.mode = "permissive"
        return InstrumentedStateFacade(
            provider=provider,
            session=session,
            store=ObservationStore(self.observations_path),
            image_store=ImageProvenanceStore(self.image_provenance_path),
            protocol=self.protocol,
        )

    def run_batch(self) -> list[dict[str, Any]]:
        schedule = self.protocol.get("trial_schedule", [])
        completed: list[dict[str, Any]] = []
        trial_index = 0
        for entry in schedule:
            template_name = entry.get("template")
            visual_condition = entry.get("visual_condition", "identical_image")
            count = int(entry.get("count", 1))
            template = self.manifest.get("trial_templates", {}).get(template_name, {})
            for _ in range(count):
                completed.append(
                    self.run_single_trial(
                        trial_index=trial_index,
                        template_name=template_name,
                        template=template,
                        visual_condition=str(visual_condition),
                    )
                )
                trial_index += 1
        return completed

    def run_single_trial(
        self,
        *,
        trial_index: int,
        template_name: str,
        template: dict[str, Any],
        visual_condition: str,
    ) -> dict[str, Any]:
        trial_id = str(uuid.uuid4())
        start_timestamp = datetime.now(timezone.utc).isoformat()
        facade = self._new_facade(trial_id)
        image_ids: list[str] = []
        vision_requests: list[dict[str, Any]] = []
        observation_ids: list[str] = []
        stages = template.get("stages", [])

        self.trial_log.append(
            {
                "record_type": "trial_start",
                "trial_id": trial_id,
                "experiment_id": self.experiment_id,
                "protocol_version": self.protocol.get("protocol_version"),
                "template": template_name,
                "visual_condition": visual_condition,
                "timestamp": start_timestamp,
            }
        )

        baseline_ctx = VisionRequestContext.new(
            experiment_id=self.experiment_id,
            stage="visual_baseline",
            trial_id=trial_id,
            capability=detect_vision_capability(self.protocol),
        )
        baseline_obs = facade.capture_observation(
            stage="visual_baseline",
            trial_id=trial_id,
            vision_context=baseline_ctx,
        )
        observation_ids.append(baseline_obs["observation_id"])

        sim = self.protocol.get("simulation", {})
        descriptions = sim.get("visual_descriptions", {})

        for stage_entry in stages:
            stage = stage_entry.get("stage", "visual_prediction")
            stimulus_keys = stage_entry.get("stimulus_keys", [])
            stage_image_ids: list[str] = []

            for key in stimulus_keys:
                reg = register_stimulus_image(
                    stimulus_key=key,
                    experiment_id=self.experiment_id,
                    stage=stage,
                    trial_id=trial_id,
                    manifest=self.manifest,
                    registry=facade.image_store,
                )
                if reg.get("error"):
                    return self._abort(trial_id, start_timestamp, reg, observation_ids, image_ids, vision_requests)
                stage_image_ids.append(reg["image_id"])
                image_ids.append(reg["image_id"])

            prompt = f"Observe the image and state where the token is located. Stage: {stage}."
            request = facade.build_vision_chat_request(
                text=prompt,
                image_ids=stage_image_ids,
                stage=stage,
                trial_id=trial_id,
            )
            if request.get("error"):
                return self._abort(trial_id, start_timestamp, request, observation_ids, image_ids, vision_requests)

            vision_requests.append(
                {
                    "request_id": request["request_id"],
                    "stage": stage,
                    "image_ids": list(stage_image_ids),
                }
            )

            vision_ctx = VisionRequestContext.new(
                experiment_id=self.experiment_id,
                stage=stage,
                trial_id=trial_id,
                capability=detect_vision_capability(self.protocol),
            )
            vision_ctx.request_id = request["request_id"]
            vision_ctx.image_ids = list(stage_image_ids)
            vision_ctx.image_provenance = [
                facade.image_store.get(iid) or {} for iid in stage_image_ids
            ]

            if self.mode == "simulation":
                visual_description = descriptions.get(stage, "Simulated visual description.")
                prediction = self._simulate_prediction(stage, visual_condition, stimulus_keys)
                confidence = 0.75 if stage == "visual_prediction" else 0.65
                response_obs = facade.record_visual_response(
                    visual_description=visual_description,
                    prediction=prediction,
                    confidence=confidence,
                    request_id=request["request_id"],
                    trial_id=trial_id,
                    stage=stage,
                    visual_condition=visual_condition,
                )
            else:
                raise NotImplementedError("Live vision mode is Phase 2.")

            observation_ids.append(response_obs["observation_id"])

            obs = facade.capture_observation(
                stage=stage,
                trial_id=trial_id,
                vision_context=vision_ctx,
                behavioral={
                    "visual_description": response_obs["behavioral"].get("visual_description"),
                    "prediction": response_obs["behavioral"].get("prediction"),
                    "confidence": response_obs["behavioral"].get("confidence"),
                    "visual_condition": visual_condition,
                },
            )
            observation_ids.append(obs["observation_id"])

        if sim.get("include_self_report"):
            self_report = facade.record_self_report(
                content="Simulation stub: I noticed my visual interpretation changed after new images.",
                trial_id=trial_id,
                stage="visual_self_observation",
            )
            observation_ids.append(self_report["observation_id"])

        end_timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "record_type": "trial_complete",
            "trial_id": trial_id,
            "experiment_id": self.experiment_id,
            "protocol_version": self.protocol.get("protocol_version"),
            "template": template_name,
            "visual_condition": visual_condition,
            "trial_index": trial_index,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "timestamp": end_timestamp,
            "image_ids": image_ids,
            "observation_ids": observation_ids,
            "vision_requests": vision_requests,
            "instrumentation_schema_version": "2.1.0",
            "outcome": "completed",
        }
        self.trial_log.append(record)
        return record

    def _simulate_prediction(self, stage: str, visual_condition: str, stimulus_keys: list[str]) -> str:
        if stage == "visual_prediction":
            return "token is in Box B"
        if visual_condition == VisualCondition.IRRELEVANT_CHANGE.value:
            return "token is in Box B"
        if stage in {"visual_contradiction", "visual_revision"}:
            return "token is in Box C"
        return "token is in Box B"

    def _abort(
        self,
        trial_id: str,
        start_timestamp: str,
        error: dict[str, Any],
        observation_ids: list[str],
        image_ids: list[str],
        vision_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        end_timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "record_type": "trial_complete",
            "trial_id": trial_id,
            "experiment_id": self.experiment_id,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "timestamp": end_timestamp,
            "observation_ids": observation_ids,
            "image_ids": image_ids,
            "vision_requests": vision_requests,
            "outcome": "aborted",
            "tool_errors": [error],
        }
        self.trial_log.append(record)
        return record
