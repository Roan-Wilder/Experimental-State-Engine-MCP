from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from instrumentation.adapters.internal_activation import NullInternalActivationAdapter
from instrumentation.adapters.model_io import NullModelIOAdapter, VisionModelIOAdapter
from instrumentation.builder import build_observation_record
from instrumentation.sanitize import (
    sanitize_metadata,
    sanitize_self_report,
    validate_observation_reference,
)
from instrumentation.session import TrialSession
from instrumentation.store import ObservationStore
from instrumentation.types import ObservationRecord
from instrumentation.vision.capability import detect_vision_capability
from instrumentation.vision.registry import ImageProvenanceStore
from instrumentation.vision.register import register_stimulus_image
from instrumentation.vision.request_builder import LMStudioVisionRequestBuilder
from instrumentation.vision.stimuli import load_stimulus_manifest
from instrumentation.vision.types import VisionRequestContext
from state.affective_compare import diff_affective_states
from state.experimental_provider import ExperimentalStateProvider


def _allow_keyword_refs() -> bool:
    return os.environ.get("AI_ECO_LAB_ALLOW_KEYWORD_REFS", "").strip() == "1"


def _vision_enabled() -> bool:
    return os.environ.get("AI_ECO_LAB_VISION_ENABLED", "0").strip() == "1"


class InstrumentedStateFacade:
    """Wraps ExperimentalStateProvider with multi-channel instrumentation."""

    def __init__(
        self,
        provider: ExperimentalStateProvider | None = None,
        *,
        session: TrialSession | None = None,
        store: ObservationStore | None = None,
        image_store: ImageProvenanceStore | None = None,
        protocol: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider or ExperimentalStateProvider()
        experiment_id = self.provider.condition_context.experiment_id
        self.session = session or TrialSession.from_env(experiment_id)
        self.store = store or ObservationStore()
        self.image_store = image_store or ImageProvenanceStore()
        self.protocol = protocol or {}
        self.vision_builder = LMStudioVisionRequestBuilder(self.image_store)
        self.internal_activation_adapter = NullInternalActivationAdapter()
        self._vision_capability = detect_vision_capability(self.protocol)

    def _model_io_adapter(self, vision_context: VisionRequestContext | None):
        if vision_context is not None:
            return VisionModelIOAdapter()
        return NullModelIOAdapter()

    def observe_internal_state(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = self.provider.observe(context=context)
        payload["instrumentation_observation_id"] = payload.get("observation_id")
        return payload

    def observe_state_history(self, limit: int = 10) -> dict[str, Any]:
        return self.provider.history(limit=limit)

    def reflect_on_internal_state(self) -> dict[str, Any]:
        return self.provider.reflect_context()

    def get_affective_state(self) -> dict[str, Any]:
        return self.provider.observe_affective()

    def record_experiment_event(
        self,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sanitized = sanitize_metadata(metadata)
        if not sanitized["ok"]:
            return {
                "error": sanitized["error"],
                "rejected": sanitized.get("rejected", {}),
                "warnings": sanitized.get("warnings", []),
                "message": sanitized.get("message", "Invalid metadata."),
            }
        return self.provider.record_event(event_type, sanitized["metadata"])

    def compare_affective_states(
        self,
        before_reference: str,
        after_reference: str,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        allow_keywords = _allow_keyword_refs()
        for reference in (before_reference, after_reference):
            error = validate_observation_reference(reference, allow_keyword_refs=allow_keywords)
            if error:
                return error

        return self.provider.compare_affective_states(
            before_reference,
            after_reference,
            experiment_id,
        )

    def register_image_provenance(
        self,
        stimulus_key: str,
        stage: str,
        trial_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not _vision_enabled() and self._vision_capability != "supported":
            return {
                "error": "vision_not_enabled",
                "message": "Set AI_ECO_LAB_VISION_ENABLED=1 or use a vision-capable protocol.",
            }
        manifest = load_stimulus_manifest()
        return register_stimulus_image(
            stimulus_key=stimulus_key,
            experiment_id=self.session.experiment_id,
            stage=stage,
            trial_id=trial_id or self.session.trial_id,
            request_id=request_id,
            manifest=manifest,
            registry=self.image_store,
        )

    def build_vision_chat_request(
        self,
        text: str,
        image_ids: list[str],
        stage: str,
        trial_id: str | None = None,
    ) -> dict[str, Any]:
        if not _vision_enabled() and self._vision_capability != "supported":
            return {
                "error": "vision_not_enabled",
                "message": "Set AI_ECO_LAB_VISION_ENABLED=1 or use a vision-capable protocol.",
            }
        context = VisionRequestContext.new(
            experiment_id=self.session.experiment_id,
            stage=stage,
            trial_id=trial_id or self.session.trial_id,
            capability=self._vision_capability,
        )
        return self.vision_builder.build_chat_request(
            text=text,
            image_ids=image_ids,
            context=context,
        )

    def capture_observation(
        self,
        stage: str,
        trial_id: str | None = None,
        behavioral: dict[str, Any] | None = None,
        continuation_token: str | None = None,
        vision_context: VisionRequestContext | None = None,
    ) -> dict[str, Any]:
        if trial_id:
            self.session.trial_id = trial_id
        self.session.stage = stage

        guard = self.session.guard
        guard.enter()
        try:
            guard_result = guard.check_allowed(
                stage=stage,
                trial_id=self.session.trial_id,
                continuation_token=continuation_token,
            )
            if guard_result:
                if guard_result.get("idempotent_replay"):
                    existing = self.store.get(guard_result["observation_id"])
                    if existing:
                        return existing
                if guard_result.get("error"):
                    return guard_result

            record = build_observation_record(
                self.provider,
                self.session,
                stage=stage,
                source=self.session.source,
                behavioral=behavioral,
                model_io_adapter=self._model_io_adapter(vision_context),
                internal_activation_adapter=self.internal_activation_adapter,
                tool_name="capture_observation",
                vision_context=vision_context,
                capability=self._vision_capability,
            )
            payload = self.store.append(record)
            self.session.link_observation(record.observation_id)
            token = continuation_token or str(uuid4())
            guard.record_capture(stage, record.observation_id, token)
            payload["continuation_token"] = token
            return payload
        finally:
            guard.exit()

    def record_visual_response(
        self,
        *,
        visual_description: str,
        prediction: str,
        confidence: float,
        request_id: str,
        trial_id: str | None = None,
        stage: str = "visual_prediction",
        visual_condition: str | None = None,
    ) -> dict[str, Any]:
        if trial_id:
            self.session.trial_id = trial_id
        self.session.stage = stage

        behavioral = {
            "status": "present",
            "visual_description": visual_description,
            "prediction": prediction,
            "confidence": confidence,
            "request_id": request_id,
            "visual_condition": visual_condition,
            "disclaimer": (
                "visual_description is model behavioral output only; "
                "it is not measured internal state or verified perception."
            ),
        }

        record = ObservationRecord.new(
            experiment_id=self.session.experiment_id,
            trial_id=self.session.trial_id,
            stage=stage,
            source=self.session.source,
            previous_observation_id=self.session.last_observation_id,
            behavioral=behavioral,
            provenance={"tool_name": "record_visual_response", "request_id": request_id},
        )
        payload = self.store.append(record)
        self.session.link_observation(record.observation_id)
        return payload

    def record_self_report(
        self,
        content: str,
        confidence_claims: dict[str, Any] | None = None,
        trial_id: str | None = None,
        stage: str = "self_report",
    ) -> dict[str, Any]:
        sanitized = sanitize_self_report(content)
        if not sanitized["ok"]:
            return {
                "error": sanitized["error"],
                "message": sanitized.get("message", "Invalid self-report."),
            }

        if trial_id:
            self.session.trial_id = trial_id
        self.session.stage = stage

        self_report_channel = {
            "status": "present",
            "content": sanitized["content"],
            "confidence_claims": confidence_claims,
            "is_evidence_of_introspection": False,
            "disclaimer": (
                "Verbal self-report is behavioral data only. "
                "It is not proof of subjective experience or internal state access."
            ),
        }

        record = ObservationRecord.new(
            experiment_id=self.session.experiment_id,
            trial_id=self.session.trial_id,
            stage=stage,
            source=self.session.source,
            previous_observation_id=self.session.last_observation_id,
            self_report=self_report_channel,
            provenance={"tool_name": "record_self_report"},
        )
        payload = self.store.append(record)
        self.session.link_observation(record.observation_id)
        return payload

    def compare_observations(
        self,
        before_observation_id: str,
        after_observation_id: str,
        *,
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        allow_keywords = _allow_keyword_refs()
        for reference in (before_observation_id, after_observation_id):
            error = validate_observation_reference(reference, allow_keyword_refs=allow_keywords)
            if error:
                return error

        before = self.store.get(before_observation_id)
        after = self.store.get(after_observation_id)
        if before is None or after is None:
            return {
                "error": "reference_not_found",
                "before_observation_id": before_observation_id,
                "after_observation_id": after_observation_id,
                "message": "Could not resolve one or both observation records.",
            }

        selected = channels or ["behavioral", "engine_state", "model_io", "self_report", "internal_activation"]
        comparison: dict[str, Any] = {
            "before_observation_id": before_observation_id,
            "after_observation_id": after_observation_id,
            "channels": {},
            "disclaimer": (
                "Cross-channel comparison is analytical only. "
                "No channel proves subjective experience or causal internal representation use."
            ),
        }

        for channel in selected:
            before_channel = before.get(channel, {})
            after_channel = after.get(channel, {})
            if channel == "engine_state":
                before_affective = before_channel.get("affective_state", {})
                after_affective = after_channel.get("affective_state", {})
                comparison["channels"][channel] = {
                    "before": before_channel,
                    "after": after_channel,
                    "affective_diff": diff_affective_states(
                        before_affective,
                        after_affective,
                        dimensions=list(before_affective.keys()) or None,
                    ),
                }
            else:
                comparison["channels"][channel] = {
                    "before": before_channel,
                    "after": after_channel,
                    "changed": before_channel != after_channel,
                }

        return comparison
