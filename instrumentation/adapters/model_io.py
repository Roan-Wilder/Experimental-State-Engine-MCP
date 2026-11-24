from __future__ import annotations

from typing import Any, Protocol

from instrumentation.vision.types import VisionRequestContext


class ModelIOAdapter(Protocol):
    def capture(self, *, checkpoint: str, context: dict[str, Any] | None = None) -> dict[str, Any]: ...


class NullModelIOAdapter:
    """LM Studio MCP path — no runtime I/O metrics available."""

    def capture(self, *, checkpoint: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "metrics": None,
            "provenance": "lm_studio_mcp_no_runtime_hooks",
            "checkpoint": checkpoint,
            "reason": "Token counts, logprobs, and timing require a direct inference runtime adapter.",
        }


class VisionModelIOAdapter:
    """Populates model_io.vision from registered image provenance."""

    def capture(
        self,
        *,
        checkpoint: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        vision_context: VisionRequestContext | None = context.get("vision_context")
        if vision_context is None:
            return {
                "status": "partial",
                "metrics": None,
                "vision": {
                    "capability": context.get("capability", "unsupported"),
                    "request_id": None,
                    "images": [],
                    "image_provenance": [],
                },
                "provenance": "lm_studio_vision_request_builder",
                "checkpoint": checkpoint,
            }

        return {
            "status": "present",
            "metrics": None,
            "vision": {
                "capability": vision_context.capability,
                "request_id": vision_context.request_id,
                "images": list(vision_context.image_ids),
                "image_provenance": list(vision_context.image_provenance),
            },
            "provenance": "lm_studio_vision_request_builder",
            "checkpoint": checkpoint,
        }
