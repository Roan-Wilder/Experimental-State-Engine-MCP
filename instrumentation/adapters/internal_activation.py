from __future__ import annotations

from typing import Any, Protocol


class InternalActivationAdapter(Protocol):
    def capture_checkpoint(
        self,
        *,
        checkpoint: str,
        layer_ids: list[str] | None = None,
    ) -> dict[str, Any]: ...


class NullInternalActivationAdapter:
    """Placeholder until direct Gemma / Transformers runtime is connected."""

    def capture_checkpoint(
        self,
        *,
        checkpoint: str,
        layer_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "adapter": "null",
            "checkpoints": None,
            "checkpoint": checkpoint,
            "layer_ids": layer_ids,
            "reason": "Hidden-state capture requires Phase 2 direct model runtime instrumentation.",
        }
