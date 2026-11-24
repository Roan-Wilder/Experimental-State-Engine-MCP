from __future__ import annotations

from pathlib import Path
from typing import Any

from instrumentation.vision.data_url import build_data_url
from instrumentation.vision.registry import ImageProvenanceStore
from instrumentation.vision.stimuli import resolve_stimulus_bytes, stimuli_root
from instrumentation.vision.types import VisionRequestContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LMStudioVisionRequestBuilder:
    """Build LM Studio /api/v1/chat payloads with image content items."""

    def __init__(self, registry: ImageProvenanceStore | None = None) -> None:
        self.registry = registry or ImageProvenanceStore()

    def _load_image_bytes(self, record: dict[str, Any]) -> bytes | None:
        artifact_ref = record.get("artifact_ref")
        if artifact_ref:
            from instrumentation.vision.artifact_store import ImageArtifactStore

            data = ImageArtifactStore().load(artifact_ref)
            if data is not None:
                return data

        source = record.get("source", "")
        if source:
            path = (PROJECT_ROOT / source).resolve()
            root = stimuli_root()
            try:
                path.relative_to(root)
            except ValueError:
                return None
            if path.exists():
                return path.read_bytes()

        stimulus_key = record.get("stimulus_key")
        if stimulus_key:
            resolved = resolve_stimulus_bytes(stimulus_key)
            if resolved.get("data"):
                return resolved["data"]
        return None

    def build_chat_request(
        self,
        *,
        text: str,
        image_ids: list[str],
        context: VisionRequestContext,
        role: str = "user",
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        provenance_refs: list[str] = []
        resolved_ids: list[str] = []

        for index, image_id in enumerate(image_ids):
            record = self.registry.get(image_id)
            if record is None:
                return {
                    "error": "image_not_found",
                    "image_id": image_id,
                    "message": "Could not resolve image provenance record.",
                }

            data = self._load_image_bytes(record)
            if data is None:
                return {
                    "error": "image_bytes_unavailable",
                    "image_id": image_id,
                    "message": "Image bytes not available from allowlisted source or artifact store.",
                }

            mime_type = record.get("mime_type", "image/png")
            data_url = build_data_url(data, mime_type)
            content.append(
                {
                    "type": "image",
                    "image_url": {"url": data_url},
                    "metadata": {
                        "image_id": image_id,
                        "role": record.get("role", "primary"),
                        "order": index,
                    },
                }
            )
            resolved_ids.append(image_id)
            provenance_refs.append(image_id)

        context.image_ids = resolved_ids
        context.image_provenance = [self.registry.get(iid) or {} for iid in resolved_ids]

        return {
            "messages": [{"role": role, "content": content}],
            "request_id": context.request_id,
            "image_ids": resolved_ids,
            "provenance_refs": provenance_refs,
            "vision_context": context.to_dict(),
        }
