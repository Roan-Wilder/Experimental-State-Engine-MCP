from __future__ import annotations

from typing import Any

from instrumentation.vision.hashing import sha256_hex
from instrumentation.vision.metadata import extract_image_metadata
from instrumentation.vision.mime import validate_mime_type
from instrumentation.vision.registry import ImageProvenanceStore
from instrumentation.vision.stimuli import resolve_stimulus_bytes
from instrumentation.vision.types import ImageProvenanceRecord


def register_image_bytes(
    *,
    data: bytes,
    mime_type: str,
    source: str,
    experiment_id: str,
    stage: str,
    source_type: str = "inline",
    role: str = "primary",
    trial_id: str | None = None,
    request_id: str | None = None,
    registry: ImageProvenanceStore | None = None,
) -> dict[str, Any]:
    store = registry or ImageProvenanceStore()
    mime_error = validate_mime_type(mime_type, data)
    if mime_error:
        return mime_error

    meta = extract_image_metadata(data, mime_type)
    if meta.get("error"):
        return meta

    digest = sha256_hex(data)
    duplicates = store.find_by_hash(digest)
    record = ImageProvenanceRecord.new(
        sha256=digest,
        mime_type=mime_type,
        width=int(meta["width"]),
        height=int(meta["height"]),
        byte_size=int(meta["byte_size"]),
        source=source,
        source_type=source_type,
        role=role,
        trial_id=trial_id,
        experiment_id=experiment_id,
        stage=stage,
        request_id=request_id,
        duplicate_of_image_id=duplicates[0]["image_id"] if duplicates else None,
    )
    from instrumentation.vision.artifact_store import persist_artifacts_enabled

    payload = store.append(record, data=data if persist_artifacts_enabled() else None)
    return payload


def register_stimulus_image(
    *,
    stimulus_key: str,
    experiment_id: str,
    stage: str,
    trial_id: str | None = None,
    request_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    registry: ImageProvenanceStore | None = None,
) -> dict[str, Any]:
    resolved = resolve_stimulus_bytes(stimulus_key, manifest)
    if resolved.get("error"):
        return resolved

    data = resolved["data"]
    from instrumentation.vision.mime import sniff_mime_type

    mime_type = sniff_mime_type(data)
    if mime_type is None:
        return {"error": "unsupported_image_format", "message": "Stimulus file format is not supported."}

    return register_image_bytes(
        data=data,
        mime_type=mime_type,
        source=resolved["source"],
        experiment_id=experiment_id,
        stage=stage,
        source_type="stimulus_manifest",
        role=str(resolved.get("role", "primary")),
        trial_id=trial_id,
        request_id=request_id,
        registry=registry,
    )
