from __future__ import annotations

import re
from typing import Any

UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

ALLOWED_METADATA_KEYS = frozenset(
    {
        "trial_id",
        "trial_phase",
        "trial_condition",
        "stage",
        "prediction",
        "confidence",
        "reasoning_summary",
        "controller_defined",
        "revision_occurred",
        "notes",
    }
)

METADATA_STRING_LIMITS: dict[str, int] = {
    "notes": 500,
    "default": 2000,
}

REJECTED_CONTENT_PATTERNS = (
    re.compile(r"tool_call", re.IGNORECASE),
    re.compile(r"<\/?[\w-]+>"),
    re.compile(r"function_call", re.IGNORECASE),
    re.compile(r'"arguments"\s*:', re.IGNORECASE),
)

LEGACY_KEYWORD_REFERENCES = frozenset({"initial", "current", "previous", "latest"})


def is_uuid_v4(value: str) -> bool:
    return bool(UUID_V4_PATTERN.match(value.strip()))


def validate_observation_reference(
    reference: str,
    *,
    allow_keyword_refs: bool = False,
) -> dict[str, Any] | None:
    """Return an error dict if reference is invalid; otherwise None."""
    ref = reference.strip()
    if not ref:
        return {
            "error": "invalid_reference",
            "reference": reference,
            "message": "Observation reference must be a non-empty string.",
        }
    if is_uuid_v4(ref):
        return None
    if allow_keyword_refs and ref.lower() in LEGACY_KEYWORD_REFERENCES:
        return None
    return {
        "error": "invalid_reference",
        "reference": reference,
        "message": "Observation reference must be a valid UUID v4.",
    }


def _contains_rejected_content(value: str) -> bool:
    return any(pattern.search(value) for pattern in REJECTED_CONTENT_PATTERNS)


def _sanitize_scalar(key: str, value: Any) -> tuple[Any | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, None
    if isinstance(value, (int, float)):
        return value, None
    if isinstance(value, str):
        limit = METADATA_STRING_LIMITS.get(key, METADATA_STRING_LIMITS["default"])
        if len(value) > limit:
            return None, f"string value for '{key}' exceeds max length {limit}"
        if _contains_rejected_content(value):
            return None, f"string value for '{key}' contains rejected protocol content"
        return value, None
    return None, f"unsupported value type for '{key}': {type(value).__name__}"


def sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Validate and sanitize experiment event metadata.

    Returns:
        {
            "ok": bool,
            "metadata": dict,          # present when ok
            "rejected": dict,          # present when not ok
            "warnings": list[str],     # stripped unknown keys
            "error": str,              # present when not ok
            "message": str,            # present when not ok
        }
    """
    if metadata is None:
        return {"ok": True, "metadata": {}, "warnings": []}

    if not isinstance(metadata, dict):
        return {
            "ok": False,
            "error": "invalid_metadata",
            "rejected": {"_root": metadata},
            "message": "Metadata must be a JSON object.",
        }

    sanitized: dict[str, Any] = {}
    rejected: dict[str, Any] = {}
    warnings: list[str] = []

    for key, value in metadata.items():
        if key not in ALLOWED_METADATA_KEYS:
            rejected[key] = value
            warnings.append(f"unknown metadata key '{key}' rejected")
            continue

        clean_value, reason = _sanitize_scalar(key, value)
        if reason:
            rejected[key] = value
            warnings.append(reason)
            continue
        if clean_value is not None:
            sanitized[key] = clean_value

    if rejected:
        return {
            "ok": False,
            "error": "invalid_metadata",
            "rejected": rejected,
            "warnings": warnings,
            "message": "One or more metadata fields were rejected.",
        }

    return {"ok": True, "metadata": sanitized, "warnings": warnings}


def sanitize_self_report(content: str) -> dict[str, Any]:
    """Sanitize model self-report text before storage in the self_report channel."""
    if not isinstance(content, str):
        return {
            "ok": False,
            "error": "invalid_self_report",
            "message": "Self-report content must be a string.",
        }

    limit = METADATA_STRING_LIMITS["default"]
    if len(content) > limit:
        return {
            "ok": False,
            "error": "invalid_self_report",
            "message": f"Self-report content exceeds max length {limit}.",
        }
    if _contains_rejected_content(content):
        return {
            "ok": False,
            "error": "invalid_self_report",
            "message": "Self-report content contains rejected protocol artifacts.",
        }

    return {"ok": True, "content": content}
