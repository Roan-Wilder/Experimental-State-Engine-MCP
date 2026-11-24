from __future__ import annotations

import uuid

import pytest

from instrumentation.sanitize import (
    sanitize_metadata,
    sanitize_self_report,
    validate_observation_reference,
)


def test_sanitize_metadata_accepts_allowed_keys():
    result = sanitize_metadata(
        {
            "trial_id": str(uuid.uuid4()),
            "trial_phase": "prediction",
            "confidence": 0.75,
            "controller_defined": True,
            "notes": "pilot trial",
        }
    )
    assert result["ok"] is True
    assert result["metadata"]["confidence"] == 0.75


def test_sanitize_metadata_rejects_unknown_keys():
    result = sanitize_metadata({"trial_id": str(uuid.uuid4()), "model_response": "hello"})
    assert result["ok"] is False
    assert result["error"] == "invalid_metadata"
    assert "model_response" in result["rejected"]


def test_sanitize_metadata_rejects_tool_call_markup():
    result = sanitize_metadata({"notes": '<tool_call>{"name": "observe"}</tool_call>'})
    assert result["ok"] is False
    assert "notes" in result["rejected"]


def test_sanitize_metadata_rejects_oversized_notes():
    result = sanitize_metadata({"notes": "x" * 501})
    assert result["ok"] is False
    assert "notes" in result["rejected"]


def test_sanitize_metadata_empty_is_ok():
    result = sanitize_metadata(None)
    assert result["ok"] is True
    assert result["metadata"] == {}


def test_sanitize_self_report_accepts_plain_text():
    result = sanitize_self_report("I notice increased uncertainty in my reasoning.")
    assert result["ok"] is True
    assert "uncertainty" in result["content"]


def test_sanitize_self_report_rejects_tool_markup():
    result = sanitize_self_report("function_call arguments: {}")
    assert result["ok"] is False
    assert result["error"] == "invalid_self_report"


def test_validate_observation_reference_accepts_uuid():
    ref = str(uuid.uuid4())
    assert validate_observation_reference(ref) is None


def test_validate_observation_reference_rejects_placeholder():
    error = validate_observation_reference("previous")
    assert error is not None
    assert error["error"] == "invalid_reference"


def test_validate_observation_reference_allows_legacy_keywords_when_enabled():
    assert validate_observation_reference("current", allow_keyword_refs=True) is None
