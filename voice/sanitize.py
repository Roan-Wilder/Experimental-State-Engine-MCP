from __future__ import annotations

import re

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SHELL_METACHAR_PATTERN = re.compile(r"[;&|`$<>]")
PATH_TRAVERSAL_PATTERN = re.compile(r"(?:\.\./|\.\.\\)")


def sanitize_voice_text(text: str, *, max_length: int) -> dict[str, str | bool]:
    if not isinstance(text, str):
        return {
            "ok": False,
            "error": "invalid_text",
            "message": "Text must be a string.",
        }
    cleaned = CONTROL_CHAR_PATTERN.sub("", text).strip()
    if not cleaned:
        return {
            "ok": False,
            "error": "empty_text",
            "message": "Text must be non-empty after sanitization.",
        }
    if len(cleaned) > max_length:
        return {
            "ok": False,
            "error": "text_too_long",
            "message": f"Text exceeds max length {max_length}.",
        }
    if SHELL_METACHAR_PATTERN.search(cleaned):
        return {
            "ok": False,
            "error": "unsafe_text",
            "message": "Text contains rejected shell metacharacters.",
        }
    if PATH_TRAVERSAL_PATTERN.search(cleaned):
        return {
            "ok": False,
            "error": "unsafe_text",
            "message": "Text contains rejected path traversal patterns.",
        }
    return {"ok": True, "text": cleaned}


def sanitize_voice_instruction(instruction: str, *, max_length: int) -> dict[str, str | bool]:
    if not isinstance(instruction, str):
        return {
            "ok": False,
            "error": "invalid_instruction",
            "message": "Voice instruction must be a string.",
        }
    cleaned = CONTROL_CHAR_PATTERN.sub("", instruction).strip()
    if not cleaned:
        return {
            "ok": False,
            "error": "empty_instruction",
            "message": "Voice instruction must be non-empty after sanitization.",
        }
    if len(cleaned) > max_length:
        return {
            "ok": False,
            "error": "instruction_too_long",
            "message": f"Voice instruction exceeds max length {max_length}.",
        }
    if SHELL_METACHAR_PATTERN.search(cleaned):
        return {
            "ok": False,
            "error": "unsafe_instruction",
            "message": "Voice instruction contains rejected shell metacharacters.",
        }
    return {"ok": True, "instruction": cleaned}
