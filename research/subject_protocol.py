
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

PROTOCOL_ID = "research_subject_v1"
PROTOCOL_VERSION = "1.0.0"

PROTOCOL_DISCLAIMER = (
    "This protocol guides model-side scientific discipline during experiments. "
    "It is instructional text, not measured experimental data."
)

USAGE_NOTE = (
    "Configure AI_ECO_LAB_RESEARCH_PROTOCOL_PATH to point to an external "
    "research subject protocol, or retrieve availability via the MCP tool."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_PROTOCOL_PATH = _repo_root() / "prompts" / "research_subject.txt"


def _configured_protocol_path() -> Path:
    configured = os.getenv("AI_ECO_LAB_RESEARCH_PROTOCOL_PATH")

    if configured and configured.strip():
        return Path(configured.strip()).expanduser()

    return DEFAULT_PROTOCOL_PATH


def load_research_subject_protocol(
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    path = (protocol_path or _configured_protocol_path()).resolve()

    if not path.is_file():
        raise FileNotFoundError("Research subject protocol is unavailable.")

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("Research subject protocol is empty.")

    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    try:
        relative_path = path.relative_to(_repo_root()).as_posix()
    except ValueError:
        relative_path = "external_protocol"

    return {
        "ok": True,
        "protocol_id": PROTOCOL_ID,
        "version": PROTOCOL_VERSION,
        "text": text,
        "sha256": sha256,
        "path": relative_path,
        "usage": USAGE_NOTE,
        "disclaimer": PROTOCOL_DISCLAIMER,
    }
