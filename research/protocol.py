from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parent / "protocols" / "prediction_revision_v1.yaml"
)


def load_protocol(protocol_path: Path | None = None) -> dict[str, Any]:
    path = protocol_path or DEFAULT_PROTOCOL_PATH
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)

    extends = protocol.get("extends")
    if extends:
        base_filename = extends if str(extends).endswith(".yaml") else f"{extends}.yaml"
        base_path = path.parent / base_filename
        base_protocol = load_protocol(base_path)
        merged = dict(base_protocol)
        merged.update({k: v for k, v in protocol.items() if k != "extends"})
        return merged

    return protocol


def get_condition_events(protocol: dict[str, Any], condition: str) -> list[str]:
    conditions = protocol.get("conditions", {})
    cfg = conditions.get(condition, {})
    events: list[str] = []
    information = cfg.get("information_event")
    if information:
        events.append(str(information))
    resolution = cfg.get("resolution_event")
    if resolution:
        events.append(str(resolution))
    return events
