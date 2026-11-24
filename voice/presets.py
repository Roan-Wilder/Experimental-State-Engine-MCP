from __future__ import annotations

from pathlib import Path

import yaml

_PRESETS_PATH = Path(__file__).resolve().parent / "presets.yaml"


def load_presets(path: Path | None = None) -> dict[str, str]:
    preset_path = path or _PRESETS_PATH
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entries = payload.get("voice_presets", {})
    resolved: dict[str, str] = {}
    for name, value in entries.items():
        if isinstance(value, dict) and "instruction" in value:
            resolved[str(name)] = str(value["instruction"])
    return resolved


def resolve_preset(name: str, *, presets: dict[str, str] | None = None) -> dict[str, str | bool]:
    catalog = presets if presets is not None else load_presets()
    if name not in catalog:
        return {
            "ok": False,
            "error": "unknown_preset",
            "message": f"Unknown voice preset: {name}",
        }
    return {"ok": True, "instruction": catalog[name]}
