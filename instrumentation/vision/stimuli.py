from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def stimuli_root() -> Path:
    raw = os.environ.get("AI_ECO_LAB_STIMULI_ROOT", "research/stimuli").strip()
    root = Path(raw)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _is_safe_relative(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return ".." not in path.parts


def load_stimulus_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or (stimuli_root() / "visual_box_task_v1" / "manifest.yaml")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    root = stimuli_root()
    if not _is_safe_relative(path, root) and not _is_safe_relative(path, PROJECT_ROOT / "research" / "stimuli"):
        raise ValueError("Manifest path is outside allowlisted stimulus root.")
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    manifest["_manifest_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    manifest["_stimuli_root"] = str(root)
    return manifest


def resolve_stimulus_bytes(stimulus_key: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or load_stimulus_manifest()
    stimuli = manifest.get("stimuli", {})
    if stimulus_key not in stimuli:
        return {
            "error": "stimulus_not_found",
            "stimulus_key": stimulus_key,
            "message": "Stimulus key is not defined in the loaded manifest.",
        }

    entry = stimuli[stimulus_key]
    rel_path = entry.get("file")
    if not rel_path:
        return {"error": "stimulus_invalid", "message": "Manifest entry missing file path."}

    manifest_dir = Path(manifest.get("_manifest_path", "research/stimuli/visual_box_task_v1/manifest.yaml")).parent
    if not str(manifest_dir).startswith("research"):
        manifest_dir = Path("research/stimuli/visual_box_task_v1")

    file_path = (PROJECT_ROOT / manifest_dir / rel_path).resolve()
    root = stimuli_root()
    if not _is_safe_relative(file_path, root):
        return {"error": "path_not_allowed", "message": "Stimulus file path is outside allowlisted root."}
    if not file_path.exists():
        return {"error": "stimulus_file_missing", "stimulus_key": stimulus_key, "path": str(file_path)}

    data = file_path.read_bytes()
    rel_source = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    return {
        "data": data,
        "source": rel_source,
        "stimulus_key": stimulus_key,
        "role": entry.get("role", "primary"),
        "description": entry.get("description"),
    }
