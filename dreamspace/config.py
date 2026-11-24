from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "True", "yes", "YES"}


def dream_space_enabled_from_env() -> bool:
    return _env_bool("DREAM_SPACE_ENABLED", False)


@dataclass(frozen=True)
class DreamSpaceConfig:
    enabled: bool
    worlds_enabled: bool
    belief_graph_enabled: bool
    counterfactuals_enabled: bool
    goals_enabled: bool
    events_path: Path

    @classmethod
    def from_env(cls, *, repo_root: Path | None = None) -> DreamSpaceConfig:
        root = repo_root or _repo_root()
        events = Path(os.environ.get("DREAM_SPACE_EVENTS_PATH", "research/raw/dream_events.jsonl"))
        if not events.is_absolute():
            events = (root / events).resolve()
        else:
            events = events.resolve()
        return cls(
            enabled=dream_space_enabled_from_env(),
            worlds_enabled=_env_bool("DREAM_SPACE_WORLDS_ENABLED", True),
            belief_graph_enabled=_env_bool("DREAM_SPACE_BELIEF_GRAPH_ENABLED", True),
            counterfactuals_enabled=_env_bool("DREAM_SPACE_COUNTERFACTUALS_ENABLED", True),
            goals_enabled=_env_bool("DREAM_SPACE_GOALS_ENABLED", True),
            events_path=events,
        )
