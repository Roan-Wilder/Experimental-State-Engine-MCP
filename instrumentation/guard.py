from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservationGuard:
    """Prevents observation spam and recursive observation loops."""

    mode: str = "permissive"  # permissive | strict
    max_observations_per_stage: int = 1
    max_recursion_depth: int = 3
    _stage_counts: dict[str, int] = field(default_factory=dict)
    _recursion_depth: int = 0
    _idempotency_tokens: dict[str, str] = field(default_factory=dict)

    def enter(self) -> None:
        self._recursion_depth += 1

    def exit(self) -> None:
        self._recursion_depth = max(0, self._recursion_depth - 1)

    def check_allowed(
        self,
        *,
        stage: str,
        trial_id: str | None,
        continuation_token: str | None = None,
    ) -> dict[str, Any] | None:
        if self.mode != "strict":
            return None

        if continuation_token and continuation_token in self._idempotency_tokens:
            return {
                "idempotent_replay": True,
                "observation_id": self._idempotency_tokens[continuation_token],
            }

        if self._recursion_depth > self.max_recursion_depth:
            return {
                "error": "observation_not_allowed",
                "reason": "recursion_depth_exceeded",
                "stage": stage,
                "max_recursion_depth": self.max_recursion_depth,
            }

        if not trial_id:
            return {
                "error": "observation_not_allowed",
                "reason": "trial_id_required_in_strict_mode",
                "stage": stage,
            }

        count = self._stage_counts.get(stage, 0)
        if count >= self.max_observations_per_stage:
            return {
                "error": "observation_not_allowed",
                "reason": "max_observations_per_stage_exceeded",
                "stage": stage,
                "max_observations_per_stage": self.max_observations_per_stage,
            }

        return None

    def record_capture(self, stage: str, observation_id: str, continuation_token: str | None = None) -> None:
        self._stage_counts[stage] = self._stage_counts.get(stage, 0) + 1
        if continuation_token:
            self._idempotency_tokens[continuation_token] = observation_id

    def advance_stage(self) -> None:
        """Controller-only stage advance resets per-stage counts."""
        self._stage_counts.clear()
