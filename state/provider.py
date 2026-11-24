from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InternalStateProvider(ABC):
    """Abstract interface for internal-state observation backends.

    Current implementation:
    - ExperimentalStateProvider (Phase 2) — deterministic external state engine

    Future implementations (not yet built; no simulated neural data):
    - JacobianLensProvider (J-Space integration)
    - NeuralStateProvider / ActivationProvider / HiddenStateProvider
    """

    @abstractmethod
    def observe(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the current measured state vector."""

    @abstractmethod
    def history(self, limit: int | None = None) -> dict[str, Any]:
        """Return bounded recent state transition history."""

    @abstractmethod
    def record_event(self, event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply an event-driven state transition."""

    @abstractmethod
    def reflect_context(self) -> dict[str, Any]:
        """Return measured context for reflection without engine interpretation."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return provider metadata for experiment logging."""

    @abstractmethod
    def save_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Persist current state and history."""

    @abstractmethod
    def restore_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Restore state and history from a snapshot."""

    def observe_affective(self) -> dict[str, Any]:
        """Return current experimental affective observation vector."""
        raise NotImplementedError

    def compare_affective_states(
        self,
        before_reference: str,
        after_reference: str,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare two recorded affective states analytically."""
        raise NotImplementedError
