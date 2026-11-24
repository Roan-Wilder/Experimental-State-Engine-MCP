from __future__ import annotations

from collections import deque
from typing import Any


class StateHistory:
    def __init__(self, limit: int = 10) -> None:
        self._limit = max(1, limit)
        self._entries: deque[dict[str, Any]] = deque(maxlen=self._limit)

    @property
    def limit(self) -> int:
        return self._limit

    def append(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)

    def get(self, limit: int | None = None) -> list[dict[str, Any]]:
        requested = self._limit if limit is None else max(1, min(limit, self._limit))
        items = list(self._entries)
        return items[-requested:]

    def clear(self) -> None:
        self._entries.clear()

    def replace(self, entries: list[dict[str, Any]]) -> None:
        self._entries.clear()
        for entry in entries[-self._limit :]:
            self._entries.append(entry)

    def summary(self, limit: int = 3) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for entry in self.get(limit):
            compact.append(
                {
                    "observation_id": entry.get("observation_id"),
                    "timestamp": entry.get("timestamp"),
                    "event": entry.get("event"),
                    "state": entry.get("state"),
                    "delta": entry.get("delta"),
                }
            )
        return compact

    def stale_entry(self, steps_back: int = 1) -> dict[str, Any] | None:
        items = list(self._entries)
        if not items:
            return None
        index = max(0, len(items) - steps_back - 1)
        return items[index]
