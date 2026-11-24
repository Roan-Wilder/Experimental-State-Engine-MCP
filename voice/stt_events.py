from __future__ import annotations

import json
from pathlib import Path

from voice.stt_types import STTEvent


class STTEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def append(self, event: STTEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
