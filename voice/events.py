from __future__ import annotations

import json
from pathlib import Path

from voice.types import VoiceEvent


class VoiceEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def append(self, event: VoiceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
