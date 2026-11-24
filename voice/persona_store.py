from __future__ import annotations

import json
from pathlib import Path

from voice.persona import VoicePersona


class VoicePersonaStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def append(self, persona: VoicePersona) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(persona.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> list[VoicePersona]:
        if not self.path.is_file():
            return []
        personas: list[VoicePersona] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            personas.append(VoicePersona.from_dict(json.loads(line)))
        return personas

    def get(self, voice_id: str, *, version: int | None = None) -> VoicePersona | None:
        matches = [p for p in self.load_all() if p.voice_id == voice_id]
        if not matches:
            return None
        if version is not None:
            for persona in matches:
                if persona.version == version:
                    return persona
            return None
        active = [p for p in matches if p.status == "active"]
        if active:
            return max(active, key=lambda p: p.version)
        return max(matches, key=lambda p: p.version)

    def list(self, *, status: str | None = None) -> list[VoicePersona]:
        personas = self.load_all()
        if status:
            personas = [p for p in personas if p.status == status]
        return sorted(personas, key=lambda p: (p.name.lower(), p.voice_id, p.version))

    def supersede_versions(self, voice_id: str) -> list[VoicePersona]:
        updated: list[VoicePersona] = []
        for persona in self.load_all():
            if persona.voice_id == voice_id and persona.status == "active":
                updated.append(
                    VoicePersona(
                        **{
                            **persona.to_dict(),
                            "status": "superseded",
                        }
                    )
                )
            else:
                updated.append(persona)
        self._rewrite(updated)
        return updated

    def _rewrite(self, personas: list[VoicePersona]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for persona in personas:
                handle.write(json.dumps(persona.to_dict(), ensure_ascii=False) + "\n")
