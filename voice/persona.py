from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PERSONA_DISCLAIMER = (
    "This voice persona is an experimental TTS output profile. "
    "It does not represent the identity, personality, consciousness, "
    "or emotional state of Gemma."
)

PERSONA_STATUSES = frozenset({"active", "superseded", "deprecated"})


@dataclass(frozen=True)
class VoicePersona:
    voice_id: str
    name: str
    description: str
    source_model: str
    model_hash: str
    codec_hash: str
    creation_seed: int | None
    creation_instruction: str
    reference_audio_hash: str
    reference_artifact_id: str
    reference_text_hash: str
    creation_timestamp: str
    version: int
    status: str
    persona_disclaimer: str = PERSONA_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "description": self.description,
            "source_model": self.source_model,
            "model_hash": self.model_hash,
            "codec_hash": self.codec_hash,
            "creation_seed": self.creation_seed,
            "creation_instruction": self.creation_instruction,
            "reference_audio_hash": self.reference_audio_hash,
            "reference_artifact_id": self.reference_artifact_id,
            "reference_text_hash": self.reference_text_hash,
            "creation_timestamp": self.creation_timestamp,
            "version": self.version,
            "status": self.status,
            "persona_disclaimer": self.persona_disclaimer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VoicePersona:
        return cls(
            voice_id=str(payload["voice_id"]),
            name=str(payload["name"]),
            description=str(payload["description"]),
            source_model=str(payload["source_model"]),
            model_hash=str(payload["model_hash"]),
            codec_hash=str(payload["codec_hash"]),
            creation_seed=payload.get("creation_seed"),
            creation_instruction=str(payload["creation_instruction"]),
            reference_audio_hash=str(payload["reference_audio_hash"]),
            reference_artifact_id=str(payload["reference_artifact_id"]),
            reference_text_hash=str(payload["reference_text_hash"]),
            creation_timestamp=str(payload["creation_timestamp"]),
            version=int(payload["version"]),
            status=str(payload["status"]),
            persona_disclaimer=str(payload.get("persona_disclaimer", PERSONA_DISCLAIMER)),
        )
