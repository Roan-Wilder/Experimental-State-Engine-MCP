from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from voice.acoustic import AcousticAnalyzerRegistry, analyze_acoustic_observation
from voice.artifacts import VoiceArtifactStore
from voice.config import VoiceConfig
from voice.events import VoiceEventStore
from voice.mock_adapter import MockVoiceAdapter
from voice.persona import PERSONA_DISCLAIMER, VoicePersona
from voice.persona_store import VoicePersonaStore
from voice.playback_adapter import build_playback_adapter
from voice.presets import resolve_preset
from voice.qwen_tts_adapter import QwenTTSAdapter
from voice.sanitize import sanitize_voice_instruction, sanitize_voice_text
from voice.types import VALID_OUTPUT_MODES, VOICE_DISCLAIMER, OutputMode, VoiceEvent, VoiceSynthesisRequest

REAL_BACKENDS = frozenset({"sidecar_http", "sidecar_rpc", "cli_oneshot"})


def runtime_provenance(config: VoiceConfig, *, status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Optional runtime metadata for voice events (no state-engine coupling)."""
    payload: dict[str, Any] = {
        "sidecar_url": config.sidecar_url,
        "ggml_backend": config.ggml_backend or os.environ.get("GGML_BACKEND"),
        "qwentts_commit": os.environ.get("QWEN_TTS_RUNTIME_COMMIT"),
        "binary_path": str(config.binary_path) if config.binary_path else None,
    }
    if status:
        payload["model_sha256"] = status.get("model_sha256")
        payload["codec_sha256"] = status.get("codec_sha256")
    return payload


class VoiceService:
    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = (config or VoiceConfig.from_env()).resolve_paths()
        self._adapter = self._build_adapter()
        self._artifact_store = VoiceArtifactStore(self.config.artifact_root)
        self._event_store = VoiceEventStore(self.config.events_path)
        self._persona_store = VoicePersonaStore(self.config.personas_path)
        self._acoustic_registry = AcousticAnalyzerRegistry()
        self._playback = build_playback_adapter(enabled=self.config.playback_enabled)

    def _build_adapter(self):
        if self.config.backend == "mock":
            return MockVoiceAdapter(self.config)
        return QwenTTSAdapter(self.config)

    def get_status(self) -> dict[str, Any]:
        try:
            if not self._adapter.get_status().initialized:
                self._adapter.initialize()
        except Exception as exc:
            status = self._adapter.get_status()
            status.last_error = str(exc)
            return status.to_dict()
        return self._adapter.get_status().to_dict()

    def list_voice_personas(self, *, status: str | None = None) -> dict[str, Any]:
        personas = self._persona_store.list(status=status)
        return {
            "ok": True,
            "personas": [persona.to_dict() for persona in personas],
            "count": len(personas),
            "persona_disclaimer": PERSONA_DISCLAIMER,
        }

    def get_voice_persona(self, voice_id: str, *, version: int | None = None) -> dict[str, Any]:
        persona = self._persona_store.get(voice_id, version=version)
        if persona is None:
            return {
                "ok": False,
                "error": "persona_not_found",
                "message": f"No voice persona found for voice_id={voice_id}.",
            }
        return {
            "ok": True,
            "persona": persona.to_dict(),
            "persona_disclaimer": PERSONA_DISCLAIMER,
        }

    def create_voice_persona(
        self,
        *,
        name: str,
        description: str,
        voice_instruction: str,
        seed: int | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        instruction_result = sanitize_voice_instruction(
            voice_instruction,
            max_length=self.config.max_text_length,
        )
        if not instruction_result.get("ok"):
            return instruction_result
        cleaned_instruction = str(instruction_result["instruction"])
        ref_text = reference_text or self.config.persona_reference_text
        text_result = sanitize_voice_text(ref_text, max_length=self.config.max_text_length)
        if not text_result.get("ok"):
            return text_result
        cleaned_reference_text = str(text_result["text"])

        synth = self.synthesize(
            text=cleaned_reference_text,
            voice_instruction=cleaned_instruction,
            seed=seed,
            save_artifact=False,
            voice_id=None,
        )
        if not synth.get("ok"):
            return synth

        wav_bytes = synth["wav_bytes"]
        voice_id = str(uuid.uuid4())
        reference_audio_hash = hashlib.sha256(wav_bytes).hexdigest()
        reference_text_hash = hashlib.sha256(cleaned_reference_text.encode("utf-8")).hexdigest()
        reference_artifact_id = self._artifact_store.write_persona_reference(
            voice_id,
            1,
            wav_bytes,
        )
        status = self._adapter.get_status()
        persona = VoicePersona(
            voice_id=voice_id,
            name=name.strip(),
            description=description.strip(),
            source_model=self.config.model_basename(),
            model_hash=status.model_sha256 or "",
            codec_hash=status.codec_sha256 or "",
            creation_seed=seed,
            creation_instruction=cleaned_instruction,
            reference_audio_hash=reference_audio_hash,
            reference_artifact_id=reference_artifact_id,
            reference_text_hash=reference_text_hash,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            version=1,
            status="active",
        )
        self._persona_store.append(persona)
        self._append_persona_event(persona, event_type="voice_persona_creation")
        return {
            "ok": True,
            "persona": persona.to_dict(),
            "persona_disclaimer": PERSONA_DISCLAIMER,
        }

    def regenerate_voice_persona(
        self,
        voice_id: str,
        *,
        voice_instruction: str | None = None,
        seed: int | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        current = self._persona_store.get(voice_id)
        if current is None:
            return {
                "ok": False,
                "error": "persona_not_found",
                "message": f"No voice persona found for voice_id={voice_id}.",
            }
        self._persona_store.supersede_versions(voice_id)
        instruction = voice_instruction or current.creation_instruction
        new_seed = seed if seed is not None else current.creation_seed
        return self._create_persona_version(
            voice_id=voice_id,
            name=current.name,
            description=current.description,
            voice_instruction=instruction,
            seed=new_seed,
            reference_text=reference_text,
            version=current.version + 1,
        )

    def _create_persona_version(
        self,
        *,
        voice_id: str,
        name: str,
        description: str,
        voice_instruction: str,
        seed: int | None,
        reference_text: str | None,
        version: int,
    ) -> dict[str, Any]:
        instruction_result = sanitize_voice_instruction(
            voice_instruction,
            max_length=self.config.max_text_length,
        )
        if not instruction_result.get("ok"):
            return instruction_result
        cleaned_instruction = str(instruction_result["instruction"])
        ref_text = reference_text or self.config.persona_reference_text
        text_result = sanitize_voice_text(ref_text, max_length=self.config.max_text_length)
        if not text_result.get("ok"):
            return text_result
        cleaned_reference_text = str(text_result["text"])

        synth = self.synthesize(
            text=cleaned_reference_text,
            voice_instruction=cleaned_instruction,
            seed=seed,
            save_artifact=False,
            voice_id=None,
        )
        if not synth.get("ok"):
            return synth

        wav_bytes = synth["wav_bytes"]
        reference_audio_hash = hashlib.sha256(wav_bytes).hexdigest()
        reference_text_hash = hashlib.sha256(cleaned_reference_text.encode("utf-8")).hexdigest()
        reference_artifact_id = self._artifact_store.write_persona_reference(
            voice_id,
            version,
            wav_bytes,
        )
        status = self._adapter.get_status()
        persona = VoicePersona(
            voice_id=voice_id,
            name=name,
            description=description,
            source_model=self.config.model_basename(),
            model_hash=status.model_sha256 or "",
            codec_hash=status.codec_sha256 or "",
            creation_seed=seed,
            creation_instruction=cleaned_instruction,
            reference_audio_hash=reference_audio_hash,
            reference_artifact_id=reference_artifact_id,
            reference_text_hash=reference_text_hash,
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            version=version,
            status="active",
        )
        self._persona_store.append(persona)
        self._append_persona_event(persona, event_type="voice_persona_regeneration")
        return {
            "ok": True,
            "persona": persona.to_dict(),
            "persona_disclaimer": PERSONA_DISCLAIMER,
        }

    def _append_persona_event(self, persona: VoicePersona, *, event_type: str) -> None:
        status = self._adapter.get_status()
        event = VoiceEvent(
            event_id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=os.environ.get("AI_ECO_LAB_EXPERIMENT_ID") or None,
            trial_id=os.environ.get("AI_ECO_LAB_TRIAL_ID") or None,
            stage=os.environ.get("AI_ECO_LAB_TRIAL_STAGE") or None,
            tts_model=persona.source_model,
            tts_model_sha256=status.model_sha256,
            tts_codec=self.config.codec_basename(),
            tts_codec_sha256=status.codec_sha256,
            tts_backend=self.config.backend,
            language=self.config.language,
            voice_instruction=persona.creation_instruction,
            voice_preset=None,
            text_hash=persona.reference_text_hash,
            audio_hash=persona.reference_audio_hash,
            sample_rate=24000,
            channels=1,
            duration_seconds=0.0,
            seed=persona.creation_seed,
            generation_time_seconds=0.0,
            artifact_id=persona.reference_artifact_id,
            acoustic_features={},
            acoustic_observation=None,
            disclaimer=PERSONA_DISCLAIMER,
            voice_id=persona.voice_id,
            persona_version=persona.version,
        )
        self._event_store.append(event)

    def synthesize(
        self,
        *,
        text: str,
        voice_instruction: str | None = None,
        voice_preset: str | None = None,
        voice_id: str | None = None,
        language: str | None = None,
        speed: float | None = None,
        seed: int | None = None,
        save_artifact: bool | None = None,
        output_mode: str | None = None,
        playback: bool | None = None,
    ) -> dict[str, Any]:
        persona_version: int | None = None
        instruction = voice_instruction
        if voice_id:
            persona = self._persona_store.get(voice_id)
            if persona is None:
                return {
                    "ok": False,
                    "error": "persona_not_found",
                    "message": f"No voice persona found for voice_id={voice_id}.",
                }
            instruction = persona.creation_instruction
            if seed is None:
                seed = persona.creation_seed
            persona_version = persona.version

        if voice_preset:
            preset_result = resolve_preset(voice_preset)
            if not preset_result.get("ok"):
                return preset_result
            instruction = str(preset_result["instruction"])
        if not instruction:
            instruction = self.config.default_instruction

        text_result = sanitize_voice_text(text, max_length=self.config.max_text_length)
        if not text_result.get("ok"):
            return text_result
        instruction_result = sanitize_voice_instruction(
            instruction,
            max_length=self.config.max_text_length,
        )
        if not instruction_result.get("ok"):
            return instruction_result

        cleaned_text = str(text_result["text"])
        cleaned_instruction = str(instruction_result["instruction"])
        request = VoiceSynthesisRequest(
            text=cleaned_text,
            voice_instruction=cleaned_instruction,
            language=language or self.config.language,
            voice_preset=voice_preset,
            voice_id=voice_id,
            speed=speed,
            seed=seed,
        )
        try:
            if not self._adapter.get_status().initialized:
                self._adapter.initialize()
            result = self._adapter.synthesize(request)
        except Exception as exc:
            status = self._adapter.get_status()
            status.last_error = str(exc)
            return {
                "ok": False,
                "error": "synthesis_failed",
                "message": str(exc),
                "status": status.to_dict(),
            }

        wav_bytes = result.wav_bytes
        text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        audio_hash = hashlib.sha256(wav_bytes).hexdigest()
        acoustic = self._acoustic_registry.run(wav_bytes)
        resolved_output_mode = output_mode or self.config.output_mode
        if resolved_output_mode not in VALID_OUTPUT_MODES:
            return {
                "ok": False,
                "error": "invalid_output_mode",
                "message": f"Output mode must be one of {sorted(VALID_OUTPUT_MODES)}.",
            }
        if resolved_output_mode == OutputMode.GENERATE_AND_PLAY.value:
            if playback is False or not self.config.playback_enabled:
                return {
                    "ok": False,
                    "error": "playback_disabled",
                    "message": "Local playback requires QWEN_TTS_PLAYBACK_ENABLED=1 and playback consent.",
                }
            self._playback.play_wav(wav_bytes, sample_rate=result.sample_rate)

        artifact_id = None
        should_save = self.config.persist_artifacts if save_artifact is None else save_artifact
        if should_save or resolved_output_mode in {
            OutputMode.GENERATE_AND_SAVE.value,
            OutputMode.GENERATE_AND_PLAY.value,
        }:
            artifact_id = self._artifact_store.write_wav(wav_bytes)

        status = self._adapter.get_status()
        event = VoiceEvent(
            event_id=str(uuid.uuid4()),
            type="voice_generation",
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=os.environ.get("AI_ECO_LAB_EXPERIMENT_ID") or None,
            trial_id=os.environ.get("AI_ECO_LAB_TRIAL_ID") or None,
            stage=os.environ.get("AI_ECO_LAB_TRIAL_STAGE") or None,
            tts_model=self.config.model_basename(),
            tts_model_sha256=status.model_sha256,
            tts_codec=self.config.codec_basename(),
            tts_codec_sha256=status.codec_sha256,
            tts_backend=result.backend,
            language=request.language,
            voice_instruction=cleaned_instruction,
            voice_preset=voice_preset,
            text_hash=text_hash,
            audio_hash=audio_hash,
            sample_rate=result.sample_rate,
            channels=result.channels,
            duration_seconds=result.duration_seconds,
            seed=seed,
            generation_time_seconds=result.generation_time_seconds,
            artifact_id=artifact_id,
            acoustic_features=acoustic["features"],
            acoustic_observation=acoustic["acoustic_observation"],
            disclaimer=VOICE_DISCLAIMER,
            voice_id=voice_id,
            persona_version=persona_version,
        )
        self._event_store.append(event)
        status_dict = status.to_dict()
        metadata = {
            "ok": True,
            "event_id": event.event_id,
            "artifact_id": artifact_id,
            "text_hash": text_hash,
            "audio_hash": audio_hash,
            "sample_rate": result.sample_rate,
            "channels": result.channels,
            "duration_seconds": result.duration_seconds,
            "generation_time_seconds": result.generation_time_seconds,
            "tts_backend": result.backend,
            "tts_model": event.tts_model,
            "tts_codec": event.tts_codec,
            "tts_model_sha256": status_dict.get("model_sha256"),
            "tts_codec_sha256": status_dict.get("codec_sha256"),
            "language": request.language,
            "voice_instruction": cleaned_instruction,
            "voice_preset": voice_preset,
            "voice_id": voice_id,
            "persona_version": persona_version,
            "seed": seed,
            "output_mode": resolved_output_mode,
            "acoustic_features": acoustic["features"],
            "acoustic_observation": acoustic["acoustic_observation"],
            "runtime_provenance": runtime_provenance(self.config, status=status_dict),
            "disclaimer": VOICE_DISCLAIMER,
            "persona_disclaimer": PERSONA_DISCLAIMER if voice_id else None,
            "scientific_note": (
                "Qwen3-TTS is an output modality. Generated speech is not evidence of "
                "consciousness, emotion, sentience, or internal affective state."
            ),
        }
        include_audio = resolved_output_mode in {
            OutputMode.GENERATE_AND_RETURN.value,
            OutputMode.GENERATE_AND_PLAY.value,
        }
        payload: dict[str, Any] = {
            "ok": True,
            "metadata": metadata,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "include_audio": include_audio,
        }
        if include_audio:
            payload["wav_bytes"] = wav_bytes
        return payload

    def format_mcp_response(self, result: dict[str, Any], audio_cls: Any) -> list | dict[str, Any]:
        if not result.get("ok"):
            return result
        if result.get("include_audio"):
            return [
                audio_cls(data=result["wav_bytes"], format="wav"),
                result["metadata_json"],
            ]
        return result["metadata"]

    def shutdown(self) -> None:
        self._playback.stop()
        self._adapter.shutdown()


_singleton: VoiceService | None = None


def get_voice_service() -> VoiceService:
    global _singleton
    if _singleton is None:
        config = VoiceConfig.from_env().resolve_paths()
        if not config.enabled:
            raise RuntimeError(
                "TTS is disabled (TTS_ENABLED=0). Set TTS_ENABLED=1 to enable voice output."
            )
        _singleton = VoiceService(config)
    return _singleton


def reset_voice_service_for_tests() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.shutdown()
    _singleton = None
