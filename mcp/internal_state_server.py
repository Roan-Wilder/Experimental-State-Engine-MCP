#!/usr/bin/env python3
"""AI Eco Lab internal-state MCP server for LM Studio."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_sdk_mcp_importable() -> None:
    """Prevent this repo's `mcp/` folder from shadowing the installed MCP SDK."""
    project_root = str(PROJECT_ROOT.resolve())
    script_dir = str(SCRIPT_DIR.resolve())
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry).resolve() != Path(project_root).resolve()
        and Path(entry).resolve() != Path(script_dir).resolve()
    ]


_ensure_sdk_mcp_importable()
from mcp.server.fastmcp import FastMCP  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instrumentation.facade import InstrumentedStateFacade  # noqa: E402

mcp = FastMCP("ai-eco-lab-state")
_facade = InstrumentedStateFacade()


def _vision_tools_enabled() -> bool:
    return os.environ.get("AI_ECO_LAB_VISION_ENABLED", "0").strip() == "1"


def _tts_tools_enabled() -> bool:
    from voice.config import tts_enabled_from_env

    return tts_enabled_from_env()


def _stt_tools_enabled() -> bool:
    return os.environ.get("STT_ENABLED", "0").strip() == "1"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@mcp.tool()
def observe_internal_state() -> dict[str, Any]:
    """Inspect the current measured experimental internal-state vector."""
    return _facade.observe_internal_state()


@mcp.tool()
def observe_state_history(limit: int = 10) -> dict[str, Any]:
    """Inspect a compact, bounded history of recent state observations."""
    return _facade.observe_state_history(limit=limit)


@mcp.tool()
def reflect_on_internal_state() -> dict[str, Any]:
    """Return measured state context for model-side reflection without engine interpretation."""
    return _facade.reflect_on_internal_state()


@mcp.tool()
def record_experiment_event(event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record an experiment event that may cause a deterministic state transition."""
    return _facade.record_experiment_event(event_type, metadata)


@mcp.tool()
def get_affective_state() -> dict[str, Any]:
    """Return the current experimental affective observation vector (not proof of subjective emotion)."""
    return _facade.get_affective_state()


@mcp.tool()
def compare_affective_states(
    before_reference: str,
    after_reference: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Compare two recorded affective states by transition_id or observation UUID."""
    return _facade.compare_affective_states(before_reference, after_reference, experiment_id)


@mcp.tool()
def capture_observation(
    stage: str,
    trial_id: str | None = None,
    behavioral: dict[str, Any] | None = None,
    continuation_token: str | None = None,
) -> dict[str, Any]:
    """Capture a read-only multi-channel observation (behavioral, engine_state, model_io, self_report, internal_activation)."""
    return _facade.capture_observation(
        stage=stage,
        trial_id=trial_id,
        behavioral=behavioral,
        continuation_token=continuation_token,
    )


@mcp.tool()
def record_self_report(
    content: str,
    confidence_claims: dict[str, Any] | None = None,
    trial_id: str | None = None,
    stage: str = "self_report",
) -> dict[str, Any]:
    """Record model verbal self-report in a separate channel without mutating engine state."""
    return _facade.record_self_report(
        content=content,
        confidence_claims=confidence_claims,
        trial_id=trial_id,
        stage=stage,
    )


@mcp.tool()
def compare_observations(
    before_observation_id: str,
    after_observation_id: str,
    channels: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two observation records by UUID across selected instrumentation channels."""
    return _facade.compare_observations(
        before_observation_id,
        after_observation_id,
        channels=channels,
    )

@mcp.tool()
def get_research_subject_protocol() -> dict[str, Any]:
    """Return the optional externally configured research subject protocol."""
    from research.subject_protocol import load_research_subject_protocol

    try:
        return load_research_subject_protocol()
    except FileNotFoundError:
        return {
            "ok": False,
            "error": "research_subject_protocol_unavailable",
            "message": (
                "No research subject protocol is configured. "
                "Set AI_ECO_LAB_RESEARCH_PROTOCOL_PATH to an external file."
            ),
        }
    except (OSError, UnicodeError, ValueError):
        return {
            "ok": False,
            "error": "research_subject_protocol_unavailable",
            "message": "The configured research subject protocol could not be loaded.",
        }


if _tts_tools_enabled():
    from mcp.server.fastmcp.utilities.types import Audio  # noqa: E402

    from voice.service import get_voice_service  # noqa: E402

    _voice_service = get_voice_service()

    @mcp.tool()
    def speak_text(
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
    ) -> list | dict[str, Any]:
        """Synthesize speech via local Qwen3-TTS VoiceDesign without mutating experimental state."""
        try:
            result = _voice_service.synthesize(
                text=text,
                voice_instruction=voice_instruction,
                voice_preset=voice_preset,
                voice_id=voice_id,
                language=language,
                speed=speed,
                seed=seed,
                save_artifact=save_artifact,
                output_mode=output_mode,
                playback=playback,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "speak_text_failed",
                "message": str(exc),
            }
        return _voice_service.format_mcp_response(result, Audio)

    @mcp.tool()
    def get_voice_status() -> dict[str, Any]:
        """Return local Qwen3-TTS adapter status (models, backend, sidecar health)."""
        try:
            return _voice_service.get_status()
        except Exception as exc:
            return {
                "initialized": False,
                "available": False,
                "last_error": str(exc),
            }

    @mcp.tool()
    def create_voice_persona(
        name: str,
        description: str,
        voice_instruction: str,
        seed: int | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        """Create a reusable experimental voice persona from a VoiceDesign instruction."""
        try:
            return _voice_service.create_voice_persona(
                name=name,
                description=description,
                voice_instruction=voice_instruction,
                seed=seed,
                reference_text=reference_text,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "create_voice_persona_failed",
                "message": str(exc),
            }

    @mcp.tool()
    def get_voice_persona(voice_id: str, version: int | None = None) -> dict[str, Any]:
        """Return a stored voice persona by voice_id and optional version."""
        try:
            return _voice_service.get_voice_persona(voice_id, version=version)
        except Exception as exc:
            return {
                "ok": False,
                "error": "get_voice_persona_failed",
                "message": str(exc),
            }

    @mcp.tool()
    def list_voice_personas(status: str | None = None) -> dict[str, Any]:
        """List stored voice personas, optionally filtered by status."""
        try:
            return _voice_service.list_voice_personas(status=status)
        except Exception as exc:
            return {
                "ok": False,
                "error": "list_voice_personas_failed",
                "message": str(exc),
            }


if _stt_tools_enabled():
    from voice.voice_input_service import get_voice_input_service  # noqa: E402

    _voice_input_service = get_voice_input_service()

    @mcp.tool()
    def get_stt_status() -> dict[str, Any]:
        """Return local STT adapter and microphone capture status."""
        try:
            return _voice_input_service.get_status()
        except Exception as exc:
            return {
                "ok": False,
                "error": "get_stt_status_failed",
                "message": str(exc),
            }

    @mcp.tool()
    def get_microphone_devices() -> dict[str, Any]:
        """Enumerate local microphone input devices (explicit capture only)."""
        try:
            devices = _voice_input_service.list_microphones()
            return {
                "ok": True,
                "devices": devices,
                "count": len(devices),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": "get_microphone_devices_failed",
                "message": str(exc),
            }

    @mcp.tool()
    def record_voice_input(
        duration_seconds: float,
        device_index: int | None = None,
    ) -> dict[str, Any]:
        """Record bounded microphone audio and transcribe locally (no cloud upload)."""
        try:
            return _voice_input_service.record_and_transcribe(
                duration_seconds=duration_seconds,
                device_index=device_index,
                source="microphone",
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "record_voice_input_failed",
                "message": str(exc),
            }

    @mcp.tool()
    def transcribe_audio(audio_base64: str) -> dict[str, Any]:
        """Transcribe explicit WAV bytes (base64) without opening the microphone."""
        import base64

        try:
            wav_bytes = base64.b64decode(audio_base64.encode("ascii"))
            return _voice_input_service.transcribe_bytes(wav_bytes, source="file_bytes")
        except Exception as exc:
            return {
                "ok": False,
                "error": "transcribe_audio_failed",
                "message": str(exc),
            }


if _vision_tools_enabled():

    @mcp.tool()
    def register_image_provenance(
        stimulus_key: str,
        stage: str,
        trial_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Register image provenance from allowlisted stimulus manifest key (metadata only by default)."""
        return _facade.register_image_provenance(
            stimulus_key=stimulus_key,
            stage=stage,
            trial_id=trial_id,
            request_id=request_id,
        )

    @mcp.tool()
    def build_vision_chat_request(
        text: str,
        image_ids: list[str],
        stage: str,
        trial_id: str | None = None,
    ) -> dict[str, Any]:
        """Build read-only LM Studio vision chat payload with data_url image items."""
        return _facade.build_vision_chat_request(
            text=text,
            image_ids=image_ids,
            stage=stage,
            trial_id=trial_id,
        )


def main() -> None:
    _log("AI Eco Lab internal-state MCP server starting (stdio)")
    _log(f"Provider metadata: {_facade.provider.metadata()}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
