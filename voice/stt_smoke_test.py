from __future__ import annotations

import argparse
import sys

from voice.stt_config import STTConfig
from voice.voice_input_service import VoiceInputService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local STT smoke test (microphone -> faster-whisper).")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index.")
    parser.add_argument(
        "--echo-tts",
        action="store_true",
        help="Optionally synthesize the transcription via existing VoiceService.",
    )
    args = parser.parse_args(argv)

    config = STTConfig.from_env().resolve_paths()
    service = VoiceInputService(config)
    status = service.get_status()
    if not status.get("available"):
        print("STT unavailable:", status.get("validation_error") or status, file=sys.stderr)
        return 1

    devices = service.list_microphones()
    if not devices:
        print("No microphone devices detected.", file=sys.stderr)
        return 1

    print("Available microphones:")
    for entry in devices:
        print(f"  [{entry['index']}] {entry['name']}")

    device = args.device
    if device is None:
        device = devices[0]["index"]
    print(f"\nRecording {args.duration:.1f}s from device {device}. Please speak now...")
    result = service.record_and_transcribe(
        duration_seconds=args.duration,
        device_index=device,
        source="microphone",
    )
    if not result.get("ok"):
        print("Transcription failed:", result, file=sys.stderr)
        return 1

    transcription = result["transcription"]
    print("\nTRANSCRIPTION:")
    print(transcription["text"])
    print("\nBACKEND:")
    print(transcription["backend"])
    print("\nMODEL:")
    print(transcription["model"])
    print("\nDEVICE:")
    print(transcription["device"])
    print("\nLATENCY:")
    print(f"{transcription['processing_time_seconds']:.3f}s")

    if args.echo_tts:
        from voice.service import VoiceService

        voice = VoiceService()
        spoken = voice.synthesize(text=transcription["text"])
        if not spoken.get("ok"):
            print("TTS echo failed:", spoken, file=sys.stderr)
            return 1
        print("\nTTS echo synthesis completed.")

    service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
