from __future__ import annotations

import argparse
import sys

from voice.stt_config import STTConfig
from voice.voice_input_service import VoiceInputService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explicit voice loop demo: microphone -> STT -> optional existing TTS."
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--device", type=int, default=None, help="Microphone device index.")
    parser.add_argument(
        "--speak",
        action="store_true",
        help="After transcription, call existing VoiceService.synthesize on the text.",
    )
    args = parser.parse_args(argv)

    config = STTConfig.from_env().resolve_paths()
    input_service = VoiceInputService(config)
    print(f"Recording {args.duration:.1f}s. Speak now...")
    result = input_service.record_and_transcribe(
        duration_seconds=args.duration,
        device_index=args.device,
        source="microphone",
    )
    if not result.get("ok"):
        print("Voice input failed:", result, file=sys.stderr)
        return 1

    text = result["transcription"]["text"]
    print("\nTranscription:")
    print(text)

    if args.speak:
        from voice.service import VoiceService

        voice_service = VoiceService()
        spoken = voice_service.synthesize(text=text)
        if not spoken.get("ok"):
            print("speak_text path failed:", spoken, file=sys.stderr)
            return 1
        print("\nExisting TTS path completed (VoiceService.synthesize).")

    input_service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
