from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class PlaybackAdapter(ABC):
    @abstractmethod
    def play_wav(self, wav_bytes: bytes, *, sample_rate: int) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class NullPlaybackAdapter(PlaybackAdapter):
    def play_wav(self, wav_bytes: bytes, *, sample_rate: int) -> None:
        return None

    def stop(self) -> None:
        return None


class LocalPlaybackAdapter(PlaybackAdapter):
    def play_wav(self, wav_bytes: bytes, *, sample_rate: int) -> None:
        temp_dir = Path(tempfile.gettempdir())
        target = temp_dir / f"ai-eco-lab-playback-{uuid.uuid4()}.wav"
        try:
            target.write_bytes(wav_bytes)
            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(str(target), winsound.SND_FILENAME)
                return
            for command in (("afplay", str(target)), ("aplay", str(target))):
                binary = shutil.which(command[0])
                if binary:
                    subprocess.run([binary, command[1]], check=False)
                    return
            print(
                "[voice] Local playback unavailable; no supported player found.",
                file=sys.stderr,
            )
        finally:
            if target.exists():
                target.unlink(missing_ok=True)

    def stop(self) -> None:
        return None


def build_playback_adapter(*, enabled: bool) -> PlaybackAdapter:
    if enabled:
        return LocalPlaybackAdapter()
    return NullPlaybackAdapter()
