"""Generate minimal synthetic PNG bytes for vision instrumentation tests."""

from __future__ import annotations

from instrumentation.vision.metadata import minimal_png_bytes


def write_stimulus_png(path: str, width: int = 8, height: int = 8, rgba: tuple[int, int, int, int] = (10, 20, 30, 255)) -> bytes:
    data = minimal_png_bytes(width=width, height=height, rgba=rgba)
    with open(path, "wb") as handle:
        handle.write(data)
    return data


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "research" / "stimuli" / "visual_box_task_v1"
    root.mkdir(parents=True, exist_ok=True)
    write_stimulus_png(str(root / "box_a_token.png"), rgba=(255, 0, 0, 255))
    write_stimulus_png(str(root / "box_b_token.png"), rgba=(0, 255, 0, 255))
    write_stimulus_png(str(root / "box_c_token.png"), rgba=(0, 0, 255, 255))
    write_stimulus_png(str(root / "irrelevant_scene.png"), rgba=(128, 128, 128, 255))
    print(f"Wrote stimulus PNGs to {root}")
