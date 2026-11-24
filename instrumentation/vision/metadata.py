from __future__ import annotations

import struct
import zlib


def extract_png_dimensions(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 24:
        return None
    # IHDR chunk starts at byte 8; width/height are big-endian uint32 at 16 and 20
    try:
        width, height = struct.unpack(">II", data[16:24])
    except struct.error:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def extract_image_metadata(data: bytes, mime_type: str) -> dict[str, int] | dict[str, str]:
    byte_size = len(data)
    if mime_type == "image/png":
        dims = extract_png_dimensions(data)
        if dims is None:
            return {"error": "metadata_extraction_failed", "message": "Could not parse PNG dimensions."}
        width, height = dims
        return {"width": width, "height": height, "byte_size": byte_size}
    return {
        "error": "metadata_extraction_unsupported",
        "message": f"Dimension extraction for '{mime_type}' requires optional Pillow; use PNG in CI.",
    }


def minimal_png_bytes(width: int = 1, height: int = 1, rgba: tuple[int, int, int, int] = (1, 2, 3, 255)) -> bytes:
    """Generate a minimal valid PNG for tests without external dependencies."""
    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b""
    row = bytes([0] + list(rgba) * width)
    for _ in range(height):
        raw += row
    idat = zlib.compress(raw, level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )
