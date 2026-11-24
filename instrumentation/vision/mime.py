from __future__ import annotations

ALLOWED_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
    }
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
WEBP_SIGNATURE = b"RIFF"
WEBP_MARKER = b"WEBP"


def sniff_mime_type(data: bytes) -> str | None:
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(JPEG_SIGNATURE):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == WEBP_SIGNATURE and data[8:12] == WEBP_MARKER:
        return "image/webp"
    return None


def validate_mime_type(mime_type: str, data: bytes) -> dict[str, str] | None:
    if mime_type not in ALLOWED_MIME_TYPES:
        return {
            "error": "unsupported_mime_type",
            "mime_type": mime_type,
            "message": f"MIME type '{mime_type}' is not supported.",
        }
    detected = sniff_mime_type(data)
    if detected is None:
        return {
            "error": "unsupported_image_format",
            "message": "Could not detect a supported image format from magic bytes.",
        }
    if detected != mime_type:
        return {
            "error": "mime_mismatch",
            "declared_mime_type": mime_type,
            "detected_mime_type": detected,
            "message": "Declared MIME type does not match image magic bytes.",
        }
    return None
