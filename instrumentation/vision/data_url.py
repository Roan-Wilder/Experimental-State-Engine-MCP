from __future__ import annotations

import base64
import re

DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<payload>.+)$", re.DOTALL)


def build_data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_data_url(data_url: str) -> dict[str, str | bytes] | dict[str, str]:
    match = DATA_URL_PATTERN.match(data_url.strip())
    if not match:
        return {"error": "invalid_data_url", "message": "Data URL must match data:{mime};base64,{payload}."}
    mime_type = match.group("mime")
    payload = match.group("payload")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return {"error": "invalid_data_url", "message": "Base64 payload is invalid."}
    return {"mime_type": mime_type, "data": raw}
