from __future__ import annotations

import os
from typing import Any


def detect_vision_capability(protocol: dict[str, Any] | None = None) -> str:
    env_enabled = os.environ.get("AI_ECO_LAB_VISION_ENABLED", "0").strip() == "1"
    protocol_flag = False
    if protocol:
        instrumentation = protocol.get("instrumentation", {})
        vision = protocol.get("vision", {})
        protocol_flag = bool(
            instrumentation.get("vision_enabled")
            or vision.get("capable")
            or protocol.get("vision_capable")
        )
    if env_enabled or protocol_flag:
        return "supported"
    return "unsupported"
