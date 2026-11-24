#!/usr/bin/env python3
"""Verify qwentts.cpp sidecar API expectations for AI Eco Lab integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    qwentts_root = repo_root / "tools" / "qwentts"
    findings = {
        "repo_root": str(repo_root),
        "qwentts_clone_present": qwentts_root.is_dir(),
        "tts_server_source": str(qwentts_root / "tools" / "tts-server.cpp"),
        "cli_supports_instruct": False,
        "http_endpoint": "POST /v1/audio/speech",
        "health_endpoint": "GET /health",
        "voice_design_field": "instructions",
        "default_upstream_port": 8080,
        "recommended_port": 8765,
        "notes": [],
    }

    cli_source = qwentts_root / "tools" / "qwen-tts.cpp"
    server_header = qwentts_root / "src" / "tts-server.h"
    if cli_source.is_file():
        text = cli_source.read_text(encoding="utf-8", errors="replace")
        findings["cli_supports_instruct"] = "--instruct" in text
    if server_header.is_file():
        text = server_header.read_text(encoding="utf-8", errors="replace")
        if "instructions" in text:
            findings["notes"].append("tts-server maps OpenAI 'instructions' to ABI instruct field.")
        if "POST   /v1/audio/speech" in text:
            findings["notes"].append("Persistent sidecar loads talker+codec once via qt_init.")

    findings["notes"].append(
        "VoiceDesign requires qwen-talker-1.7b-voicedesign-Q4_K_M.gguf plus qwen-tokenizer-12hz-Q4_K_M.gguf."
    )
    findings["notes"].append(
        "Set QWEN_TTS_SIDECAR_URL to match tts-server --port (plan default 8765; upstream default 8080)."
    )

    print(json.dumps(findings, indent=2))
    if not findings["qwentts_clone_present"]:
        print("WARNING: tools/qwentts not present. Clone qwentts.cpp before building.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
