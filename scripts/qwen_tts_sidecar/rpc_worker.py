#!/usr/bin/env python3
"""JSON-lines RPC worker that keeps a local tts-server process resident."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _sidecar_health(url: str) -> bool:
    request = urllib.request.Request(f"{url.rstrip('/')}/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def _start_sidecar(model_path: Path, codec_path: Path, url: str) -> subprocess.Popen:
    parsed_port = url.rsplit(":", 1)[-1]
    host = "127.0.0.1"
    port = parsed_port if parsed_port.isdigit() else "8766"
    binary = os.environ.get("QWEN_TTS_BINARY", "").strip()
    if not binary:
        for name in ("tts-server", "tts-server.exe"):
            from shutil import which

            found = which(name)
            if found:
                binary = found
                break
    if not binary:
        raise RuntimeError("QWEN_TTS_BINARY or tts-server must be available for rpc_worker.")
    env = os.environ.copy()
    command = [
        binary,
        "--model",
        str(model_path),
        "--codec",
        str(codec_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )


def _synthesize(url: str, params: dict) -> bytes:
    payload = {
        "input": params["text"],
        "instructions": params.get("instructions") or "",
        "response_format": "wav",
    }
    if params.get("seed") is not None:
        payload["seed"] = params["seed"]
    if params.get("speed") is not None:
        payload["speed"] = params["speed"]
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.environ.get("QWEN_TTS_TIMEOUT_SECONDS", "120"))) as response:
        return response.read()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = Path(os.environ.get("QWEN_TTS_MODEL_PATH", "models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf"))
    codec_path = Path(os.environ.get("QWEN_TTS_CODEC_PATH", "models/qwen-tokenizer-12hz-Q4_K_M.gguf"))
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()
    if not codec_path.is_absolute():
        codec_path = (repo_root / codec_path).resolve()
    sidecar_url = os.environ.get("QWEN_TTS_RPC_INTERNAL_URL", "http://127.0.0.1:8766")

    proc = _start_sidecar(model_path, codec_path, sidecar_url)
    deadline = time.time() + 120.0
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            print(stderr, file=sys.stderr)
            return 1
        if _sidecar_health(sidecar_url):
            break
        time.sleep(0.5)
    else:
        print("Timed out waiting for internal tts-server.", file=sys.stderr)
        proc.kill()
        return 1

    print("ready", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("method") != "synthesize":
                response = {"id": request_id, "error": "unsupported method"}
            else:
                wav_bytes = _synthesize(sidecar_url, request.get("params") or {})
                response = {
                    "id": request_id,
                    "result": {"wav_b64": base64.b64encode(wav_bytes).decode("ascii")},
                }
        except Exception as exc:
            response = {"id": None, "error": str(exc)}
        print(json.dumps(response), flush=True)
    proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
