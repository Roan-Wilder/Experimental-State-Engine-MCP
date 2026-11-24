from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class SidecarHandle:
    pid: int | None
    url: str
    managed: bool


class VoiceProcessManager:
    def __init__(
        self,
        *,
        sidecar_url: str,
        binary_path: Path | None,
        model_path: Path,
        codec_path: Path,
        auto_start: bool,
        ggml_backend: str | None,
        timeout_seconds: float,
    ) -> None:
        self.sidecar_url = sidecar_url.rstrip("/")
        self.binary_path = binary_path
        self.model_path = model_path
        self.codec_path = codec_path
        self.auto_start = auto_start
        self.ggml_backend = ggml_backend
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[Any] | None = None
        self._rpc_process: subprocess.Popen[Any] | None = None
        self._rpc_lock = threading.Lock()
        self._request_counter = 0

    @property
    def sidecar_pid(self) -> int | None:
        proc = self._process or self._rpc_process
        if proc and proc.poll() is None:
            return proc.pid
        return None

    def health_check(self) -> bool:
        request = urllib.request.Request(
            f"{self.sidecar_url}/health",
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False

    def ensure_sidecar_http(self) -> SidecarHandle:
        if self.health_check():
            return SidecarHandle(pid=self.sidecar_pid, url=self.sidecar_url, managed=False)
        if not self.auto_start:
            raise RuntimeError(
                f"TTS sidecar is not reachable at {self.sidecar_url}. "
                "Start tts-server manually or set QWEN_TTS_AUTO_START_SIDECAR=1."
            )
        self._start_http_sidecar()
        deadline = time.time() + min(self.timeout_seconds, 120.0)
        while time.time() < deadline:
            if self.health_check():
                return SidecarHandle(pid=self.sidecar_pid, url=self.sidecar_url, managed=True)
            if self._process and self._process.poll() is not None:
                raise RuntimeError("TTS sidecar exited during startup.")
            time.sleep(0.5)
        raise RuntimeError("Timed out waiting for TTS sidecar health check.")

    def _start_http_sidecar(self) -> None:
        if not self.binary_path:
            raise RuntimeError("Cannot auto-start sidecar without QWEN_TTS_BINARY.")
        parsed = urlparse(self.sidecar_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8765
        command = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--codec",
            str(self.codec_path),
            "--host",
            host,
            "--port",
            str(port),
        ]
        env = os.environ.copy()
        if self.ggml_backend:
            env["GGML_BACKEND"] = self.ggml_backend
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )

    def synthesize_http(
        self,
        *,
        text: str,
        instructions: str,
        language: str,
        seed: int | None,
        speed: float | None,
    ) -> bytes:
        self.ensure_sidecar_http()
        payload: dict[str, Any] = {
            "input": text,
            "instructions": instructions,
            "response_format": "wav",
        }
        if seed is not None:
            payload["seed"] = seed
        if speed is not None:
            payload["speed"] = speed
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.sidecar_url}/v1/audio/speech",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                wav_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS sidecar HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TTS sidecar request failed: {exc}") from exc
        if not wav_bytes:
            raise RuntimeError("TTS sidecar returned empty WAV payload.")
        return wav_bytes

    def ensure_sidecar_rpc(self) -> None:
        if self._rpc_process and self._rpc_process.poll() is None:
            return
        if not self.binary_path:
            raise RuntimeError("Cannot start RPC worker without QWEN_TTS_BINARY.")
        worker = Path(__file__).resolve().parents[1] / "scripts" / "qwen_tts_sidecar" / "rpc_worker.py"
        if not worker.is_file():
            raise RuntimeError(f"RPC worker script not found: {worker}")
        env = os.environ.copy()
        env["QWEN_TTS_MODEL_PATH"] = str(self.model_path)
        env["QWEN_TTS_CODEC_PATH"] = str(self.codec_path)
        env["QWEN_TTS_LANGUAGE"] = os.environ.get("QWEN_TTS_LANGUAGE", "English")
        env.setdefault("QWEN_TTS_RPC_INTERNAL_URL", "http://127.0.0.1:8766")
        if self.ggml_backend:
            env["GGML_BACKEND"] = self.ggml_backend
        self._rpc_process = subprocess.Popen(
            [sys.executable, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        assert self._rpc_process.stdout is not None
        deadline = time.time() + min(self.timeout_seconds, 120.0)
        while time.time() < deadline:
            if self._rpc_process.poll() is not None:
                stderr = (self._rpc_process.stderr.read() if self._rpc_process.stderr else "") or ""
                raise RuntimeError(f"RPC worker exited during startup: {stderr}")
            if self._rpc_process.stdout.readline().strip() == "ready":
                return
        raise RuntimeError("Timed out waiting for RPC worker readiness.")

    def synthesize_rpc(
        self,
        *,
        text: str,
        instructions: str,
        language: str,
        seed: int | None,
        speed: float | None,
    ) -> bytes:
        self.ensure_sidecar_rpc()
        with self._rpc_lock:
            assert self._rpc_process and self._rpc_process.stdin and self._rpc_process.stdout
            self._request_counter += 1
            request_id = str(self._request_counter)
            payload = {
                "id": request_id,
                "method": "synthesize",
                "params": {
                    "text": text,
                    "instructions": instructions,
                    "language": language,
                    "seed": seed,
                    "speed": speed,
                },
            }
            self._rpc_process.stdin.write(json.dumps(payload) + "\n")
            self._rpc_process.stdin.flush()
            line = self._rpc_process.stdout.readline()
            if not line:
                raise RuntimeError("RPC worker closed stdout unexpectedly.")
            response = json.loads(line)
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            result = response.get("result") or {}
            wav_b64 = result.get("wav_b64")
            if not isinstance(wav_b64, str):
                raise RuntimeError("RPC worker returned no wav_b64 payload.")
            import base64

            return base64.b64decode(wav_b64.encode("ascii"))

    def synthesize_cli_oneshot(
        self,
        *,
        text: str,
        instructions: str,
        language: str,
        seed: int | None,
    ) -> bytes:
        if not self.binary_path:
            raise RuntimeError("Cannot run cli_oneshot without QWEN_TTS_BINARY.")
        command = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--codec",
            str(self.codec_path),
            "--instruct",
            instructions,
            "--lang",
            language,
            "-o",
            "-",
        ]
        if seed is not None:
            command.extend(["--seed", str(seed)])
        env = os.environ.copy()
        if self.ggml_backend:
            env["GGML_BACKEND"] = self.ggml_backend
        completed = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=self.timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"qwen-tts CLI failed ({completed.returncode}): {stderr}")
        if not completed.stdout:
            raise RuntimeError("qwen-tts CLI returned empty WAV payload.")
        return completed.stdout

    def shutdown(self) -> None:
        for proc in (self._process, self._rpc_process):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._process = None
        self._rpc_process = None
