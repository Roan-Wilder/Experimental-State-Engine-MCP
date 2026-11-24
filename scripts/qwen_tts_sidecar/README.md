# Qwen3-TTS Sidecar Setup

AI Eco Lab integrates with [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) **outside** the Python package. Build the native tools locally, then point environment variables at the binaries and GGUF files.

## Verified API (upstream)

| Component | Capability |
|-----------|------------|
| `qwen-tts` CLI | VoiceDesign via `--instruct "<description>"` |
| `tts-server` | Persistent HTTP server; loads talker + codec once |
| `POST /v1/audio/speech` | OpenAI-compatible synthesis |
| JSON field `instructions` | Maps to VoiceDesign instruct string |
| `GET /health` | Liveness probe |
| Output | 24 kHz mono WAV when `response_format` is `wav` |

Run verification:

```powershell
python scripts/qwen_tts_sidecar/verify_sidecar.py
```

## Build (Windows example)

```powershell
# Run from the AI Eco Lab repository root
git clone https://github.com/ServeurpersoCom/qwentts.cpp.git tools/qwentts
cd tools/qwentts
# Follow upstream README for CMake build (CPU/Vulkan/CUDA as available)
```

Expected binaries:

- `tts-server.exe` — production path (`QWEN_TTS_BACKEND=sidecar_http`)
- `qwen-tts.exe` — dev fallback (`QWEN_TTS_BACKEND=cli_oneshot`)

## Models

Download GGUF files (example names):

- `qwen-talker-1.7b-voicedesign-Q4_K_M.gguf`
- `qwen-tokenizer-12hz-Q4_K_M.gguf`

Place under `models/` (gitignored) or set absolute paths via environment variables.

## Start persistent sidecar

```powershell
$env:GGML_BACKEND = "CPU"   # or CUDA0 / Vulkan0 when available
.\tts-server.exe `
  --model C:\path\to\qwen-talker-1.7b-voicedesign-Q4_K_M.gguf `
  --codec C:\path\to\qwen-tokenizer-12hz-Q4_K_M.gguf `
  --host 127.0.0.1 `
  --port 8765
```

Or auto-start from MCP when `QWEN_TTS_AUTO_START_SIDECAR=1`.

## Example synthesis request

```json
POST /v1/audio/speech
{
  "input": "Hello. This is the AI Eco Lab experimental voice system.",
  "instructions": "adult male voice, clear, calm, moderately deep, conversational, natural pacing",
  "response_format": "wav",
  "seed": 42
}
```

## RPC worker (optional)

`scripts/qwen_tts_sidecar/rpc_worker.py` exposes JSON-lines RPC over stdin/stdout while keeping an internal `tts-server` resident. Use with `QWEN_TTS_BACKEND=sidecar_rpc`.
