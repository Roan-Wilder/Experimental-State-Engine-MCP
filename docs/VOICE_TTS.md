# AI Eco Lab — Qwen3-TTS VoiceDesign

**Status:** Phase 1 local TTS integration (parallel to state engine)

---

## Scientific separation

Qwen3-TTS is an **output modality**. Generated speech is **not** evidence of consciousness, emotion, sentience, or internal affective state.

VoiceDesign `voice_instruction` strings are **experimental controls for speech generation**, not measurements of the underlying Gemma model.

| Category | Storage | Must NOT |
|----------|---------|----------|
| Model self-report | `self_report` channel | Be treated as acoustic ground truth |
| Experimental state | `engine_state` / provider | Change because TTS ran |
| Voice output | `voice_events.jsonl` + optional artifacts | Be labeled as emotion or introspection |

---

## Architecture

```text
LM Studio Gemma
  -> MCP speak_text / get_voice_status
       -> VoiceService (voice/service.py)
            -> VoiceAdapter (mock | QwenTTSAdapter)
                 -> tts-server sidecar (qwentts.cpp)
            -> research/raw/voice_events.jsonl
            -> research/audio_artifacts/{uuid}.wav
```

Voice tools **do not** call `record_experiment_event`, `observe_*`, or mutate `ExperimentalStateProvider`.

---

## Prerequisites

1. Build [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) outside the repo (see [`scripts/qwen_tts_sidecar/README.md`](../scripts/qwen_tts_sidecar/README.md)).
2. Download GGUF models:
   - `qwen-talker-1.7b-voicedesign-Q4_K_M.gguf`
   - `qwen-tokenizer-12hz-Q4_K_M.gguf`
3. Start `tts-server` manually (recommended) or set `QWEN_TTS_AUTO_START_SIDECAR=1`.

### Windows runtime (verified path)

Start the server **once** in its own terminal (do not auto-start a second copy if already running):

```powershell
$QWEN = "C:\Users\John\ai-eco-lab\Qwen3-TTS"
$BIN  = "$QWEN\qwentts-runtime\build\Release"
$env:GGML_BACKEND = "Vulkan0"   # RX 580; use CPU if Vulkan unavailable

& "$BIN\tts-server.exe" `
  --model "$QWEN\qwen-talker-1.7b-voicedesign-Q4_K_M.gguf" `
  --codec "$QWEN\qwen-tokenizer-12hz-Q4_K_M.gguf" `
  --host 127.0.0.1 --port 8765 --lang English
```

LM Studio / MCP environment (preserve existing variable names):

```powershell
TTS_ENABLED=1
QWEN_TTS_BACKEND=sidecar_http
QWEN_TTS_MODEL_PATH=C:\Users\John\ai-eco-lab\Qwen3-TTS\qwen-talker-1.7b-voicedesign-Q4_K_M.gguf
QWEN_TTS_CODEC_PATH=C:\Users\John\ai-eco-lab\Qwen3-TTS\qwen-tokenizer-12hz-Q4_K_M.gguf
QWEN_TTS_SIDECAR_URL=http://127.0.0.1:8765
QWEN_TTS_BINARY=C:\Users\John\ai-eco-lab\Qwen3-TTS\qwentts-runtime\build\Release\tts-server.exe
QWEN_TTS_AUTO_START_SIDECAR=0
QWEN_TTS_RUNTIME_COMMIT=71ad93d591a2811f35db77e27c02acba091c9e9b
GGML_BACKEND=Vulkan0
```

`QwenTTSAdapter` uses existing server when `GET /health` succeeds; it does not spawn another process unless `QWEN_TTS_AUTO_START_SIDECAR=1` and health check fails.

Verify API expectations:

```powershell
python scripts/qwen_tts_sidecar/verify_sidecar.py
```

---

## Master switch and resource profiles

TTS is controlled independently from STT via the master switch **`TTS_ENABLED`** (default **`0`**).

When `TTS_ENABLED=0`, the MCP server does **not** initialize `VoiceService`, load Qwen GGUF models, start `tts-server`, initialize Vulkan/GPU resources for TTS, or register TTS MCP tools (`speak_text`, personas, etc.). Research/state/experiment MCP tools remain available.

When `TTS_ENABLED=1`, existing Qwen3-TTS behavior is unchanged.

Legacy alias: if `TTS_ENABLED` is unset, `QWEN_TTS_ENABLED` is consulted for backward compatibility.

### Useful configuration profiles

| Profile | `STT_ENABLED` | `TTS_ENABLED` | Use case |
|---------|---------------|---------------|----------|
| **Research / tool-only** (default) | `0` | `0` | Lightweight MCP — state, observations, experiments only |
| **Voice input** | `1` | `0` | Local microphone STT without loading Qwen3-TTS |
| **Full voice** | `1` | `1` | Microphone input + Qwen3-TTS speech output |

Disabling TTS (`TTS_ENABLED=0`) is intentional: it prevents initializing the Qwen3-TTS runtime and associated GPU resources when speech output is not needed.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TTS_ENABLED` | `0` | **Master gate** for all TTS (MCP tools, VoiceService, sidecar) |
| `QWEN_TTS_ENABLED` | `0` | Legacy alias when `TTS_ENABLED` is unset |
| `QWEN_TTS_BACKEND` | `sidecar_http` | `sidecar_http`, `sidecar_rpc`, `cli_oneshot`, `mock` |
| `QWEN_TTS_MODEL_PATH` | `models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf` | Talker GGUF |
| `QWEN_TTS_CODEC_PATH` | `models/qwen-tokenizer-12hz-Q4_K_M.gguf` | Codec GGUF |
| `QWEN_TTS_BINARY` | _(PATH discovery)_ | `tts-server` or `qwen-tts` |
| `QWEN_TTS_SIDECAR_URL` | `http://127.0.0.1:8765` | HTTP sidecar base URL |
| `QWEN_TTS_AUTO_START_SIDECAR` | `0` | Auto-launch `tts-server` |
| `QWEN_TTS_LANGUAGE` | `English` | Default language |
| `QWEN_TTS_DEFAULT_INSTRUCTION` | neutral preset text | Fallback instruction |
| `QWEN_TTS_MAX_TEXT_LENGTH` | `500` | Sanitization cap |
| `QWEN_TTS_TIMEOUT_SECONDS` | `120` | Synthesis timeout |
| `QWEN_TTS_PERSIST_ARTIFACTS` | `1` | Save UUID WAV artifacts |
| `QWEN_TTS_ARTIFACT_ROOT` | `research/audio_artifacts` | Allowlisted artifact root |
| `GGML_BACKEND` | auto | Pass through to sidecar (`CPU`, `CUDA0`, `Vulkan0`, ...) |
| `QWEN_TTS_RUNTIME_COMMIT` | _(unset)_ | Optional qwentts.cpp git commit recorded in voice event metadata |
| `QWEN_TTS_HARDWARE_ARTIFACT_ROOT` | `Qwen3-TTS/artifacts/_integration_test` | Hardware test WAV output directory |

Optional correlation (only when set — never invented):

- `AI_ECO_LAB_EXPERIMENT_ID`
- `AI_ECO_LAB_TRIAL_ID`
- `AI_ECO_LAB_TRIAL_STAGE`

---

## MCP tools

When `TTS_ENABLED=1`:

### `speak_text`

Parameters: `text`, optional `voice_instruction`, `voice_preset`, `language`, `speed`, `seed`, `save_artifact`.

Returns:

1. FastMCP `Audio` block (`audio/wav`, 24 kHz mono)
2. JSON metadata text (`event_id`, `artifact_id`, hashes, timing, disclaimer)

On failure: structured JSON error (MCP server stays alive).

### `get_voice_status`

Returns adapter status: backend, model/codec paths and SHA-256, sidecar PID/URL, `last_error`.

---

## Voice presets

[`voice/presets.yaml`](../voice/presets.yaml):

- `research_neutral`
- `calm`
- `energetic`

---

## Sidecar API (verified upstream)

`tts-server` exposes OpenAI-compatible endpoints:

- `POST /v1/audio/speech` with `input`, `instructions`, `response_format: "wav"`
- `GET /health`

Example:

```json
{
  "input": "Hello. This is the AI Eco Lab experimental voice system.",
  "instructions": "adult male voice, clear, calm, moderately deep, conversational, natural pacing",
  "response_format": "wav",
  "seed": 42
}
```

---

## Testing

Unit tests (mock backend):

```powershell
cd C:\Users\John\ai-eco-lab
.\venv\Scripts\python.exe -m pytest tests/test_voice_tts.py --basetemp=tmp_pytest -q
```

Hardware integration tests (fail if server unavailable; never fall back to mock):

```powershell
# tts-server must already be listening on QWEN_TTS_SIDECAR_URL
$env:QWEN_TTS_RUN_HARDWARE_TEST = "1"
$env:QWEN_TTS_BACKEND = "sidecar_http"
# set model/codec/binary paths as in Windows runtime section above
.\venv\Scripts\python.exe -m pytest tests/integration/test_qwen_tts_hardware.py -m integration -q
```

On Windows, if pytest fails with `PermissionError` on `%TEMP%\pytest-of-*`, use project-local temp (also set in `pyproject.toml`):

```powershell
.\venv\Scripts\python.exe -m pytest tests/integration/test_qwen_tts_hardware.py -m integration -q --basetemp=tmp_pytest
```

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Tools missing | `TTS_ENABLED=1` in LM Studio MCP env |
| Sidecar unreachable | `GET /health`, port matches `QWEN_TTS_SIDECAR_URL` |
| VoiceDesign error | Talker must be VoiceDesign variant; `--instruct` / `instructions` required |
| Slow synthesis | Use persistent `tts-server`, not `cli_oneshot` |
| No audio in LM Studio UI | Metadata + local artifact still logged; client may ignore MCP audio blocks |

---

## Voice personas (Steps 21)

Two-stage VoiceDesign workflow:

**Stage A — creation:** `create_voice_persona` → reference WAV → `research/raw/voice_personas.jsonl`
**Stage B — reuse:** `speak_text(..., voice_id=...)` applies stored instruction + seed for consistent experimental voice.

Persona records include versioning (`active` / `superseded`). Regeneration creates version N+1 when model hash, instruction, or seed changes.

> This voice persona is an experimental TTS output profile. It does not represent the identity, personality, consciousness, or emotional state of Gemma.

VoiceDesign reuse uses stored instruction + seed (not WAV cloning). Reference WAV is archival/reproducibility only.

---

## Output modes and playback (Step 23)

| Mode | Behavior |
|------|----------|
| `generate_only` | Log event + metadata only |
| `generate_and_save` | Save artifact; no inline MCP audio |
| `generate_and_return` | Default — MCP Audio + metadata |
| `generate_and_play` | Local playback if `QWEN_TTS_PLAYBACK_ENABLED=1` |

Playback uses `PlaybackAdapter` (separate from `VoiceAdapter`). Disabled by default for automated experiments.

---

## Hardware benchmark (Step 22)

```powershell
python -m voice.benchmark --seed 42
```

Appends rows to `research/benchmarks/voice_tts.jsonl` (separate from observations). Does not mutate experimental state.

---

## STT input boundary (Step 24 — implemented)

Local speech-to-text is implemented as a **separate input modality**. See [`VOICE_STT.md`](VOICE_STT.md).

```text
Microphone -> STTAdapter -> Gemma -> MCP speak_text -> VoiceService -> Qwen3-TTS
```

STT does not affect the experimental state engine and is not coupled to TTS inside `VoiceInputService`.

---

## Acoustic analysis hook (Step 25)

`AcousticAnalyzerRegistry` runs measurable analyzers after synthesis. `BasicAcousticAnalyzer` emits duration, RMS, and peak amplitude. Reserved null fields: f0, speech/pause timing, spectral metrics.

Acoustic measurements are **signal properties only** — never mapped to emotion, consciousness, sentience, or subjective affect.

---

## Limitations

- No streaming PCM over MCP
- No `stop_speaking` cancel API
- STT is separate; see [`VOICE_STT.md`](VOICE_STT.md)
- `cli_oneshot` reloads models each call (dev only)
