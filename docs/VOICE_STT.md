# AI Eco Lab — Local Speech-to-Text (STT)

**Status:** Local microphone input modality (parallel to Qwen3-TTS output)

---

## Scientific separation

Speech recognition is an **input modality**. Transcribed text is **not** evidence of consciousness, emotion, intent, self-awareness, or internal state.

| Category | Storage | Must NOT |
|----------|---------|----------|
| STT input | `research/raw/stt_events.jsonl` | Be labeled as emotion or introspection |
| Voice output (TTS) | `research/raw/voice_events.jsonl` | Be merged with STT as subjective inference |
| Experimental state | `engine_state` / provider | Change because STT ran |

See also: [`VOICE_TTS.md`](VOICE_TTS.md) for the verified Qwen3-TTS output path.

---

## Architecture

```text
Microphone (explicit, bounded)
  -> AudioCaptureAdapter
       -> VoiceInputService
            -> STTAdapter (faster-whisper CPU)
                 -> TranscriptionResult
            -> research/raw/stt_events.jsonl

Future/full loop (not auto-wired):
  STT -> Gemma -> MCP speak_text -> VoiceService -> Qwen3-TTS
```

STT is **not** coupled to TTS inside `VoiceInputService`. Optional demos call existing `VoiceService` only when explicitly requested.

---

## Install

Core project dependencies do **not** include STT. Install optional extras:

```powershell
cd C:\Users\John\ai-eco-lab
.\venv\Scripts\python.exe -m pip install -e ".[stt]"
```

Or:

```powershell
pip install -r requirements-stt.txt
```

No Whisper model weights are bundled. Models are loaded only on explicit STT initialization.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `STT_ENABLED` | `0` | Gate MCP STT tools |
| `STT_BACKEND` | `faster_whisper` | `faster_whisper`, `null`, `mock` (mock = unit tests only) |
| `STT_MODEL` | `base` | faster-whisper model size name |
| `STT_MODEL_PATH` | _(unset)_ | Local model directory (offline; no Hub download) |
| `STT_DEVICE` | `cpu` | Inference device (CPU first on AMD/Vulkan systems) |
| `STT_COMPUTE_TYPE` | `int8` | faster-whisper compute type |
| `STT_LANGUAGE` | `en` | Language hint |
| `STT_SAMPLE_RATE` | `16000` | Capture + STT input sample rate |
| `STT_MAX_RECORD_SECONDS` | `10` | Hard cap for MCP/capture duration |
| `STT_EVENTS_PATH` | `research/raw/stt_events.jsonl` | Provenance log |
| `STT_TEMP_DIR` | `tmp_stt` | Controlled temp audio directory |
| `STT_PERSIST_AUDIO` | `0` | Keep raw mic audio (default: delete after transcription) |
| `STT_ALLOW_MODEL_DOWNLOAD` | `0` | Allow Hub download on init if model not cached |
| `STT_HARDWARE_TEST` | `0` | Enable hardware integration tests |

Optional correlation (only when set):

- `AI_ECO_LAB_EXPERIMENT_ID`
- `AI_ECO_LAB_TRIAL_ID`
- `AI_ECO_LAB_TRIAL_STAGE`

---

## Privacy

- **No always-on microphone** — capture only on explicit MCP tool or CLI invocation.
- **No cloud upload** — all processing remains local.
- **No telemetry** — no remote speech APIs.
- Temp WAV files are deleted after transcription unless `STT_PERSIST_AUDIO=1`.

---

## MCP tools

When `STT_ENABLED=1` (independent of `TTS_ENABLED` / Qwen3-TTS):

| Tool | Purpose |
|------|---------|
| `get_stt_status` | Backend, model, device, microphone count |
| `get_microphone_devices` | Enumerate local input devices |
| `record_voice_input` | Bounded mic capture + local transcription |
| `transcribe_audio` | Transcribe explicit base64 WAV (no mic) |

There is **no** always-listening MCP tool.

---

## Manual smoke test

```powershell
$env:STT_BACKEND = "faster_whisper"
$env:STT_ALLOW_MODEL_DOWNLOAD = "1"   # first run only, or set STT_MODEL_PATH
.\venv\Scripts\python.exe -m voice.stt_smoke_test --duration 5
```

Optional TTS echo (uses existing `VoiceService`, requires TTS env separately):

```powershell
.\venv\Scripts\python.exe -m voice.stt_smoke_test --duration 5 --echo-tts
```

---

## Voice loop demo (explicit)

```powershell
.\venv\Scripts\python.exe -m voice.voice_loop_demo --duration 5 --speak
```

`--speak` calls the **existing** `VoiceService.synthesize()` path — it does not rewrite TTS integration.

---

## Testing

Unit tests (mock backend):

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_voice_stt.py tests/test_voice_stt_boundary.py --basetemp=tmp_pytest -q
```

Hardware integration (fail if mic/model/deps missing; never mock fallback):

```powershell
$env:STT_HARDWARE_TEST = "1"
$env:STT_BACKEND = "faster_whisper"
$env:STT_ALLOW_MODEL_DOWNLOAD = "1"   # or STT_MODEL_PATH
.\venv\Scripts\python.exe -m pytest tests/integration/test_stt_hardware.py -m integration -v
```

---

## Limitations

- CPU-first inference (no CUDA assumption)
- No streaming STT over MCP
- No always-on or background recording
- Model must be local or explicitly allowed to download
- Transcription quality depends on microphone and environment noise
