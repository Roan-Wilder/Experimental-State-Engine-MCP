# AI Eco Lab — Multi-Channel Instrumentation

**Schema version:** 2.1.0
**Status:** Research instrumentation layer (Phase 1 + vision extension)

---

## Scientific separation

This system distinguishes five categories that must never be conflated:

| Channel | What it is | What it is NOT |
|---------|------------|----------------|
| **behavioral** | Predictions, confidence, revisions, tool outcomes | Proof of internal experience |
| **engine_state** | Externally defined experimental state vectors | Neural activations or subjective emotion |
| **model_io** | Runtime metrics + vision provenance (`model_io.vision`) when enabled | Fabricated logprobs or token counts |
| **self_report** | Model verbal descriptions of its own state | Evidence of genuine introspection |
| **internal_activation** | Future hidden-state checkpoint stats | Embeddings substituted for activations |
| **voice output** | Local TTS acoustic events (`research/raw/voice_events.jsonl`) | Evidence of consciousness, emotion, or introspection |
| **speech input (STT)** | Local transcription events (`research/raw/stt_events.jsonl`) | Evidence of intent, emotion, consciousness, or internal state |

A successful self-report is **behavioral data**, not evidence of introspection. Model-side discipline for self-reports and epistemic categories is defined in [`RESEARCH_SUBJECT_PROTOCOL.md`](RESEARCH_SUBJECT_PROTOCOL.md).
Generated speech from Qwen3-TTS is **acoustic output**, stored in a separate voice channel and never merged into engine_state or self_report. Local STT transcriptions are stored in `research/raw/stt_events.jsonl` as input-modality provenance only. Voice personas (`research/raw/voice_personas.jsonl`) and TTS benchmarks (`research/benchmarks/voice_tts.jsonl`) are also separate from multi-channel observations. See [`VOICE_TTS.md`](VOICE_TTS.md) and [`VOICE_STT.md`](VOICE_STT.md).
An engine_state change is **externally defined**, not proof the model experienced that state.

---

## Architecture

```text
TrialRunner / MCP tools
  → InstrumentedStateFacade
       → ExperimentalStateProvider (unchanged transition engine)
       → ObservationStore (research/raw/observations.jsonl)
       → NullModelIOAdapter / NullInternalActivationAdapter
```

Legacy tool logs remain in `data/experiments/{experiment_id}.jsonl`.
Multi-channel observations are append-only in `research/raw/observations.jsonl`.

---

## Observation record

Each observation has a real UUID (`observation_id`) and optional chain link (`previous_observation_id`).

Schema: [`instrumentation/schemas/observation_v1.json`](../instrumentation/schemas/observation_v1.json)

Unavailable metrics are stored as `"status": "unsupported"` with `null` metrics — never approximated.

---

## MCP tools

| Tool | Read-only | Purpose |
|------|-----------|---------|
| `capture_observation` | Yes | Unified 5-channel snapshot |
| `record_self_report` | Self-report channel only | Verbal self-report without engine mutation |
| `compare_observations` | Yes | Diff two observation UUIDs across channels |
| Existing 6 tools | Mixed | Backward compatible; routed through facade |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_ECO_LAB_TRIAL_ID` | — | Bind MCP observations to a trial |
| `AI_ECO_LAB_TRIAL_STAGE` | `free` | Current protocol stage |
| `AI_ECO_LAB_OBSERVATION_MODE` | `permissive` | `strict` enforces one observation per stage |
| `AI_ECO_LAB_ALLOW_KEYWORD_REFS` | off | Set `1` to allow legacy `initial`/`current` refs |

---

## Metadata validation

`record_experiment_event` rejects:

- Unknown metadata keys
- Protocol artifacts (`<tool_call>`, `function_call`, etc.)
- Oversized strings

Invalid metadata returns `{"error": "invalid_metadata", ...}` with **no state change**.

---

## Loop prevention

Observation tools are read-only and do not trigger other tools automatically.

In `strict` observation mode:

- `trial_id` required
- One observation per stage (configurable)
- Recursion depth limit (default 3)

---

## Pilot protocol

Run 10–20 simulation trials:

```powershell
cd C:\Users\John\ai-eco-lab
python -m research.run_trial `
  --protocol research/protocols/instrumentation_pilot_v1.yaml `
  --experiment-id exp-instrumentation-pilot-001 `
  --master-seed 42 `
  --trials-per-condition 4 `
  --mode simulation `
  --analyze
```

Protocol: [`research/protocols/instrumentation_pilot_v1.yaml`](../research/protocols/instrumentation_pilot_v1.yaml)

---

## Historical data

Existing logs in `data/experiments/` and `research/raw/trials.jsonl` are **never rewritten**.
Known anomalies remain as evidence of prior system behavior.

---

## Vision / image extension

See [`docs/VISION_INSTRUMENTATION.md`](VISION_INSTRUMENTATION.md) for image provenance, visual trial stages, and LM Studio data_url request construction.

---

## Future phases

| Phase | Scope |
|-------|-------|
| **2** | Direct Gemma/Transformers hidden-state adapter |
| **3** | Probing and representation similarity analysis |
| **4** | Causal intervention at activation checkpoints |
