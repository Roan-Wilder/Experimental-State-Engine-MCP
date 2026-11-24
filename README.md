# AI Eco Lab

Experimental internal-state MCP server for LM Studio consciousness research.

This project investigates whether giving a language model access to a persistent, dynamically changing representation of its own **experimental internal state** produces measurable changes in self-monitoring, reflection, and behavior.

This is **not** a claim of consciousness, sentience, or genuine emotions.

## Architecture

```text
LM Studio Desktop
  -> Gemma 4 v2 Q4_K_M
  -> native tool orchestration
  -> existing plugins (untouched)
  -> AI Eco Lab MCP (10 core tools)
       -> InternalStateProvider
            -> ExperimentalStateProvider (Phase 2 + affective layer)
            -> JacobianLensProvider (future)
```

AI Eco Lab is CPU-only. It does **not** load models, use GPU memory, or perform inference.

## Tools

| Tool | Purpose |
|------|---------|
| `observe_internal_state` | Return measured core state vector + affective_state (no interpretation) |
| `observe_state_history` | Return bounded recent transition history |
| `reflect_on_internal_state` | Return measured context for model-side reflection |
| `record_experiment_event` | Apply an event-driven state transition (core + affective) |
| `get_affective_state` | Return current experimental affective observation vector |
| `compare_affective_states` | Analytical comparison of two recorded affective states |
| `capture_observation` | Read-only 5-channel observation snapshot |
| `record_self_report` | Store model self-report without mutating engine state |
| `compare_observations` | Compare two observation UUIDs across channels |
| `get_research_subject_protocol` | Return canonical research subject discipline protocol (read-only) |

The server never returns claims like "you are anxious." Interpretation belongs to the model, not the engine.

See [`docs/RESEARCH_SUBJECT_PROTOCOL.md`](docs/RESEARCH_SUBJECT_PROTOCOL.md) for model-side research discipline and experimental safeguards.

## Affective observation layer

Phase 2.1 adds a parallel 12-dimensional **affective observation vector** (love, affection, joy, trust, curiosity, surprise, fear, anger, frustration, sadness, disgust, anxiety) using the same YAML delta engine as core state.

See [`docs/AFFECTIVE_STATE.md`](docs/AFFECTIVE_STATE.md) for schema, transitions, logging, and limitations.

> The affective state is an externally defined experimental representation. It is not, by itself, evidence of subjective emotional experience or direct access to neural activations.

## State dimensions (core)

Continuous values in `[0.0, 1.0]`:

- activation, stability, novelty, uncertainty, attention, persistence
- coherence, exploration, engagement, conflict, prediction_pressure, internal_activity

Transitions are configured in [`state/config.yaml`](state/config.yaml).

## Experimental conditions

Set via environment variables in LM Studio `mcp.json`:

| Condition | Behavior |
|-----------|----------|
| A | Baseline — remove MCP entry from `mcp.json` |
| B | Observation + event recording + get_affective_state |
| C | Full tool set with genuine measured state |
| D | Observation returns randomized state |
| E | Observation returns stale state |
| F | Observation returns perturbed state |

Environment variables:

- `AI_ECO_LAB_CONDITION` — `A` through `F` (default `C`)
- `AI_ECO_LAB_EXPERIMENT_ID` — experiment identifier
- `AI_ECO_LAB_SESSION_ID` — session identifier (auto-generated if omitted)
- `AI_ECO_LAB_RNG_SEED` — optional seed for condition D reproducibility

## Setup

Run PowerShell from the downloaded project directory. Requires Python 3.10+.

```powershell
.\setup.ps1
```

## Start MCP server manually

```powershell
.\venv\Scripts\python.exe mcp\internal_state_server.py
```

Diagnostics go to **stderr** only. **stdout** is reserved for MCP protocol traffic.

## Memory contamination control

Keep persistent-memory settings and automatic fact extraction consistent across controlled experimental conditions.

For controlled trials, disable persistent memory auto-inject and AI fact extraction in LM Studio.

## Experimental rigor / trial runner

Phase 3 adds an offline-first **trial runner** for controlled prediction/revision experiments:

- Append-only trial logs: `research/raw/trials.jsonl`
- Derived analysis: `research/analyzed/`
- Reports: `research/reports/`
- Protocol: [`research/protocols/prediction_revision_v1.yaml`](research/protocols/prediction_revision_v1.yaml)

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for lifecycle, schema, and reproducibility.

```powershell
python -m research.run_trial `
  --experiment-id exp-pred-rev-001 `
  --master-seed 42 `
  --trials-per-condition 3 `
  --mode simulation `
  --analyze
```

Trial conditions (NEUTRAL, CONTRADICTION, RESOLUTION) are controller-defined and separate from MCP observation conditions A–F.

## Multi-channel instrumentation (Phase 3.1)

Five scientifically separated channels: **behavioral**, **engine_state**, **model_io**, **self_report**, **internal_activation**.

- Observations: `research/raw/observations.jsonl` (UUID chain with `previous_observation_id`)
- MCP tools: `capture_observation`, `record_self_report`, `compare_observations` (+ existing 10 core tools)
- Pilot protocol: [`research/protocols/instrumentation_pilot_v1.yaml`](research/protocols/instrumentation_pilot_v1.yaml)

See [`docs/INSTRUMENTATION.md`](docs/INSTRUMENTATION.md) for channel definitions, validation, and loop prevention.

### Vision / image instrumentation (Phase 3.2)

- Image provenance: `research/raw/image_provenance.jsonl` (SHA-256 + dimensions; no raw screenshots by default)
- Stimuli: [`research/stimuli/visual_box_task_v1/manifest.yaml`](research/stimuli/visual_box_task_v1/manifest.yaml)
- Protocol: [`research/protocols/visual_prediction_v1.yaml`](research/protocols/visual_prediction_v1.yaml)

See [`docs/VISION_INSTRUMENTATION.md`](docs/VISION_INSTRUMENTATION.md).

### Voice / TTS (Qwen3-TTS VoiceDesign)

- Voice events: `research/raw/voice_events.jsonl` (separate from state engine)
- Voice personas: `research/raw/voice_personas.jsonl` (reusable experimental voices)
- Benchmarks: `research/benchmarks/voice_tts.jsonl`
- Optional artifacts: `research/audio_artifacts/{uuid}.wav`
- MCP tools (when `QWEN_TTS_ENABLED=1`): `speak_text`, `get_voice_status`, `create_voice_persona`, `get_voice_persona`, `list_voice_personas`

See [`docs/VOICE_TTS.md`](docs/VOICE_TTS.md). Requires external [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) build; Gemma inference in LM Studio remains unchanged.

```powershell
python -m voice.benchmark --seed 42
```

```powershell
python -m research.run_visual_trial `
  --experiment-id exp-visual-pilot-001 `
  --mode simulation `
  --analyze
```

```powershell
python -m research.run_trial `
  --protocol research/protocols/instrumentation_pilot_v1.yaml `
  --experiment-id exp-instrumentation-pilot-001 `
  --trials-per-condition 4 `
  --mode simulation `
  --analyze
```

## Logging

Append-only JSONL logs: `data/experiments/{experiment_id}.jsonl`

Schema: [`experiments/schema.json`](experiments/schema.json)

## Security

This MCP server has no broad filesystem access or credentials. Voice TTS (when enabled) may call a **localhost-only** qwentts sidecar and write UUID artifacts under `research/audio_artifacts/`; subprocess invocation uses argv lists without shell interpolation.
