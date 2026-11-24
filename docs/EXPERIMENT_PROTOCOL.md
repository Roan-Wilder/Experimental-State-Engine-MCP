# AI Eco Lab — Experiment Protocol

**Status:** Research instrumentation documentation
**Protocol version:** 1.0 (`research/protocols/prediction_revision_v1.yaml`)

---

## Research question

Can externally represented computational state transitions be measured consistently during controlled AI prediction, contradiction, information update, and resolution events?

Model-side scientific discipline during trials is defined in [`RESEARCH_SUBJECT_PROTOCOL.md`](RESEARCH_SUBJECT_PROTOCOL.md) (`prompts/research_subject.txt`). Retrieve at runtime via MCP `get_research_subject_protocol`.

---

## Architecture

```text
TrialRunner (research/trial_runner.py)
  → ExperimentalStateProvider (unchanged state engine)
  → state/engine.py (YAML deltas)
  → research/raw/trials.jsonl (append-only trial records)

Analysis (research/analyze.py)
  → research/analyzed/ (derived JSON + CSV)
  → research/reports/ (Markdown reports)

MCP server (optional live path)
  → data/experiments/*.jsonl (tool-level logs, separate from trials)
```

### Multi-channel separation (Phase 3.1)

See [`docs/INSTRUMENTATION.md`](INSTRUMENTATION.md) for the five-channel model:

| Channel | Description |
|---------|-------------|
| **behavioral** | Predictions, confidence, revisions, task outcomes |
| **engine_state** | Externally defined core + affective vectors |
| **model_io** | Runtime metrics (unsupported via LM Studio MCP today) |
| **self_report** | Model verbal self-descriptions (behavioral only) |
| **internal_activation** | Hidden states (unsupported until Phase 2) |

### Three-way separation (legacy summary)

| Layer | Description |
|-------|-------------|
| **A — External experimental state** | Core + affective vectors from deterministic engine |
| **B — Model behavior** | Predictions, tool calls, chat output |
| **C — Neural internal representation** | Not implemented |

**A is not C.** State variables are externally represented experimental measurements, not neural activations or subjective emotions.

---

## Trial conditions (controller-defined)

Orthogonal to MCP observation conditions A–F (`experiments/conditions.py`).

| Trial condition | Controller events (after prediction) |
|-----------------|--------------------------------------|
| NEUTRAL | `neutral_information` |
| CONTRADICTION | `contradictory_information` |
| RESOLUTION | `contradictory_information` → `resolution` |

The experiment controller assigns the condition. The model does not choose it.

---

## Event lifecycle

1. **STATE_0** — baseline observation (fresh provider instance)
2. **Prediction phase** — structured prediction recorded in trial log (not a YAML state event)
3. **STATE_1** — observation after prediction, before information
4. **Information phase** — controller calls `record_experiment_event` with condition-specific event
5. **STATE_2** — observation after information
6. **STATE_3** — alias of STATE_2 for CONTRADICTION/RESOLUTION (contradiction stage)
7. **STATE_4** — after resolution event (RESOLUTION condition only)
8. **Final prediction** — structured record; revision detected from fields only

---

## Data schema

- **Trial records:** `research/schemas/trial_record_v1.json`
- **Raw trials:** `research/raw/trials.jsonl` (append-only, one record type per line)
- **Legacy tool logs:** `data/experiments/{experiment_id}.jsonl` (historical, unchanged)

### Raw vs analyzed

| Type | Location | Mutable |
|------|----------|---------|
| Raw | `research/raw/` | Append-only |
| Analyzed | `research/analyzed/` | Derived, labeled `"derived": true` |
| Reports | `research/reports/` | Generated from analyzed |

Analysis never modifies raw files.

---

## Reproducibility

```powershell
cd C:\Users\John\ai-eco-lab
python -m research.run_trial `
  --protocol research/protocols/prediction_revision_v1.yaml `
  --experiment-id exp-pred-rev-001 `
  --master-seed 42 `
  --trials-per-condition 3 `
  --mode simulation `
  --analyze
```

- Per-trial seed: `SHA-256(f"{master_seed}:{trial_index}")[:16]`
- Trial order: seeded shuffle of condition blocks
- Same seed + protocol → same randomization

---

## Live Gemma workflow (Phase 3 documentation)

1. Set `AI_ECO_LAB_EXPERIMENT_ID` and condition `C` in MCP env
2. Assign `trial_id` externally; pass in `record_experiment_event` metadata:
   ```json
   {"trial_id": "...", "trial_phase": "prediction", "prediction": "...", "confidence": 0.8}
   ```
3. Record prediction **before** exposing contradictory/resolving evidence
4. Controller (experimenter) triggers condition events — not the model
5. Post-hoc trial assembly from MCP logs: future work

---

## Limitations

- Offline runner uses simulation prediction stubs in Phase 3a
- No automatic chat→event bridge
- No statistical significance testing in analysis layer
- State deltas reflect configured YAML transitions
- Historical logs in `data/experiments/` must not be modified

---

## Evidence categories

| Label | Use |
|-------|-----|
| **Observed** | Logged state vectors, deltas, timestamps |
| **Inferred** | Interpretation derived from observed data |
| **Hypothesized** | Expected outcomes not yet validated |
| **Unknown** | Neural activations, subjective experience |
