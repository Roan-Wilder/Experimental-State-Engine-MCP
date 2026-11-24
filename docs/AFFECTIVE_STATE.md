# AI Eco Lab — Affective Observation Layer

**Schema version:** 1.1.0
**Status:** Experimental extension to Phase 2 internal-state MCP server

---

## Disclaimer

**The affective state is an externally defined experimental representation. It is not, by itself, evidence of subjective emotional experience or direct access to neural activations.**

Affective observation variables describe a deterministic experimental vector exposed via MCP tools. They are **not** proof that the language model literally feels these emotions.

---

## Architecture

```text
CORE STATE (12 dimensions, unchanged)
  activation, stability, novelty, uncertainty, attention, persistence,
  coherence, exploration, engagement, conflict, prediction_pressure, internal_activity

AFFECTIVE OBSERVATION STATE (12 dimensions, additive)
  love, affection, joy, trust, curiosity, surprise,
  fear, anger, frustration, sadness, disgust, anxiety

Both vectors:
  - share one YAML transitions table
  - use one apply_transition() engine (filtered by dimension list)
  - log to the same JSONL experiment files
```

### Three-way separation

| Layer | Description |
|-------|-------------|
| **A — External experimental state** | Deterministic engine (core + affective vectors) |
| **B — Model behavior** | Language model outputs and tool calls |
| **C — Model internal representation** | Future neural/activation providers (not implemented) |

Research may compare A ↔ B now, and A ↔ C when instrumentation exists. **A is not C.**

---

## Affective dimensions

All values are continuous in `[0.0, 1.0]`:

- `0.0` — absent/minimal represented intensity
- `1.0` — maximum represented intensity

Values are clamped; the model cannot arbitrarily set affective values except through configured experimental events.

---

## YAML configuration

Configuration lives in [`state/config.yaml`](../state/config.yaml):

- `affective_dimensions` — list of 12 variable names
- `affective_initial_values` — baseline vector (default all 0.0)
- `transitions` — shared event→delta map; core keys route to core vector, affective keys to affective vector

Example stimulus events (experimental, not scientifically validated):

| Event | Example affective deltas |
|-------|--------------------------|
| `neutral_information` | `{}` (zero-delta control) |
| `positive_social_feedback` | affection +0.15, trust +0.12, joy +0.10 |
| `contradictory_information` | frustration +0.20, curiosity +0.10; core uncertainty +0.15 |
| `threatening_information` | fear +0.20, anxiety +0.15, disgust +0.05 |

Existing Phase 2 events (`novel_information`, `contradiction`, etc.) are **unchanged** for core deltas.

---

## MCP tools

### Existing (preserved)

| Tool | Purpose |
|------|---------|
| `observe_internal_state` | Core state + **additive** `affective_state` field |
| `observe_state_history` | Bounded transition history |
| `reflect_on_internal_state` | Reflection context + affective fields |
| `record_experiment_event` | Applies core and affective deltas from YAML |

### New

| Tool | Purpose |
|------|---------|
| `get_affective_state` | Returns current affective observation vector only |
| `compare_affective_states` | Analytical diff between two recorded references |

Comparison output uses neutral language:

> *"The affective observation variable 'frustration' increased by 0.2100."*

It does **not** infer causality or claim the model "became frustrated."

---

## Event recording

Use **`record_experiment_event`** only (no separate affective recording tool).

```json
{
  "event_type": "positive_social_feedback",
  "metadata": {
    "description": "Controlled positive feedback stimulus"
  }
}
```

Response includes additive fields:

- `previous_affective_state`
- `new_affective_state`
- `affective_delta`

---

## JSONL logging

New records (schema 1.1.0) may include optional fields:

- `previous_affective_state`
- `new_affective_state`
- `affective_delta`
- `schema_version`

Historical logs without these fields remain valid evidence and are **not modified**.

---

## Control conditions

- **A–F deception conditions** apply to both core and affective observations (conditions D/E/F).
- **`neutral_information`** provides explicit zero-delta affective control.
- Condition B enables `get_affective_state` alongside observe/record.

---

## Reproducibility

The same event sequence on the same initial state produces identical final vectors. Use `replay_transitions()` in [`state/engine.py`](../state/engine.py) or replay logged events from JSONL.

---

## Limitations

- Affective state resets when the MCP subprocess restarts (same as core state)
- No automatic chat→event bridge
- No neural readout — affective variables are experimental labels, not measured activations
- Stimulus→delta mappings are configurable hypotheses, not validated psychology

---

## Related documents

- Experiment protocol: [`docs/experiments/EXP-002-AFFECTIVE-OBSERVATION.md`](experiments/EXP-002-AFFECTIVE-OBSERVATION.md)
- IP documentation: `docs/ip/` (unchanged by this extension)
