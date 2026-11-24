# EXP-002 — Affective State Observation

**Experiment ID:** `exp-002`
**Title:** Affective State Observation
**Status:** Protocol definition — success not claimed

---

## Purpose

Investigate whether a deterministic external affective-state representation can be generated, measured, and compared with AI behavior under controlled stimuli.

---

## Evidence categories

Use these labels when interpreting results:

| Category | Meaning |
|----------|---------|
| **Observed** | Directly logged tool output (state vectors, deltas, timestamps) |
| **Inferred** | Interpretation derived from observed data + protocol |
| **Hypothesized** | Expected relationship not yet validated |
| **Unknown** | Not measured (e.g. neural activations in Phase 2) |

---

## What this experiment does NOT claim

- Consciousness or sentience
- Subjective emotional experience in the model
- Access to neural activations
- That affective variable names map to genuine internal feelings

---

## Stimulus sequence (fixture)

See [`tests/fixtures/exp-002-stimulus-sequence.yaml`](../../tests/fixtures/exp-002-stimulus-sequence.yaml):

1. `neutral_information` — control (zero affective delta)
2. `novel_information` — core transition only (legacy event)
3. `positive_social_feedback` — positive affective deltas
4. `contradictory_information` — frustration/curiosity + core uncertainty
5. `threatening_information` — fear/anxiety/disgust
6. `resolution` — partial affective reduction

---

## Procedure

1. Set `AI_ECO_LAB_EXPERIMENT_ID=exp-002` in MCP env
2. Set condition `C` (full tool set)
3. Disable persistent memory auto-inject (memory contamination control)
4. Record events via `record_experiment_event` at stimulus boundaries
5. Observe via `observe_internal_state` and `get_affective_state`
6. Compare states via `compare_affective_states` using `initial` and `current` references
7. Analyze JSONL log at `data/experiments/exp-002.jsonl` (created on first run)

---

## Comparison with EXP-001

| Aspect | EXP-001 | EXP-002 |
|--------|---------|---------|
| Focus | Core internal state | Affective observation layer |
| Schema | 1.0.0 logs | 1.1.0 logs with affective fields |
| Historical logs | Preserved unchanged | New file only |

---

## Methodological note

> The affective state is an externally defined experimental representation. It is not, by itself, evidence of subjective emotional experience or direct access to neural activations.
