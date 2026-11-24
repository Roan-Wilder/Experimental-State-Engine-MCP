# AI Eco Lab — Vision / Image Instrumentation

**Schema version:** 2.1.0
**Status:** Research instrumentation (vision Model I/O extension)

---

## Purpose

Extend the existing `model_io` channel with **vision metadata** for controlled image/screenshot experiments via LM Studio's `/api/v1/chat` image `data_url` content items.

This layer records **provenance and hashes**, not subjective perception or engine state.

---

## Scientific separation

| Field | Channel | Notes |
|-------|---------|-------|
| `visual_description` | behavioral | What the model said it sees |
| `prediction` / `confidence` | behavioral | Task outputs |
| `self_report` | self_report | Introspective claims |
| `engine_state` | engine_state | External YAML engine only |
| Image hash/dimensions | model_io.vision | Provenance metadata |

Never map `"I can see the object moved"` into `engine_state`.

---

## Image provenance

Each image receives:

- `image_id` (UUID v4)
- `sha256` content hash
- MIME type, width, height, byte size
- `source` (allowlisted relative path)
- `trial_id`, `stage`, `timestamp`, optional `request_id`

Stored append-only in `research/raw/image_provenance.jsonl`.

Duplicate bytes across trials are linked via `duplicate_of_image_id`.

**Raw screenshots are not persisted by default.** Enable only with:

```powershell
$env:AI_ECO_LAB_PERSIST_IMAGE_ARTIFACTS = "1"
```

Artifacts (if enabled) go to `research/stimuli/_artifacts/` (local, gitignored).

---

## Stimulus allowlist

Images load only from `research/stimuli/` via manifest keys:

[`research/stimuli/visual_box_task_v1/manifest.yaml`](../research/stimuli/visual_box_task_v1/manifest.yaml)

Never pass arbitrary filesystem paths to MCP tools.

---

## Visual stages

- `visual_baseline`
- `visual_prediction`
- `visual_contradiction`
- `visual_revision`
- `visual_self_observation`

---

## Visual conditions

`identical_image`, `relevant_change`, `irrelevant_change`, `contradiction`, `reordered_visual_evidence`, `degraded_image`, `ambiguous_image`

---

## MCP tools (env-gated)

Set `AI_ECO_LAB_VISION_ENABLED=1` in LM Studio MCP env.

| Tool | Purpose |
|------|---------|
| `register_image_provenance` | Register manifest stimulus by key |
| `build_vision_chat_request` | Build read-only LM Studio chat payload |

No HTTP calls are made from the MCP server. No raw bytes returned in tool responses.

---

## Run visual simulation pilot

```powershell
cd C:\Users\John\ai-eco-lab
python -m research.run_visual_trial `
  --protocol research/protocols/visual_prediction_v1.yaml `
  --experiment-id exp-visual-pilot-001 `
  --mode simulation `
  --analyze
```

Protocol: [`research/protocols/visual_prediction_v1.yaml`](../research/protocols/visual_prediction_v1.yaml)

---

## Security

- Local-only image handling
- Allowlisted stimulus root (`AI_ECO_LAB_STIMULI_ROOT`)
- No automatic upload
- No integration with public Neurovorex systems

---

## Future Phase 2

- Optional local HTTP adapter for LM Studio `/api/v1/chat`
- Post-hoc chat export correlation via `request_id` / `image_id`
