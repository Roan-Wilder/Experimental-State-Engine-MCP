from __future__ import annotations

from pathlib import Path
from typing import Any

from research.protocol import load_protocol
from research.visualize import generate_condition_comparison_table, generate_stage_tables

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS = PROJECT_ROOT / "reports"


def generate_report(
    summary: dict[str, Any],
    *,
    reports_dir: Path | None = None,
    protocol_path: Path | None = None,
) -> Path:
    protocol = load_protocol(protocol_path)
    reports_dir = reports_dir or DEFAULT_REPORTS
    reports_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = summary.get("experiment_id", "unknown")
    comparison = summary.get("neutral_vs_contradiction", {})
    trials = summary.get("trials", [])

    revisions = sum(1 for t in trials if t.get("revision_occurred"))
    maintained = sum(1 for t in trials if t.get("prediction_maintained"))
    tool_failures = sum(len(t.get("tool_errors", [])) for t in trials)

    content = f"""# Experiment Report: {experiment_id}

## 1. Experiment title

{protocol.get('experiment_name', experiment_id)}

## 2. Research question

{protocol.get('research_question', '').strip()}

## 3. Hypothesis

**Hypothesized:** {protocol.get('hypothesis', '').strip()}

## 4. Experimental design

Offline simulation trials with controller-defined conditions (NEUTRAL, CONTRADICTION, RESOLUTION).
Predictions recorded before information exposure. State observations captured at STATE_0–STATE_4.

## 5. Model

{protocol.get('model_identifier', 'unknown')} (simulation mode in this run)

## 6. Conditions

| Condition | Trials |
|-----------|--------|
| NEUTRAL | {summary.get('trials_by_condition', {}).get('NEUTRAL', 0)} |
| CONTRADICTION | {summary.get('trials_by_condition', {}).get('CONTRADICTION', 0)} |
| RESOLUTION | {summary.get('trials_by_condition', {}).get('RESOLUTION', 0)} |

## 7. Number of trials

**Observed:** {summary.get('trial_count', 0)} completed trials.

## 8. Randomization

**Observed:** Per-trial seeds derived from master seed via SHA-256. Object identity and box locations randomized.

## 9. State variables

**Observed:** 12 core dimensions + 12 affective observation variables (externally represented experimental state).

## 10. Results

**Observed:** All trials produced append-only raw records in `research/raw/`.

## 11. State-transition statistics

### Neutral vs contradiction (STATE_1 → STATE_2)

{generate_condition_comparison_table(comparison)}

{generate_stage_tables(summary)}

## 12. Prediction revision statistics

| Metric | Count |
|--------|-------|
| Revisions occurred | {revisions} |
| Predictions maintained | {maintained} |
| Total trials | {len(trials)} |

**Observed:** Revision detected from structured initial vs final prediction fields only.

## 13. Tool failures

**Observed:** {tool_failures} tool error record(s) across trials.

## 14. Limitations

- Simulation mode uses deterministic prediction stubs, not live Gemma output.
- State variables are externally represented; not neural activations or subjective emotions.
- No statistical significance testing performed.
- MCP tool logs in `data/experiments/` are separate from trial raw logs.

## 15. Alternative explanations

**Inferred:** State deltas may reflect configured YAML transitions rather than model-internal changes.
**Unknown:** Relationship between model behavior (B) and external state (A) in live sessions.

## 16. Reproducibility information

- Protocol version: {protocol.get('protocol_version')}
- Output schema: research/schemas/trial_record_v1.json
- Derived analysis labeled: `"derived": true`
- Raw data path: research/raw/trials.jsonl

---

*This report distinguishes Observed / Inferred / Hypothesized / Unknown. It does not claim consciousness, subjective emotion, or neural-state access.*
"""
    report_path = reports_dir / f"{experiment_id}_report.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path
