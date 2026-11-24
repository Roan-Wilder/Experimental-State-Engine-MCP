from __future__ import annotations

from pathlib import Path
from typing import Any


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_stage_tables(summary: dict[str, Any]) -> str:
    """Markdown tables for stability, conflict, uncertainty across stages."""
    trials = summary.get("trials", [])
    if not trials:
        return "_No trials available._\n"

    focus_dims = ["stability", "conflict", "uncertainty"]
    sections: list[str] = []

    for dimension in focus_dims:
        rows: list[list[str]] = []
        for trial in trials:
            deltas = trial.get("stage_deltas", {}).get("STATE_1->STATE_2", {}).get("core", [])
            match = next((d for d in deltas if d.get("dimension") == dimension), None)
            if match:
                rows.append(
                    [
                        trial.get("trial_id", "")[:8],
                        trial.get("trial_condition", ""),
                        f"{match.get('before', 0):.3f}",
                        f"{match.get('after', 0):.3f}",
                        f"{match.get('absolute_delta', 0):+.3f}",
                    ]
                )
        if rows:
            sections.append(f"### {dimension} (STATE_1 → STATE_2)\n")
            sections.append(
                markdown_table(
                    ["Trial", "Condition", "Before", "After", "Delta"],
                    rows,
                )
            )
            sections.append("")
    return "\n".join(sections) if sections else "_No stage delta data._\n"


def generate_condition_comparison_table(comparison: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for dimension in ("stability", "conflict", "uncertainty"):
        stats = comparison.get("core", {}).get(dimension, {})
        neutral = stats.get("neutral", {})
        contradiction = stats.get("contradiction", {})
        rows.append(
            [
                dimension,
                _fmt(neutral.get("mean_delta")),
                _fmt(contradiction.get("mean_delta")),
                str(neutral.get("trial_count", 0)),
                str(contradiction.get("trial_count", 0)),
            ]
        )
    return markdown_table(
        ["Dimension", "Neutral mean Δ", "Contradiction mean Δ", "N neutral", "N contradiction"],
        rows,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.4f}"
