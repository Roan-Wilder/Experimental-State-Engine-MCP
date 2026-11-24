from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.analyze import analyze_experiment
from research.report import generate_report
from research.trial_runner import TrialRunner


@pytest.fixture
def analyzed_run(tmp_path: Path):
    raw_log = tmp_path / "raw" / "trials.jsonl"
    analyzed_dir = tmp_path / "analyzed"
    reports_dir = tmp_path / "reports"
    experiment_id = "test-analysis-001"

    runner = TrialRunner(
        experiment_id=experiment_id,
        raw_log_path=raw_log,
        master_seed=42,
        trials_per_condition=2,
        mode="simulation",
    )
    runner.run_batch()
    raw_before = raw_log.read_bytes()

    summary = analyze_experiment(
        experiment_id,
        raw_path=raw_log,
        analyzed_dir=analyzed_dir,
    )
    report_path = generate_report(summary, reports_dir=reports_dir)

    raw_after = raw_log.read_bytes()
    return {
        "raw_log": raw_log,
        "raw_before": raw_before,
        "raw_after": raw_after,
        "summary": summary,
        "analyzed_dir": analyzed_dir,
        "report_path": report_path,
        "experiment_id": experiment_id,
    }


def test_raw_data_untouched_by_analysis(analyzed_run):
    assert analyzed_run["raw_before"] == analyzed_run["raw_after"]


def test_analyzed_output_marked_derived(analyzed_run):
    assert analyzed_run["summary"]["derived"] is True
    summary_path = analyzed_run["analyzed_dir"] / f"{analyzed_run['experiment_id']}_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["derived"] is True


def test_comparison_stats_present(analyzed_run):
    comparison = analyzed_run["summary"]["neutral_vs_contradiction"]
    assert "core" in comparison
    assert "disclaimer" in comparison
    stability = comparison["core"].get("stability", {})
    assert "neutral" in stability
    assert "contradiction" in stability


def test_report_generated(analyzed_run):
    assert analyzed_run["report_path"].exists()
    content = analyzed_run["report_path"].read_text(encoding="utf-8")
    assert "Observed" in content
    assert "consciousness" not in content.lower() or "does not claim consciousness" in content.lower()
