from __future__ import annotations

import argparse
from pathlib import Path

from research.analyze import analyze_experiment
from research.report import generate_report
from research.visual_trial_runner import VisualTrialRunner

DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "protocols" / "visual_prediction_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Eco Lab visual trial runner")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--experiment-id", type=str, default="exp-visual-pilot-001")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation"])
    parser.add_argument("--raw-log", type=Path, default=None)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    raw_path = args.raw_log or Path(__file__).resolve().parent / "raw" / "trials.jsonl"
    runner = VisualTrialRunner(
        experiment_id=args.experiment_id,
        protocol_path=args.protocol,
        raw_log_path=raw_path,
        mode=args.mode,
    )
    completed = runner.run_batch()
    print(f"Completed {len(completed)} visual trials -> {raw_path}")

    if args.analyze:
        summary = analyze_experiment(
            args.experiment_id,
            raw_path=raw_path,
            protocol_path=args.protocol,
            include_observations=True,
            include_image_provenance=True,
        )
        report_path = generate_report(summary, protocol_path=args.protocol)
        print(f"Analysis -> research/analyzed/{args.experiment_id}_summary.json")
        print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
