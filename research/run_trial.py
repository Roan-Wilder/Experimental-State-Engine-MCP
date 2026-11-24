from __future__ import annotations

import argparse
from pathlib import Path

from research.analyze import analyze_experiment
from research.protocol import DEFAULT_PROTOCOL_PATH
from research.report import generate_report
from research.trial_runner import TrialRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Eco Lab trial runner")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Path to experiment protocol YAML",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="exp-pred-rev-001",
        help="Experiment identifier",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Master random seed for trial randomization",
    )
    parser.add_argument(
        "--trials-per-condition",
        type=int,
        default=3,
        help="Number of trials per condition",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="simulation",
        choices=["simulation"],
        help="Execution mode",
    )
    parser.add_argument(
        "--raw-log",
        type=Path,
        default=None,
        help="Override raw trial JSONL path",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis and generate report after trials",
    )
    args = parser.parse_args()

    raw_path = args.raw_log or Path(__file__).resolve().parent / "raw" / "trials.jsonl"
    runner = TrialRunner(
        experiment_id=args.experiment_id,
        protocol_path=args.protocol,
        raw_log_path=raw_path,
        master_seed=args.master_seed,
        trials_per_condition=args.trials_per_condition,
        mode=args.mode,
    )
    completed = runner.run_batch()
    print(f"Completed {len(completed)} trials -> {raw_path}")

    if args.analyze:
        summary = analyze_experiment(
            args.experiment_id,
            raw_path=raw_path,
            protocol_path=args.protocol,
        )
        report_path = generate_report(summary, protocol_path=args.protocol)
        print(f"Analysis -> research/analyzed/{args.experiment_id}_summary.json")
        print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
