"""CLI for phone detection threshold calibration (issue #12).

Workflow:
  1. Start the script and enter an experiment name (or pass --experiment).
  2. A live preview opens with YOLO bounding boxes and confidence readout.
  3. Type a scenario name in the terminal while watching the preview.
  4. Press SPACE in the preview window to record ~30 frames (~3s at 12 FPS).
  5. Repeat for as many scenarios as you want; press Q in the preview to quit.

Results append to:
docs/experiments/phone-calibration/results/<experiment>/phone_calibration.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from safe_exam.utils.logging_utils import configure_logging
from safe_exam.utils.paths_initializer import get_paths, verify_paths

from .analyze import (
    DEFAULT_SUMMARY_THRESHOLDS,
    default_output_path,
    parse_thresholds,
    print_csv_summary,
)

configure_logging()
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Record phone detection confidences for threshold calibration."),
    )
    parser.add_argument(
        "--experiment",
        help="Experiment name — groups scenarios in the CSV and output folder.",
    )
    parser.add_argument(
        "--label",
        help=(
            "Scenario name for the first capture. "
            "Omit to name each scenario interactively."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames to record per scenario (default: 30, ~2.5s at 12 FPS).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=12.0,
        help="Capture frame rate (default: 12).",
    )
    parser.add_argument(
        "--thresholds",
        default="0.25,0.50",
        help=(
            "Comma-separated thresholds for detection-rate summary "
            "(default: 0.25,0.50)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV path (default: docs/experiments/phone-calibration/results/"
            "<experiment>/phone_calibration.csv)."
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run headless — no live window (not recommended for manual testing).",
    )
    parser.add_argument(
        "--summarize",
        nargs="?",
        const="default",
        metavar="CSV",
        help=(
            "Print summary tables from a results CSV without capturing. "
            "Omit the path to summarize the latest experiment_1 results "
            "(or pass an explicit CSV path)."
        ),
    )
    return parser


def resolve_summarize_path(raw: str) -> Path:
    if raw != "default":
        return Path(raw)
    paths = get_paths()
    return (
        paths.PHONE_CALIBRATION_RESULTS_DIR
        / "experiment_1_desktop_pc_camera"
        / "phone_calibration.csv"
    )


def main() -> int:
    verify_paths()
    parser = build_parser()
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames must be at least 1.")

    if args.summarize is not None:
        thresholds = (
            parse_thresholds(args.thresholds)
            if args.thresholds != "0.25,0.50"
            else DEFAULT_SUMMARY_THRESHOLDS
        )
        csv_path = resolve_summarize_path(args.summarize)
        if not csv_path.is_file():
            parser.error(f"CSV not found: {csv_path}")
        print_csv_summary(csv_path, thresholds)
        return 0

    thresholds = parse_thresholds(args.thresholds)

    try:
        from .record import prompt_experiment, run_calibration_session

        experiment = prompt_experiment(args.experiment)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    from safe_exam.capture.capture_config import CaptureConfig
    from safe_exam.detectors.object import ObjectDetector

    output_path = args.output or default_output_path(experiment)
    detector = ObjectDetector()
    capture_config = CaptureConfig(
        camera_index=args.camera_index,
        target_fps=args.target_fps,
    )

    try:
        summaries = run_calibration_session(
            detector=detector,
            config=detector.config,
            capture_config=capture_config,
            experiment=experiment,
            initial_label=args.label,
            frame_count=args.frames,
            thresholds=thresholds,
            output_path=output_path,
            show_preview=not args.no_preview,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        summaries = []

    if summaries:
        from .analyze import format_table

        print(f"\nExperiment summary: {experiment}")
        print(format_table(summaries, thresholds))
        print(
            "Summarize later with: "
            "python -m experiments.phone_calibration "
            f"--summarize {output_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
