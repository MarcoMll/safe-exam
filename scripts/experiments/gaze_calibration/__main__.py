"""CLI for gaze off-screen threshold calibration (issue #13).

Workflow:
  1. Start the script and enter an experiment name (or pass --experiment).
  2. A live preview opens with face-gaze overlay and angle readout.
  3. Type a scenario name in the terminal (natural_* or suspicious_*).
  4. Press SPACE — hold the behavior for --duration seconds (default 45).
  5. Repeat for as many scenarios as you want; press Q to quit.

Results append to:
docs/experiments/gaze-calibration/results/<experiment>/gaze_calibration.csv.

Use --summarize to evaluate duration thresholds offline (no camera).
Use --backtest to sweep angle/duration configs against a recorded CSV.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from safe_exam.utils.logging_utils import configure_logging
from safe_exam.utils.paths_initializer import get_paths, verify_paths

from .analyze import (
    AnalysisConfig,
    AngleThresholds,
    default_output_path,
    parse_float_list,
    print_csv_summary,
    run_backtest,
)

configure_logging()
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record gaze angles for off-screen duration threshold calibration."
        ),
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
        "--duration",
        type=float,
        default=45.0,
        help="Seconds to record per scenario (default: 45).",
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
        "--pitch-threshold",
        type=float,
        default=10.0,
        help="Pitch degrees for off-center (default: 10).",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=10.0,
        help="Yaw degrees for off-center (default: 10).",
    )
    parser.add_argument(
        "--duration-thresholds",
        default="4,6,8,12",
        help="Comma-separated duration thresholds in seconds (default: 4,6,8,12).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV path (default: docs/experiments/gaze-calibration/results/"
            "<experiment>/gaze_calibration.csv)."
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run headless — no live window (not recommended for manual testing).",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "yaw_only", "pitch_only"],
        default="both",
        help="Off-center rule: both axes, yaw only, or pitch only (default: both).",
    )
    parser.add_argument(
        "--iris-threshold",
        type=float,
        help="Iris offset threshold for signal=iris (default 0.12 in summaries).",
    )
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=0.0,
        help="Seconds of face-loss to tolerate inside a streak (default: 0).",
    )
    parser.add_argument(
        "--summarize",
        nargs="?",
        const="default",
        metavar="CSV",
        help=(
            "Print duration-threshold tables from a results CSV without capturing. "
            "Omit the path to summarize experiment_1 if present."
        ),
    )
    parser.add_argument(
        "--backtest",
        nargs="?",
        const="default",
        metavar="CSV",
        help=(
            "Sweep angle/duration configs on a recorded CSV (no camera). "
            "Writes backtest_grid.csv next to the source CSV when used."
        ),
    )
    parser.add_argument(
        "--backtest-top",
        type=int,
        default=15,
        help="Number of top backtest configs to print (default: 15).",
    )
    return parser


def resolve_csv_path(raw: str) -> Path:
    if raw != "default":
        return Path(raw)
    paths = get_paths()
    return (
        paths.GAZE_CALIBRATION_RESULTS_DIR
        / "experiment_1_desktop_pc_camera"
        / "gaze_calibration.csv"
    )


def main() -> int:
    verify_paths()
    parser = build_parser()
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive.")

    duration_thresholds = parse_float_list(args.duration_thresholds)

    if args.backtest is not None:
        csv_path = resolve_csv_path(args.backtest)
        if not csv_path.is_file():
            parser.error(f"CSV not found: {csv_path}")
        gap = args.gap_tolerance if args.gap_tolerance > 0 else 0.4
        run_backtest(
            csv_path,
            gap_tolerance_s=gap,
            top_n=args.backtest_top,
            output_path=csv_path.parent / "backtest_grid.csv",
        )
        return 0

    if args.summarize is not None:
        csv_path = resolve_csv_path(args.summarize)
        if not csv_path.is_file():
            parser.error(f"CSV not found: {csv_path}")
        analysis = AnalysisConfig(
            signal="gaze",
            angles=AngleThresholds(
                pitch_deg=args.pitch_threshold,
                yaw_deg=args.yaw_threshold,
                mode=args.mode,  # type: ignore[arg-type]
            ),
            iris_offset_thr=args.iris_threshold,
            gap_tolerance_s=args.gap_tolerance,
        )
        print_csv_summary(
            csv_path,
            config=analysis,
            duration_thresholds=duration_thresholds,
        )
        return 0

    try:
        from .record import prompt_experiment, run_calibration_session

        experiment = prompt_experiment(args.experiment)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    output_path = args.output or default_output_path(experiment)
    from safe_exam.capture.capture_config import CaptureConfig
    from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector

    face_gaze_config = FaceGazeConfig(
        draw_landmarks=True,
        mirror_preview=False,
        refine_landmarks=True,
        yaw_threshold_deg=args.yaw_threshold,
        pitch_threshold_deg=args.pitch_threshold,
    )
    detector = FaceGazeDetector(config=face_gaze_config)
    capture_config = CaptureConfig(
        camera_index=args.camera_index,
        target_fps=args.target_fps,
    )

    try:
        completed = run_calibration_session(
            detector=detector,
            capture_config=capture_config,
            experiment=experiment,
            initial_label=args.label,
            duration_s=args.duration,
            pitch_thr=args.pitch_threshold,
            yaw_thr=args.yaw_threshold,
            duration_thresholds=duration_thresholds,
            output_path=output_path,
            show_preview=not args.no_preview,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        completed = []

    if completed:
        print(f"\nRecorded scenarios ({len(completed)}): {', '.join(completed)}")
        print(
            "Summarize with: "
            "python -m experiments.gaze_calibration "
            f"--summarize {output_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
