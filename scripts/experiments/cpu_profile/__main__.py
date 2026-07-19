"""CLI for CPU/RAM profiling of the frame processor (issue #15).

Examples (from the scripts/ directory):

  # Cost drill-down: object vs face_gaze vs both (~30s each)
  python -m experiments.cpu_profile --experiment desk_pc --mode all \\
      --target-fps 5 --duration 30

  # Sustained production-like run (10 min)
  python -m experiments.cpu_profile --experiment desk_pc --mode both \\
      --target-fps 5 --duration 600

Modes:
  object     — YOLO only
  face_gaze  — MediaPipe only
  both       — full process_frame (YOLO + MediaPipe)
  all        — object, then face_gaze, then both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safe_exam.utils.logging_utils import configure_logging
from safe_exam.utils.paths_initializer import verify_paths

from .profile import ProfileMode, default_output_path, run_cpu_profile, run_mode_suite

configure_logging()

MODE_CHOICES = ("object", "face_gaze", "both", "all")
ALL_MODES: list[ProfileMode] = ["object", "face_gaze", "both"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile CPU/RAM for object detection, face gaze, or both (issue #15)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tip: use --mode all --duration 30 for a quick cost split; "
            "use --mode both --duration 600 for a sustained go/no-go run.\n"
            "Findings: docs/experiments/cpu-profiling/README.md"
        ),
    )
    parser.add_argument(
        "--experiment",
        default="experiment_1",
        help="Experiment name for the results folder (default: experiment_1).",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="both",
        help="object | face_gaze | both | all (default: both).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5.0,
        help="Target capture frame rate (default: 5).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Seconds to measure after warmup (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=15.0,
        help="Seconds of warmup before measuring (default: 15).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV path (default: docs/experiments/cpu-profiling/results/"
            "<experiment>/cpu_profile.csv)."
        ),
    )
    return parser


def main() -> int:
    verify_paths()
    parser = build_parser()
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be positive.")
    if args.target_fps <= 0:
        parser.error("--target-fps must be positive.")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0.")

    experiment = args.experiment.strip() or "experiment_1"
    output_path = args.output or default_output_path(experiment)

    print(f"Experiment: {experiment}")
    print(f"CSV output: {output_path}")
    print("Headless run (no debug overlay).")

    try:
        if args.mode == "all":
            run_mode_suite(
                modes=ALL_MODES,
                target_fps=args.target_fps,
                duration_s=args.duration,
                warmup_s=args.warmup,
                camera_index=args.camera_index,
                experiment=experiment,
                output_path=output_path,
            )
        else:
            run_cpu_profile(
                mode=args.mode,  # type: ignore[arg-type]
                target_fps=args.target_fps,
                duration_s=args.duration,
                warmup_s=args.warmup,
                camera_index=args.camera_index,
                experiment=experiment,
                output_path=output_path,
            )
    except KeyboardInterrupt:
        print("\nStopped (partial run not saved unless a mode finished).")
        return 0

    print("\nDone. See docs/experiments/cpu-profiling/README.md for interpretation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
