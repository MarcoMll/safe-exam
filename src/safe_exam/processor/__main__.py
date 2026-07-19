import argparse

from safe_exam.processor.runner import run_processor
from safe_exam.utils.logging_utils import configure_logging

configure_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the safe-exam frame processor")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable the composite debug overlay window",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=5.0,
        help=(
            "Target capture frame rate (default: 5; use 10 for denser "
            "local debug — see docs/experiments/cpu-profiling/)"
        ),
    )
    args = parser.parse_args()
    run_processor(
        debug=not args.no_debug,
        camera_index=args.camera_index,
        target_fps=args.target_fps,
    )
