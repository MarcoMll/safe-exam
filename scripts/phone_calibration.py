"""
Record phone-detection confidences for threshold calibration (issue #12).

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
import csv
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.object import ObjectDetector, ObjectDetectorConfig
from safe_exam.utils.logging_utils import configure_logging
from safe_exam.utils.paths_initializer import get_paths, verify_paths

configure_logging()
logger = logging.getLogger(__name__)

WINDOW_NAME = "phone calibration"

CSV_FIELDS = (
    "recorded_at",
    "experiment",
    "label",
    "frame_index",
    "phone_confidence",
)

DEFAULT_SUMMARY_THRESHOLDS = [0.25, 0.35, 0.45, 0.5, 0.55, 0.6]


@dataclass(frozen=True)
class ScenarioSummary:
    label: str
    frame_count: int
    mean_confidence: float
    max_confidence: float
    detection_rates: dict[float, float]


@dataclass
class SessionState:
    label: str = ""
    awaiting_label: bool = True
    ready_for_input: bool = False
    recording: bool = False
    quitting: bool = False
    confidences: list[float] = field(default_factory=list)
    capture_max_confidence: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def slugify_experiment(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    slug = slug.strip("_")
    return slug or "experiment"


def default_output_path(experiment: str) -> Path:
    paths = get_paths()
    return (
        paths.PHONE_CALIBRATION_RESULTS_DIR
        / slugify_experiment(experiment)
        / "phone_calibration.csv"
    )


def detection_rate(confidences: list[float], threshold: float) -> float:
    if not confidences:
        return 0.0
    hits = sum(1 for value in confidences if value >= threshold)
    return hits / len(confidences)


def summarize_scenario(
    label: str,
    confidences: list[float],
    thresholds: list[float],
) -> ScenarioSummary:
    return ScenarioSummary(
        label=label,
        frame_count=len(confidences),
        mean_confidence=sum(confidences) / len(confidences),
        max_confidence=max(confidences),
        detection_rates={
            threshold: detection_rate(confidences, threshold)
            for threshold in thresholds
        },
    )


def format_summary(summary: ScenarioSummary, thresholds: list[float]) -> str:
    rate_lines = [
        f"  detection_rate @ {threshold:.2f}: {summary.detection_rates[threshold]:.0%}"
        for threshold in thresholds
    ]
    return (
        f"=== {summary.label} ===\n"
        f"  frames: {summary.frame_count}\n"
        f"  mean_confidence: {summary.mean_confidence:.3f}\n"
        f"  max_confidence: {summary.max_confidence:.3f}\n" + "\n".join(rate_lines)
    )


def format_table(summaries: list[ScenarioSummary], thresholds: list[float]) -> str:
    label_width = max(len("label"), *(len(summary.label) for summary in summaries))
    threshold_headers = " ".join(f"@ {t:.2f}".rjust(8) for t in thresholds)
    header = (
        f"{'label':<{label_width}} {'mean':>6} {'max':>6} {'frames':>6} "
        f"{threshold_headers}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        rates = " ".join(f"{summary.detection_rates[t]:>7.0%}" for t in thresholds)
        lines.append(
            f"{summary.label:<{label_width}} "
            f"{summary.mean_confidence:>6.3f} "
            f"{summary.max_confidence:>6.3f} "
            f"{summary.frame_count:>6} "
            f"{rates}"
        )
    return "\n".join(lines)


def load_csv_summaries(
    csv_path: Path,
    thresholds: list[float],
) -> tuple[str | None, list[ScenarioSummary]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_label: dict[str, list[float]] = defaultdict(list)
    label_order: list[str] = []
    experiment: str | None = None
    for row in rows:
        label = row["label"]
        if label not in by_label:
            label_order.append(label)
        by_label[label].append(float(row["phone_confidence"]))
        if experiment is None:
            experiment = row.get("experiment") or None

    summaries = [
        summarize_scenario(label, by_label[label], thresholds) for label in label_order
    ]
    return experiment, summaries


def format_tradeoff_table(
    summaries: list[ScenarioSummary],
    thresholds: list[float],
) -> str:
    phone = [s for s in summaries if s.label.startswith("phone_")]
    nophone = [s for s in summaries if s.label.startswith("nophone_")]
    header = f"{'threshold':>10} {'FP':>8} {'TP':>8}"
    lines = [header, "-" * len(header)]
    for threshold in thresholds:
        if nophone:
            fp_hits = sum(
                round(s.detection_rates[threshold] * s.frame_count) for s in nophone
            )
            fp_total = sum(s.frame_count for s in nophone)
            fp = fp_hits / fp_total if fp_total else 0.0
        else:
            fp = 0.0
        if phone:
            tp_hits = sum(
                round(s.detection_rates[threshold] * s.frame_count) for s in phone
            )
            tp_total = sum(s.frame_count for s in phone)
            tp = tp_hits / tp_total if tp_total else 0.0
        else:
            tp = 0.0
        lines.append(f"{threshold:>10.2f} {fp:>7.1%} {tp:>7.1%}")
    return "\n".join(lines)


def print_csv_summary(csv_path: Path, thresholds: list[float]) -> None:
    experiment, summaries = load_csv_summaries(csv_path, thresholds)
    if not summaries:
        print(f"No rows found in {csv_path}")
        return

    title = experiment or csv_path.parent.name
    print(f"\nCSV summary: {title}")
    print(f"Source: {csv_path}")
    print(f"Scenarios: {len(summaries)}")
    print()
    print(format_table(summaries, thresholds))
    print()
    print("Threshold tradeoff (nophone_ = FP, phone_ = TP)")
    print(format_tradeoff_table(summaries, thresholds))


def append_rows(
    output_path: Path,
    *,
    experiment: str,
    label: str,
    confidences: list[float],
    recorded_at: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for index, confidence in enumerate(confidences, start=1):
            writer.writerow(
                {
                    "recorded_at": recorded_at,
                    "experiment": experiment,
                    "label": label,
                    "frame_index": index,
                    "phone_confidence": f"{confidence:.4f}",
                }
            )


def _draw_panel(
    frame, x: int, y: int, lines: list[str], *, accent: tuple[int, int, int]
):
    import cv2  # pylint: disable=import-outside-toplevel

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    line_height = 24
    padding = 10

    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    panel_w = max(widths) + padding * 2
    panel_h = len(lines) * line_height + padding * 2

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_w, y + panel_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), accent, 2)

    text_y = y + padding + 16
    for line in lines:
        cv2.putText(frame, line, (x + padding, text_y), font, scale, accent, thickness)
        text_y += line_height


def draw_calibration_overlay(
    frame,
    *,
    experiment: str,
    label: str,
    awaiting_label: bool,
    confidence: float,
    threshold: float,
    thresholds: list[float],
    recording: bool,
    frames_done: int,
    frames_total: int,
    capture_max_confidence: float,
) -> np.ndarray:
    import cv2  # pylint: disable=import-outside-toplevel

    display = frame.copy()
    height, width = display.shape[:2]

    if recording:
        cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)

    detected = confidence >= threshold
    status = "RECORDING" if recording else "READY"
    accent = (0, 0, 255) if recording else (0, 255, 0)

    scenario_line = label if label else "(type scenario name in terminal)"
    if awaiting_label:
        scenario_line = "(waiting for scenario name in terminal)"

    threshold_bits = "  ".join(
        f"{t:.2f}:{'Y' if confidence >= t else 'n'}" for t in thresholds
    )

    lines = [
        f"Experiment: {experiment}",
        f"Scenario: {scenario_line}",
        f"Status: {status}",
        f"Phone confidence: {confidence:.3f}",
        f"Current threshold {threshold:.2f}: {'DETECTED' if detected else 'clear'}",
        f"Check thresholds: {threshold_bits}",
    ]

    if recording:
        lines.append(f"Frame {frames_done}/{frames_total}")
        lines.append(f"Capture max so far: {capture_max_confidence:.3f}")
    elif awaiting_label:
        lines.append("Preview live — type scenario name below")
    elif label:
        lines.append("SPACE = record   N = rename scenario   Q = quit")
    else:
        lines.append("SPACE = record   Q = quit")

    _draw_panel(display, 10, 10, lines, accent=accent)
    return display


def prompt_non_empty(message: str) -> str:
    while True:
        try:
            raw = input(message).strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if raw:
            return raw
        print("Name cannot be empty — try again.")


def prompt_experiment(provided: str | None) -> str:
    if provided:
        return provided.strip()
    return prompt_non_empty("\nExperiment name: ")


def start_label_prompt_thread(state: SessionState) -> threading.Thread:
    def worker() -> None:
        while True:
            with state.lock:
                if state.quitting:
                    return
                should_prompt = state.awaiting_label and state.ready_for_input

            if not should_prompt:
                time.sleep(0.05)
                continue

            try:
                label = input("\nScenario name: ").strip()
            except EOFError:
                with state.lock:
                    state.quitting = True
                return

            if not label:
                print("Name cannot be empty — try again.")
                continue

            with state.lock:
                state.label = label
                state.awaiting_label = False
            print(
                f"Scenario set to '{label}'. "
                "Press SPACE in the preview window to record."
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def request_new_label(state: SessionState) -> None:
    with state.lock:
        state.label = ""
        state.awaiting_label = True
        state.recording = False
        state.confidences.clear()
        state.capture_max_confidence = 0.0
    print("\nEnter a new scenario name in the terminal.")


def finalize_capture(
    *,
    state: SessionState,
    experiment: str,
    frame_count: int,
    thresholds: list[float],
    output_path: Path,
) -> ScenarioSummary | None:
    with state.lock:
        label = state.label
        confidences = list(state.confidences)
        state.recording = False
        state.confidences.clear()
        state.capture_max_confidence = 0.0
        state.awaiting_label = True

    if not confidences:
        return None

    recorded_at = datetime.now(timezone.utc).isoformat()
    append_rows(
        output_path,
        experiment=experiment,
        label=label,
        confidences=confidences,
        recorded_at=recorded_at,
    )
    summary = summarize_scenario(label, confidences, thresholds)
    print(format_summary(summary, thresholds))
    print(f"Appended {len(confidences)} rows to {output_path}")
    print("Enter the next scenario name in the terminal.")
    return summary


def run_calibration_session(
    *,
    detector: ObjectDetector,
    config: ObjectDetectorConfig,
    capture_config: CaptureConfig,
    experiment: str,
    initial_label: str | None,
    frame_count: int,
    thresholds: list[float],
    output_path: Path,
    show_preview: bool,
) -> list[ScenarioSummary]:
    import cv2  # pylint: disable=import-outside-toplevel

    summaries: list[ScenarioSummary] = []
    state = SessionState()

    if initial_label:
        state.label = initial_label
        state.awaiting_label = False

    print("\nPhone threshold calibration")
    print(f"Experiment: {experiment}")
    print(
        f"Each capture records {frame_count} frames "
        f"(~{frame_count / capture_config.target_fps:.1f}s at "
        f"{capture_config.target_fps:.0f} FPS)."
    )
    print(f"CSV output: {output_path}")
    if show_preview:
        print("\nLive preview opening — watch the window while you test.")
        print(
            "Controls (click the preview window first): "
            "SPACE = record, N = rename, Q = quit"
        )
    if initial_label:
        print(
            f"Scenario: {initial_label} — press SPACE in the preview window to record."
        )
    else:
        print("Type the first scenario name in the terminal when prompted.")

    if not initial_label:
        with state.lock:
            state.ready_for_input = True
        start_label_prompt_thread(state)

    frame_iter = capture_frames(capture_config)

    try:
        while True:
            with state.lock:
                if state.quitting:
                    break

            try:
                frame = next(frame_iter)
            except StopIteration:
                break

            results = detector.detect(
                frame=frame,
                classes=[config.phone_class_id],
            )
            _, confidence = detector.look_for_class(
                results=results,
                target_class_index=config.phone_class_id,
                threshold=0.0,
            )

            finish_capture = False
            with state.lock:
                if state.recording:
                    state.confidences.append(confidence)
                    state.capture_max_confidence = max(
                        state.capture_max_confidence,
                        confidence,
                    )
                    if len(state.confidences) >= frame_count:
                        state.recording = False
                        finish_capture = True

                snapshot = (
                    state.label,
                    state.awaiting_label,
                    state.recording,
                    len(state.confidences),
                    state.capture_max_confidence,
                    state.quitting,
                )

            label, awaiting_label, recording, frames_done, capture_max, quitting = (
                snapshot
            )
            if quitting:
                break

            if finish_capture:
                summary = finalize_capture(
                    state=state,
                    experiment=experiment,
                    frame_count=frame_count,
                    thresholds=thresholds,
                    output_path=output_path,
                )
                if summary:
                    summaries.append(summary)
                if initial_label:
                    break

            if show_preview:
                base = results[0].plot()
                display = draw_calibration_overlay(
                    base,
                    experiment=experiment,
                    label=label,
                    awaiting_label=awaiting_label,
                    confidence=confidence,
                    threshold=config.confidence_threshold,
                    thresholds=thresholds,
                    recording=recording,
                    frames_done=frames_done,
                    frames_total=frame_count,
                    capture_max_confidence=capture_max,
                )
                cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF if show_preview else 0xFF

            if key == ord(" "):
                with state.lock:
                    if state.awaiting_label or not state.label:
                        print("Set a scenario name in the terminal before recording.")
                    elif not state.recording:
                        state.recording = True
                        state.confidences.clear()
                        state.capture_max_confidence = 0.0
                        logger.info(
                            "Recording started | experiment=%s label=%s frames=%s",
                            experiment,
                            state.label,
                            frame_count,
                        )
            elif key == ord("n"):
                request_new_label(state)
            elif key == ord("q"):
                with state.lock:
                    state.quitting = True
                    state.recording = False
                    state.confidences.clear()
                break
    finally:
        with state.lock:
            state.quitting = True
        frame_iter.close()
        if show_preview:
            cv2.destroyAllWindows()

    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record phone detection confidences for threshold calibration.",
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
        experiment = prompt_experiment(args.experiment)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    output_path = args.output or default_output_path(experiment)

    detector = ObjectDetector()
    capture_config = CaptureConfig(
        camera_index=args.camera_index,
        target_fps=args.target_fps,
    )

    summaries: list[ScenarioSummary] = []

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

    if summaries:
        print(f"\nExperiment summary: {experiment}")
        print(format_table(summaries, thresholds))

    return 0


if __name__ == "__main__":
    sys.exit(main())
