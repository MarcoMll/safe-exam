"""Live recording session for gaze calibration."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector
from safe_exam.detectors.face_gaze.overlay import draw_face_gaze_on_frame

from .analyze import (
    FrameSample,
    _config_for_signal,
    _legacy_analysis_config,
    append_rows,
    is_off_center,
    max_off_center_streak_s,
    off_center_fraction,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "gaze calibration"


@dataclass
class SessionState:
    label: str = ""
    awaiting_label: bool = True
    ready_for_input: bool = False
    recording: bool = False
    quitting: bool = False
    record_started_at: float | None = None
    samples: list[FrameSample] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _draw_panel(
    frame, x: int, y: int, lines: list[str], *, accent: tuple[int, int, int]
):
    import cv2  # pylint: disable=import-outside-toplevel

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    line_height = 22
    padding = 10
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    panel_w = max(widths) + padding * 2
    panel_h = len(lines) * line_height + padding * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), accent, 2)
    text_y = y + padding + 16
    for line in lines:
        cv2.putText(frame, line, (x + padding, text_y), font, scale, accent, thickness)
        text_y += line_height


def draw_calibration_overlay(
    frame,
    result,
    config: FaceGazeConfig,
    *,
    experiment: str,
    label: str,
    awaiting_label: bool,
    recording: bool,
    elapsed_s: float,
    duration_s: float,
    pitch_thr: float,
    yaw_thr: float,
):
    import cv2  # pylint: disable=import-outside-toplevel

    display = draw_face_gaze_on_frame(frame, result, config)
    height, width = display.shape[:2]

    gaze_off = is_off_center(result.gaze_pitch, result.gaze_yaw, pitch_thr, yaw_thr)
    eye_off = is_off_center(result.eye_pitch, result.eye_yaw, pitch_thr, yaw_thr)
    head_off = is_off_center(result.head_pitch, result.head_yaw, pitch_thr, yaw_thr)

    if recording:
        cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)

    accent = (0, 0, 255) if recording else (0, 255, 0)
    scenario = label if label else "(type scenario name in terminal)"
    if awaiting_label:
        scenario = "(waiting for scenario name in terminal)"

    remaining = max(0.0, duration_s - elapsed_s) if recording else duration_s
    lines = [
        f"Experiment: {experiment}",
        f"Scenario: {scenario}",
        f"Status: {'RECORDING' if recording else 'READY'}",
        f"Head p/y: {result.head_pitch:+.1f}/{result.head_yaw:+.1f}"
        f"  ({'OFF' if head_off else 'ok'})",
        f"Eye  p/y: {result.eye_pitch:+.1f}/{result.eye_yaw:+.1f}"
        f"  ({'OFF' if eye_off else 'ok'})",
        f"Gaze p/y: {result.gaze_pitch:+.1f}/{result.gaze_yaw:+.1f}"
        f"  ({'OFF' if gaze_off else 'ok'})",
        f"Angle thr: p={pitch_thr:.0f} y={yaw_thr:.0f}",
    ]
    if recording:
        lines.append(
            f"Elapsed {elapsed_s:.1f}s / {duration_s:.0f}s  ({remaining:.1f}s left)"
        )
    elif awaiting_label:
        lines.append("Preview live — type scenario name below")
    else:
        lines.append("SPACE = record   N = rename   Q = quit")

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
        state.record_started_at = None
        state.samples.clear()
    print("\nEnter a new scenario name in the terminal.")


def finalize_capture(
    *,
    state: SessionState,
    experiment: str,
    output_path: Path,
    pitch_thr: float,
    yaw_thr: float,
    duration_thresholds: list[float],
) -> str | None:
    with state.lock:
        label = state.label
        samples = list(state.samples)
        state.recording = False
        state.record_started_at = None
        state.samples.clear()
        state.awaiting_label = True

    if not samples:
        return None

    recorded_at = datetime.now(timezone.utc).isoformat()
    append_rows(
        output_path,
        experiment=experiment,
        label=label,
        samples=samples,
        recorded_at=recorded_at,
    )

    print(f"\n=== {label} ===")
    print(f"  frames: {len(samples)}")
    print(f"  duration: {samples[-1].elapsed_s:.1f}s")
    for signal in ("gaze", "eye", "head", "iris"):
        run_config = _config_for_signal(
            _legacy_analysis_config(
                signal=signal,
                pitch_thr=pitch_thr,
                yaw_thr=yaw_thr,
            ),
            signal,
        )
        max_streak = max_off_center_streak_s(samples, run_config)
        off_frac = off_center_fraction(samples, run_config)
        fires = ", ".join(
            f"@{d:.0f}s={'Y' if max_streak >= d else 'n'}" for d in duration_thresholds
        )
        print(f"  {signal}: off={off_frac:.0%} max_streak={max_streak:.2f}s  {fires}")
    print(f"Appended {len(samples)} rows to {output_path}")
    print("Enter the next scenario name in the terminal.")
    return label


def run_calibration_session(
    *,
    detector: FaceGazeDetector,
    capture_config: CaptureConfig,
    experiment: str,
    initial_label: str | None,
    duration_s: float,
    pitch_thr: float,
    yaw_thr: float,
    duration_thresholds: list[float],
    output_path: Path,
    show_preview: bool,
) -> list[str]:
    import cv2  # pylint: disable=import-outside-toplevel

    completed: list[str] = []
    state = SessionState()

    if initial_label:
        state.label = initial_label
        state.awaiting_label = False

    print("\nGaze off-screen threshold calibration")
    print(f"Experiment: {experiment}")
    print(
        f"Each capture records ~{duration_s:.0f}s "
        f"at {capture_config.target_fps:.0f} FPS."
    )
    print(f"CSV output: {output_path}")
    print("Name scenarios natural_* or suspicious_* (see docs).")
    if show_preview:
        print("\nLive preview opening — watch head/eye/gaze OFF indicators.")
        print(
            "Controls (click the preview window first): "
            "SPACE = record, N = rename, Q = quit"
        )
    if initial_label:
        print(f"Scenario: {initial_label} — press SPACE to record.")
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

            result = detector.detect(frame)
            now = time.perf_counter()
            finish_capture = False

            with state.lock:
                elapsed = 0.0
                if state.recording and state.record_started_at is not None:
                    elapsed = now - state.record_started_at
                    state.samples.append(
                        FrameSample(
                            elapsed_s=elapsed,
                            face_detected=result.face_detected,
                            head_pitch=result.head_pitch,
                            head_yaw=result.head_yaw,
                            eye_pitch=result.eye_pitch,
                            eye_yaw=result.eye_yaw,
                            gaze_pitch=result.gaze_pitch,
                            gaze_yaw=result.gaze_yaw,
                            iris_offset_x=result.iris_offset_x,
                            iris_offset_y=result.iris_offset_y,
                        )
                    )
                    if elapsed >= duration_s:
                        state.recording = False
                        finish_capture = True

                snapshot = (
                    state.label,
                    state.awaiting_label,
                    state.recording,
                    elapsed,
                    state.quitting,
                )

            label, awaiting_label, recording, elapsed_s, quitting = snapshot
            if quitting:
                break

            if finish_capture:
                done = finalize_capture(
                    state=state,
                    experiment=experiment,
                    output_path=output_path,
                    pitch_thr=pitch_thr,
                    yaw_thr=yaw_thr,
                    duration_thresholds=duration_thresholds,
                )
                if done:
                    completed.append(done)
                if initial_label:
                    break

            if show_preview:
                display = draw_calibration_overlay(
                    frame,
                    result,
                    detector.config,
                    experiment=experiment,
                    label=label,
                    awaiting_label=awaiting_label,
                    recording=recording,
                    elapsed_s=elapsed_s,
                    duration_s=duration_s,
                    pitch_thr=pitch_thr,
                    yaw_thr=yaw_thr,
                )
                cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF if show_preview else 0xFF

            if key == ord(" "):
                with state.lock:
                    if state.awaiting_label or not state.label:
                        print("Set a scenario name in the terminal before recording.")
                    elif not state.recording:
                        state.recording = True
                        state.record_started_at = time.perf_counter()
                        state.samples.clear()
                        logger.info(
                            "Recording started | experiment=%s label=%s duration=%.0fs",
                            experiment,
                            state.label,
                            duration_s,
                        )
            elif key == ord("n"):
                request_new_label(state)
            elif key == ord("q"):
                with state.lock:
                    state.quitting = True
                    state.recording = False
                    state.samples.clear()
                break
    finally:
        with state.lock:
            state.quitting = True
        frame_iter.close()
        detector.close()
        if show_preview:
            cv2.destroyAllWindows()

    return completed
