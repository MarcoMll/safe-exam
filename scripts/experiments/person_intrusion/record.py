"""Live recording session for person intrusion calibration."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.object import ObjectDetector, ObjectDetectorConfig
from safe_exam.processor.frame_result import FrameResult
from safe_exam.processor.intrusion_policy import (
    DEFAULT_INTRUSION_POLICY,
    intrusion_features_for_frame,
)

from .analyze import (
    ScenarioSummary,
    append_rows,
    boxes_to_json,
    format_summary,
    summarize_scenario,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "person intrusion calibration"


@dataclass
class SessionState:
    label: str = ""
    awaiting_label: bool = True
    ready_for_input: bool = False
    recording: bool = False
    quitting: bool = False
    rows: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


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
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
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
    person_count: int,
    intrusion_suspected: bool,
    recording: bool,
    frames_done: int,
    frames_total: int,
) -> np.ndarray:
    import cv2  # pylint: disable=import-outside-toplevel

    display = frame.copy()
    height, width = display.shape[:2]

    if recording:
        cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)

    status = "RECORDING" if recording else "READY"
    accent = (0, 0, 255) if recording else (0, 255, 0)
    scenario_line = label if label else "(type scenario name in terminal)"
    if awaiting_label:
        scenario_line = "(waiting for scenario name in terminal)"

    lines = [
        f"Experiment: {experiment}",
        f"Scenario: {scenario_line}",
        f"Status: {status}",
        f"Person count: {person_count}",
        f"Intrusion @ default policy: {'YES' if intrusion_suspected else 'no'}",
    ]

    if recording:
        lines.append(f"Frame {frames_done}/{frames_total}")
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
        state.rows.clear()
    print("\nEnter a new scenario name in the terminal.")


def _frame_row(
    *,
    frame_result: FrameResult,
) -> dict[str, str | int | float | bool]:
    features = intrusion_features_for_frame(frame_result, DEFAULT_INTRUSION_POLICY)
    return {
        "frame_width": frame_result.frame_width,
        "frame_height": frame_result.frame_height,
        "person_count": features.person_count,
        "primary_area_pct": f"{features.primary_area_pct:.6f}",
        "max_secondary_area_pct": f"{features.max_secondary_area_pct:.6f}",
        "max_secondary_iou": f"{features.max_secondary_iou:.6f}",
        "any_secondary_in_roi": int(features.any_secondary_in_roi),
        "intrusion_suspected": int(features.intrusion_suspected),
        "person_boxes_json": boxes_to_json(frame_result.person_boxes),
    }


def finalize_capture(
    *,
    state: SessionState,
    experiment: str,
    output_path: Path,
) -> ScenarioSummary | None:
    with state.lock:
        label = state.label
        rows = list(state.rows)
        state.recording = False
        state.rows.clear()
        state.awaiting_label = True

    if not rows:
        return None

    recorded_at = datetime.now(timezone.utc).isoformat()
    append_rows(
        output_path,
        experiment=experiment,
        label=label,
        rows=rows,
        recorded_at=recorded_at,
    )
    summary = summarize_scenario(label, rows)
    print(format_summary(summary))
    print(f"Appended {len(rows)} rows to {output_path}")
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
    output_path: Path,
    show_preview: bool,
) -> list[ScenarioSummary]:
    import cv2  # pylint: disable=import-outside-toplevel

    summaries: list[ScenarioSummary] = []
    state = SessionState()

    if initial_label:
        state.label = initial_label
        state.awaiting_label = False

    print("\nPerson intrusion calibration")
    print(f"Experiment: {experiment}")
    print(
        f"Each capture records {frame_count} frames "
        f"(~{frame_count / capture_config.target_fps:.1f}s at "
        f"{capture_config.target_fps:.0f} FPS)."
    )
    print(f"CSV output: {output_path}")
    print("\nUse scenario prefixes:")
    print("  solo_*        — only the seated student")
    print("  background_*  — classmates visible but not intrusive")
    print("  intrusion_*   — lean-in / close overlap (should flag)")
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
    elif not initial_label:
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

            height, width = frame.shape[:2]
            yolo_results = detector.detect(
                frame=frame,
                classes=[config.person_class_id],
            )
            person_boxes = detector.list_boxes(
                results=yolo_results,
                target_class_index=config.person_class_id,
                threshold=config.confidence_threshold,
            )
            frame_result = FrameResult(
                person_count=len(person_boxes),
                person_boxes=person_boxes,
                frame_width=width,
                frame_height=height,
            )
            features = intrusion_features_for_frame(
                frame_result,
                DEFAULT_INTRUSION_POLICY,
            )

            finish_capture = False
            with state.lock:
                if state.recording:
                    state.rows.append(_frame_row(frame_result=frame_result))
                    if len(state.rows) >= frame_count:
                        state.recording = False
                        finish_capture = True

                snapshot = (
                    state.label,
                    state.awaiting_label,
                    state.recording,
                    len(state.rows),
                    state.quitting,
                )

            label, awaiting_label, recording, frames_done, quitting = snapshot
            if quitting:
                break

            if finish_capture:
                summary = finalize_capture(
                    state=state,
                    experiment=experiment,
                    output_path=output_path,
                )
                if summary:
                    summaries.append(summary)
                if initial_label:
                    break

            if show_preview:
                base = yolo_results[0].plot()
                display = draw_calibration_overlay(
                    base,
                    experiment=experiment,
                    label=label,
                    awaiting_label=awaiting_label,
                    person_count=features.person_count,
                    intrusion_suspected=features.intrusion_suspected,
                    recording=recording,
                    frames_done=frames_done,
                    frames_total=frame_count,
                )
                cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF if show_preview else 0xFF

            if key == ord(" "):
                with state.lock:
                    if state.awaiting_label or not state.label:
                        print("Set a scenario name in the terminal before recording.")
                    elif not state.recording:
                        state.recording = True
                        state.rows.clear()
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
                    state.rows.clear()
                break
    finally:
        with state.lock:
            state.quitting = True
        frame_iter.close()
        if show_preview:
            cv2.destroyAllWindows()

    return summaries
