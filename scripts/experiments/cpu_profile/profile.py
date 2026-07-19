"""CPU/RAM profiling for the unified processor (issue #15).

Runs the live capture loop headless and samples this process with psutil.
Supports detector modes so YOLO and MediaPipe can be measured separately.
"""

from __future__ import annotations

import csv
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import psutil

from safe_exam.capture.capture import capture_frames
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector
from safe_exam.detectors.object import ObjectDetector
from safe_exam.processor.frame_processor import process_frame

ProfileMode = Literal["object", "face_gaze", "both"]

CSV_FIELDS = (
    "recorded_at",
    "experiment",
    "mode",
    "target_fps",
    "duration_s",
    "warmup_s",
    "logical_cpu_count",
    "platform",
    "processor",
    "frame_count",
    "avg_fps",
    "avg_inference_ms",
    "avg_process_cpu_percent",
    "peak_process_cpu_percent",
    "avg_machine_cpu_percent",
    "peak_machine_cpu_percent",
    "avg_system_cpu_percent",
    "avg_ram_mb",
    "peak_ram_mb",
    "sample_count",
)

PROGRESS_EVERY_S = 30.0


def slugify_experiment(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    slug = slug.strip("_")
    return slug or "experiment"


def default_output_path(experiment: str) -> Path:
    from safe_exam.utils.paths_initializer import get_paths

    paths = get_paths()
    return (
        paths.CPU_PROFILING_RESULTS_DIR
        / slugify_experiment(experiment)
        / "cpu_profile.csv"
    )


def append_result_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(result: dict) -> None:
    target = float(result["target_fps"])
    actual = float(result["avg_fps"])
    machine = float(result["avg_machine_cpu_percent"])
    peak_machine = float(result["peak_machine_cpu_percent"])
    hit_target = actual >= target * 0.95

    print("\n=== CPU Profile ===")
    print(f"  mode:              {result['mode']}")
    print(f"  target_fps:        {target}")
    fps_note = "hit target" if hit_target else "below target"
    print(f"  avg_fps:           {actual}  ({fps_note})")
    print(f"  avg_inference_ms:  {result['avg_inference_ms']}")
    print(
        f"  machine_cpu %:     avg {machine}  peak {peak_machine}  "
        f"(process share of whole CPU; compare to <30%)"
    )
    print(
        f"  process_cpu %:     avg {result['avg_process_cpu_percent']}  "
        f"peak {result['peak_process_cpu_percent']}  "
        f"(100% ≈ one core; can exceed 100%)"
    )
    print(
        f"  system_cpu %:      avg {result['avg_system_cpu_percent']}  "
        f"(entire machine)"
    )
    print(
        f"  ram_mb:            avg {result['avg_ram_mb']}  "
        f"peak {result['peak_ram_mb']}"
    )
    print(
        f"  frames / samples:  {result['frame_count']} frames, "
        f"{result['sample_count']} CPU samples"
    )


def run_cpu_profile(
    *,
    mode: ProfileMode = "both",
    target_fps: float = 5.0,
    duration_s: float = 600.0,
    warmup_s: float = 15.0,
    camera_index: int = 0,
    experiment: str = "experiment",
    output_path: Path | None = None,
) -> dict:
    """Profile one detector mode for ``duration_s`` after an optional warmup."""
    logical_cores = psutil.cpu_count(logical=True) or 1

    object_detector: ObjectDetector | None = None
    face_gaze_detector: FaceGazeDetector | None = None

    if mode in ("object", "both"):
        object_detector = ObjectDetector()
    if mode in ("face_gaze", "both"):
        face_gaze_detector = FaceGazeDetector(
            config=FaceGazeConfig(
                draw_landmarks=False,
                mirror_preview=False,
                refine_landmarks=True,
            )
        )

    capture_config = CaptureConfig(
        camera_index=camera_index,
        target_fps=target_fps,
    )

    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)

    process_cpu_samples: list[float] = []
    system_cpu_samples: list[float] = []
    ram_samples: list[float] = []

    frame_count = 0
    total_inference_ms = 0.0
    session_start: float | None = None
    last_sample_at = 0.0
    last_progress_at = 0.0
    loop_start = time.perf_counter()

    print(
        f"\nProfiling mode={mode!r} target_fps={target_fps} "
        f"warmup_s={warmup_s} duration_s={duration_s} "
        f"logical_cores={logical_cores}"
    )

    try:
        for frame in capture_frames(capture_config):
            now = time.perf_counter()
            elapsed_total = now - loop_start

            if session_start is None and elapsed_total >= warmup_s:
                session_start = now
                last_sample_at = now
                last_progress_at = now
                proc.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
                print("Warmup done — measuring...")

            start = time.perf_counter()
            if mode == "object":
                assert object_detector is not None
                object_detector.detect(frame=frame)
            elif mode == "face_gaze":
                assert face_gaze_detector is not None
                face_gaze_detector.detect(frame=frame)
            else:
                assert object_detector is not None
                assert face_gaze_detector is not None
                process_frame(frame, object_detector, face_gaze_detector)
            inference_ms = (time.perf_counter() - start) * 1000.0

            if session_start is None:
                continue

            frame_count += 1
            total_inference_ms += inference_ms
            now = time.perf_counter()
            measured_elapsed = now - session_start

            if now - last_sample_at >= 1.0:
                process_cpu = proc.cpu_percent(interval=None)
                system_cpu = psutil.cpu_percent(interval=None)
                ram_mb = proc.memory_info().rss / (1024 * 1024)
                process_cpu_samples.append(process_cpu)
                system_cpu_samples.append(system_cpu)
                ram_samples.append(ram_mb)
                last_sample_at = now

            if (
                duration_s >= PROGRESS_EVERY_S
                and now - last_progress_at >= PROGRESS_EVERY_S
            ):
                live_fps = (
                    frame_count / measured_elapsed if measured_elapsed > 0 else 0.0
                )
                live_machine = (
                    _mean(process_cpu_samples) / logical_cores
                    if process_cpu_samples
                    else 0.0
                )
                remaining = max(0.0, duration_s - measured_elapsed)
                print(
                    f"  … {measured_elapsed:.0f}s / {duration_s:.0f}s "
                    f"| ~{live_fps:.1f} fps "
                    f"| ~{live_machine:.1f}% machine CPU "
                    f"| ~{remaining:.0f}s left"
                )
                last_progress_at = now

            if measured_elapsed >= duration_s:
                break
    finally:
        if face_gaze_detector is not None:
            face_gaze_detector.close()

    measured_s = duration_s if session_start is not None else 0.0
    if session_start is not None:
        measured_s = min(duration_s, time.perf_counter() - session_start)

    avg_process_cpu = _mean(process_cpu_samples)
    peak_process_cpu = max(process_cpu_samples) if process_cpu_samples else 0.0
    avg_machine_cpu = avg_process_cpu / logical_cores
    peak_machine_cpu = peak_process_cpu / logical_cores
    avg_system_cpu = _mean(system_cpu_samples)
    avg_ram = _mean(ram_samples)
    peak_ram = max(ram_samples) if ram_samples else 0.0
    avg_fps = frame_count / measured_s if measured_s > 0 else 0.0
    avg_inference_ms = total_inference_ms / frame_count if frame_count else 0.0

    result = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "mode": mode,
        "target_fps": target_fps,
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "logical_cpu_count": logical_cores,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "frame_count": frame_count,
        "avg_fps": round(avg_fps, 3),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "avg_process_cpu_percent": round(avg_process_cpu, 2),
        "peak_process_cpu_percent": round(peak_process_cpu, 2),
        "avg_machine_cpu_percent": round(avg_machine_cpu, 2),
        "peak_machine_cpu_percent": round(peak_machine_cpu, 2),
        "avg_system_cpu_percent": round(avg_system_cpu, 2),
        "avg_ram_mb": round(avg_ram, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "sample_count": len(process_cpu_samples),
    }

    _print_summary(result)

    if output_path is not None:
        append_result_row(output_path, result)
        print(f"\nAppended row to {output_path}")

    return result


def run_mode_suite(
    *,
    modes: list[ProfileMode],
    target_fps: float,
    duration_s: float,
    warmup_s: float,
    camera_index: int,
    experiment: str,
    output_path: Path,
) -> list[dict]:
    """Run several modes back-to-back and save each row."""
    rows: list[dict] = []
    for i, mode in enumerate(modes, start=1):
        print(f"\n--- Mode {i}/{len(modes)}: {mode} ---")
        rows.append(
            run_cpu_profile(
                mode=mode,
                target_fps=target_fps,
                duration_s=duration_s,
                warmup_s=warmup_s,
                camera_index=camera_index,
                experiment=experiment,
                output_path=output_path,
            )
        )
    return rows
