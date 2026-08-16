"""Tiny demo for clip staging (#36).

Writes a short synthetic H.264 MP4 + JSON sidecar into data/staged_clips/
and appends to upload_queue.jsonl so you can inspect the files.

Usage (from repo root, venv active, package installed):

    pip install -r requirements.txt   # includes imageio-ffmpeg
    python scripts/stage_clip_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_exam.agent.config import (
    Config,
    DetectorConfig,
    FusionConfig,
    GazeConfig,
    LoggingConfig,
    MultiPersonConfig,
    PhoneConfig,
)
from safe_exam.agent.fusion import FlagEvent
from safe_exam.agent.network import load_pending_uploads, stage_clip
from safe_exam.processor.frame_result import FrameResult


def _demo_config() -> Config:
    return Config(
        server_url="http://127.0.0.1:8000",
        exam_id="DEMO_EXAM",
        student_id="S999",
        auth_token="demo",
        sampling_fps=5.0,
        ring_buffer_seconds=60.0,
        clip_before_flag_seconds=15.0,
        clip_after_flag_seconds=5.0,
        clip_bitrate="500k",
        clip_dir=Path("data/staged_clips"),
        detectors=DetectorConfig(
            phone=PhoneConfig(enabled=True, confidence_threshold=0.5),
            gaze=GazeConfig(
                enabled=True,
                head_pitch_threshold=20.0,
                head_yaw_threshold=25.0,
                eye_pitch_threshold=15.0,
                eye_yaw_threshold=20.0,
                duration_threshold_seconds=4.0,
            ),
            multi_person=MultiPersonConfig(enabled=True, roi_mode="proximity"),
        ),
        fusion=FusionConfig(
            phone_weight=0.5,
            gaze_weight=0.3,
            extra_person_weight=0.4,
            multi_signal_bonus=0.3,
            flag_threshold=0.5,
            flag_cooldown_seconds=30.0,
        ),
        logging=LoggingConfig(level="INFO", log_dir=Path("logs")),
    )


def main() -> None:
    config = _demo_config()
    t0 = 1_720_000_000.0
    frames = [
        (t0 + i * 0.2, np.full((120, 160, 3), (i * 20) % 255, dtype=np.uint8))
        for i in range(15)
    ]
    flag = FlagEvent(timestamp=t0, score=0.82, reasons=["phone"])

    clip_path, sidecar_path = stage_clip(
        frames,
        flag=flag,
        config=config,
        frame_result=FrameResult(phone_confidence=0.71, person_count=1),
        gaze_off_seconds=4.0,
    )

    queue_path = config.clip_dir / "upload_queue.jsonl"
    pending = load_pending_uploads(queue_path)

    print("Staged clip demo (#36)")
    print(f"  mp4:     {clip_path.resolve()} ({clip_path.stat().st_size} bytes)")
    print(f"  sidecar: {sidecar_path.resolve()}")
    print(f"  queue:   {queue_path.resolve()} ({len(pending)} pending)")
    print(f"  bitrate: {config.clip_bitrate}")
    print(f"Open {config.clip_dir}/ to inspect the files.")


if __name__ == "__main__":
    main()
