"""Tests for clip staging (#36): H.264 MP4 + JSON sidecar + jsonl queue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
from safe_exam.agent.network import (
    _add_to_upload_queue,
    _encode_frames_to_mp4,
    _load_upload_queue,
    _write_clip_sidecar,
    load_pending_uploads,
    stage_clip,
)
from safe_exam.processor.frame_result import FrameResult


def _build_config() -> Config:
    """Minimal agent Config for staging tests (no YAML file needed)."""
    return Config(
        server_url="https://localtest",
        exam_id="EXAM_2026_FINAL",
        student_id="321481",
        auth_token="secret",
        sampling_fps=5.0,
        ring_buffer_seconds=60.0,
        clip_before_flag_seconds=15.0,
        clip_after_flag_seconds=5.0,
        clip_bitrate="500k",
        clip_dir=Path("data/staged_clips"),
        metadata_interval_seconds=5.0,
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


def test_write_clip_sidecar_writes_required_metadata(tmp_path: Path) -> None:
    """Sidecar JSON includes issue fields plus FlagEvent reasons."""
    config = _build_config()
    flag = FlagEvent(
        timestamp=1720000000.0,
        score=0.82,
        reasons=["phone", "gaze_violation"],
    )
    sidecar_path = tmp_path / "clip.json"

    _write_clip_sidecar(
        sidecar_path,
        flag=flag,
        config=config,
        phone_confidence=0.71,
        gaze_off_seconds=12.0,
        extra_person_detected=False,
    )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert sidecar == {
        "exam_id": "EXAM_2026_FINAL",
        "student_id": "321481",
        "timestamp": 1720000000.0,
        "phone_confidence": 0.71,
        "gaze_off_seconds": 12.0,
        "extra_person_detected": False,
        "fused_score": 0.82,
        "reasons": ["phone", "gaze_violation"],
    }


def test_add_to_upload_queue_persists_clip_and_sidecar_paths(
    tmp_path: Path,
) -> None:
    """Enqueue appends one jsonl object with absolute clip/sidecar paths."""
    queue_path = tmp_path / "upload_queue.jsonl"
    clip_path = tmp_path / "clip.mp4"
    sidecar_path = tmp_path / "clip.json"

    _add_to_upload_queue(
        queue_path,
        clip_path=clip_path,
        sidecar_path=sidecar_path,
    )

    queue = _load_upload_queue(queue_path)

    assert queue == [
        {
            "clip_path": str(clip_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
        }
    ]


def test_load_upload_queue_returns_empty_list_when_missing(tmp_path: Path) -> None:
    """Missing queue file means nothing pending on first startup."""
    queue_path = tmp_path / "missing_queue.jsonl"
    assert _load_upload_queue(queue_path) == []
    assert load_pending_uploads(queue_path) == []


def test_load_upload_queue_raises_for_invalid_json(tmp_path: Path) -> None:
    """Corrupt jsonl lines fail loudly instead of silent data loss."""
    queue_path = tmp_path / "upload_queue.jsonl"
    queue_path.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        _load_upload_queue(queue_path)


def test_stage_clip_writes_sidecar_and_queue_without_real_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stage_clip orchestration works even when encode is mocked."""
    config = _build_config()
    flag = FlagEvent(
        timestamp=1720000000.0,
        score=0.82,
        reasons=["phone"],
    )
    frames = [
        (1720000000.0, np.zeros((4, 4, 3), dtype=np.uint8)),
        (1720000000.2, np.ones((4, 4, 3), dtype=np.uint8)),
    ]

    def fake_encode_frames_to_mp4(frames, *, clip_path, fps, bitrate):
        clip_path.write_bytes(b"fake mp4 data")

    monkeypatch.setattr(
        "safe_exam.agent.network._encode_frames_to_mp4",
        fake_encode_frames_to_mp4,
    )

    clip_path, sidecar_path = stage_clip(
        frames,
        flag=flag,
        config=config,
        phone_confidence=0.71,
        gaze_off_seconds=12.0,
        extra_person_detected=True,
        clip_bitrate="500k",
        clip_dir=tmp_path,
    )

    assert clip_path == tmp_path / "EXAM_2026_FINAL_321481_1720000000.mp4"
    assert sidecar_path == tmp_path / "EXAM_2026_FINAL_321481_1720000000.json"
    assert clip_path.read_bytes() == b"fake mp4 data"

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["exam_id"] == "EXAM_2026_FINAL"
    assert sidecar["student_id"] == "321481"
    assert sidecar["timestamp"] == 1720000000.0
    assert sidecar["phone_confidence"] == 0.71
    assert sidecar["gaze_off_seconds"] == 12.0
    assert sidecar["extra_person_detected"] is True
    assert sidecar["fused_score"] == 0.82
    assert sidecar["reasons"] == ["phone"]

    queue = _load_upload_queue(tmp_path / "upload_queue.jsonl")
    assert queue == [
        {
            "clip_path": str(clip_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
        }
    ]


def test_stage_clip_uses_frame_result_when_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional FrameResult fills phone/person sidecar fields for session use."""
    config = _build_config()
    flag = FlagEvent(timestamp=1720000000.0, score=0.5, reasons=["phone"])

    monkeypatch.setattr(
        "safe_exam.agent.network._encode_frames_to_mp4",
        lambda frames, *, clip_path, fps, bitrate: clip_path.write_bytes(b"x"),
    )

    _, sidecar_path = stage_clip(
        [(1720000000.0, np.zeros((8, 8, 3), dtype=np.uint8))],
        flag=flag,
        config=config,
        frame_result=FrameResult(phone_confidence=0.91, person_count=2),
        gaze_off_seconds=3.0,
        clip_dir=tmp_path,
    )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["phone_confidence"] == 0.91
    assert sidecar["extra_person_detected"] is True
    assert sidecar["gaze_off_seconds"] == 3.0


def test_encode_frames_to_mp4_writes_real_mp4(tmp_path: Path) -> None:
    """Real ffmpeg/libx264 encode produces a non-empty MP4 on disk."""
    frames = [
        (1720000000.0, np.zeros((16, 16, 3), dtype=np.uint8)),
        (1720000000.2, np.full((16, 16, 3), 80, dtype=np.uint8)),
        (1720000000.4, np.full((16, 16, 3), 160, dtype=np.uint8)),
        (1720000000.6, np.full((16, 16, 3), 255, dtype=np.uint8)),
    ]
    clip_path = tmp_path / "clip.mp4"

    _encode_frames_to_mp4(
        frames,
        clip_path=clip_path,
        fps=5.0,
        bitrate="500k",
    )

    assert clip_path.exists()
    assert clip_path.stat().st_size > 0


def test_upload_queue_persists_across_restart_simulation(tmp_path: Path) -> None:
    """Two enqueues survive a simulated process restart via jsonl reload."""
    queue_path = tmp_path / "upload_queue.jsonl"

    first_clip_path = tmp_path / "clip-one.mp4"
    first_sidecar_path = tmp_path / "clip-one.json"
    second_clip_path = tmp_path / "clip-two.mp4"
    second_sidecar_path = tmp_path / "clip-two.json"

    _add_to_upload_queue(
        queue_path,
        clip_path=first_clip_path,
        sidecar_path=first_sidecar_path,
    )
    assert load_pending_uploads(queue_path) == [
        (first_clip_path.resolve(), first_sidecar_path.resolve()),
    ]

    _add_to_upload_queue(
        queue_path,
        clip_path=second_clip_path,
        sidecar_path=second_sidecar_path,
    )
    assert load_pending_uploads(queue_path) == [
        (first_clip_path.resolve(), first_sidecar_path.resolve()),
        (second_clip_path.resolve(), second_sidecar_path.resolve()),
    ]
