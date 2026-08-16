"""Clip staging, metadata stream, and clip upload. Owned by #36–#38."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urljoin

import cv2  # pylint: disable=no-member
import imageio_ffmpeg
import numpy as np

from safe_exam.agent.buffer import FrameEntry
from safe_exam.agent.config import Config
from safe_exam.agent.fusion import FlagEvent
from safe_exam.processor.frame_result import FrameResult, ProcessFrameOutput

METADATA_INGEST_PATH = "metadata/ingest"
DEFAULT_CLIP_DIR = Path("data/staged_clips")
QUEUE_FILENAME = "upload_queue.jsonl"
DEFAULT_CLIP_BITRATE = "500k"

_queue_lock = Lock()


class MetadataStreamThread:
    """Collect lightweight per-frame metadata for periodic background upload."""

    def __init__(
        self,
        *,
        server_url: str,
        session_id: str,
        auth_token: str,
        interval_seconds: float = 5.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than 0")

        self.recording = False
        self.server_url = server_url.rstrip("/")
        self.endpoint_url = urljoin(f"{self.server_url}/", METADATA_INGEST_PATH)
        self.session_id = session_id
        self.auth_token = auth_token
        self.interval_seconds = float(interval_seconds)

        self._signals: list[dict] = []
        self._lock = Lock()

        self._thread: Thread | None = None

    def start(self) -> None:
        """Allow the main loop to begin recording metadata entries."""
        self.recording = True

    def stop(self) -> None:
        """Stop accepting new metadata entries."""
        self.recording = False

    def start_recording(self) -> None:
        """Compatibility alias while the session lifecycle is still taking shape."""
        self.start()

    def stop_recording(self) -> None:
        """Compatibility alias while the session lifecycle is still taking shape."""
        self.stop()

    def record_frame(
        self,
        output: ProcessFrameOutput,
        *,
        gaze_off_seconds: float,
        fused_score: float,
    ) -> None:
        """
        Copy one processed frame into the metadata buffer.

        This method stays intentionally cheap: no HTTP, no disk writes, just a
        small dictionary append protected by a lock.
        """
        if not self.recording:
            return

        signal = self._build_signal(
            output,
            gaze_off_seconds=gaze_off_seconds,
            fused_score=fused_score,
        )

        with self._lock:
            self._signals.append(signal)

    def _drain_signals(self) -> tuple[dict, ...]:
        """Take the current buffered entries and clear the in-memory buffer."""
        with self._lock:
            signals = tuple(self._signals)
            self._signals.clear()

        return signals

    @staticmethod
    def _build_signal(
        output: ProcessFrameOutput,
        *,
        gaze_off_seconds: float,
        fused_score: float,
    ) -> dict:
        """Build one JSON-ready signal entry from processor and fusion output."""
        signal = output.as_dict()
        signal["extra_person_detected"] = output.result.person_count > 1
        signal["gaze_off_seconds"] = float(gaze_off_seconds)
        signal["fused_score"] = float(fused_score)

        return signal


def _load_upload_queue(queue_path: Path) -> list[dict[str, str]]:
    """Load the jsonl upload queue, or return [] if the file is missing."""
    if not queue_path.is_file():
        return []

    entries: list[dict[str, str]] = []
    with queue_path.open(encoding="utf-8") as queue_file:
        for line_number, line in enumerate(queue_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Upload queue line {line_number} is not valid JSON: {queue_path}"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Upload queue line {line_number} must be a JSON object: "
                    f"{queue_path}"
                )
            entries.append(entry)
    return entries


def _add_to_upload_queue(
    queue_path: Path, *, clip_path: Path, sidecar_path: Path
) -> None:
    """Append one clip/sidecar pair to the persistent jsonl upload queue."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "clip_path": str(clip_path.resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
    }
    line = json.dumps(entry) + "\n"
    with _queue_lock:
        with queue_path.open("a", encoding="utf-8") as queue_file:
            queue_file.write(line)


def load_pending_uploads(queue_path: Path) -> list[tuple[Path, Path]]:
    """Return pending (clip, sidecar) paths for #38 / tests."""
    pending: list[tuple[Path, Path]] = []
    for entry in _load_upload_queue(queue_path):
        pending.append((Path(entry["clip_path"]), Path(entry["sidecar_path"])))
    return pending


def _write_clip_sidecar(
    sidecar_path: Path,
    *,
    flag: FlagEvent,
    config: Config,
    phone_confidence: float,
    gaze_off_seconds: float,
    extra_person_detected: bool,
) -> None:
    """Write clip metadata alongside the staged MP4."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar_data = {
        "exam_id": config.exam_id,
        "student_id": config.student_id,
        "timestamp": float(flag.timestamp),
        "phone_confidence": float(phone_confidence),
        "gaze_off_seconds": float(gaze_off_seconds),
        "extra_person_detected": bool(extra_person_detected),
        "fused_score": float(flag.score),
        "reasons": list(flag.reasons),
    }

    sidecar_path.write_text(
        json.dumps(sidecar_data, indent=2),
        encoding="utf-8",
    )


def _encode_frames_to_mp4(
    frames: list[FrameEntry],
    *,
    clip_path: Path,
    fps: float,
    bitrate: str,
) -> None:
    """Encode BGR frames to an H.264 MP4 via ffmpeg (imageio-ffmpeg)."""
    if not frames:
        raise ValueError("No frames to encode")

    if fps <= 0:
        raise ValueError("FPS must be greater than 0")

    first_frame = frames[0][1]

    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError("Frames must be BGR images with 3 channels")

    if first_frame.dtype != np.uint8:
        raise ValueError("Frames must use uint8 pixel values")

    height, width = first_frame.shape[:2]
    # libx264 + yuv420p typically requires even dimensions.
    out_w = width - (width % 2)
    out_h = height - (height % 2)
    if out_w < 2 or out_h < 2:
        raise ValueError("Frame size is too small to encode")

    clip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required to encode clips but was not found"
        ) from exc

    command = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-b:v",
        bitrate,
        "-pix_fmt",
        "yuv420p",
        str(clip_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to encode clips but was not found"
        ) from exc

    assert process.stdin is not None
    assert process.stderr is not None

    return_code = -1
    stderr = ""
    try:
        for _, frame in frames:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Frames must be BGR images with 3 channels")
            if frame.dtype != np.uint8:
                raise ValueError("Frames must use uint8 pixel values")

            if frame.shape[0] != out_h or frame.shape[1] != out_w:
                frame = cv2.resize(frame, (out_w, out_h))

            frame = np.ascontiguousarray(frame)
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()

    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed while encoding clip. Stderr: {stderr}")


def stage_clip(
    frames: list[FrameEntry],
    *,
    flag: FlagEvent,
    config: Config,
    phone_confidence: float = 0.0,
    gaze_off_seconds: float = 0.0,
    extra_person_detected: bool = False,
    frame_result: FrameResult | None = None,
    clip_bitrate: str | None = None,
    clip_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Encode extracted frames to H.264 MP4, write a JSON sidecar, enqueue for upload.

    When ``frame_result`` is set, ``phone_confidence`` and ``extra_person_detected``
    are taken from that result. Bitrate and directory default from ``config``
    (``clip_bitrate``, ``clip_dir``) unless overridden.

    Returns:
        (clip_path, sidecar_path)
    """
    if frame_result is not None:
        phone_confidence = float(frame_result.phone_confidence)
        extra_person_detected = frame_result.person_count > 1

    resolved_bitrate = clip_bitrate or config.clip_bitrate
    stage_dir = clip_dir or config.clip_dir
    stage_dir.mkdir(parents=True, exist_ok=True)

    timestamp_unix = int(flag.timestamp)
    filename_base = f"{config.exam_id}_{config.student_id}_{timestamp_unix}"

    clip_path = stage_dir / f"{filename_base}.mp4"
    sidecar_path = stage_dir / f"{filename_base}.json"
    queue_path = stage_dir / QUEUE_FILENAME

    _encode_frames_to_mp4(
        frames,
        clip_path=clip_path,
        fps=config.sampling_fps,
        bitrate=resolved_bitrate,
    )

    _write_clip_sidecar(
        sidecar_path,
        flag=flag,
        config=config,
        phone_confidence=phone_confidence,
        gaze_off_seconds=gaze_off_seconds,
        extra_person_detected=extra_person_detected,
    )

    _add_to_upload_queue(
        queue_path,
        clip_path=clip_path,
        sidecar_path=sidecar_path,
    )

    return clip_path, sidecar_path
