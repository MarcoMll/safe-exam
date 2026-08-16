"""Clip staging, metadata stream, and clip upload. Owned by #36–#38."""

import json
import subprocess
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urljoin

import imageio_ffmpeg
import numpy as np

from safe_exam.agent.buffer import FrameEntry
from safe_exam.agent.config import Config
from safe_exam.agent.fusion import FlagEvent
from safe_exam.processor.frame_result import ProcessFrameOutput

METADATA_INGEST_PATH = "metadata/ingest"


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
    """Load the upload queue from disk, or return an empty list if it doesn't exist."""
    if not queue_path.exists():
        return []

    with queue_path.open("r", encoding="utf-8") as queue_file:
        try:
            return json.load(queue_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Upload queue is not valid JSON: {queue_path}") from exc


# I think the Queue path could be added to a config
def _save_upload_queue(queue_path: Path, queue: list[dict[str, str]]) -> None:
    """Write clip metadata alongside the staged MP4."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    with queue_path.open("w", encoding="utf-8") as queue_file:
        json.dump(queue, queue_file, indent=2)


def _add_to_upload_queue(
    queue_path: Path, *, clip_path: Path, sidecar_path: Path
) -> None:
    """Add a new clip to the upload queue."""
    queue = _load_upload_queue(queue_path)

    entry = {
        "clip_path": str(clip_path),
        "sidecar_path": str(sidecar_path),
    }
    queue.append(entry)

    _save_upload_queue(queue_path, queue)


def _write_clip_sidecar(
    sidecar_path: Path,
    *,
    flag: FlagEvent,
    config: Config,
    phone_confidence: float,
    gaze_off_seconds: float,
    extra_person_detected: bool,
) -> None:
    """
    Write clip metadata along side the staged MP4
    """
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar_data = {
        "exam_id": config.exam_id,
        "student_id": config.student_id,
        "timestamp": float(flag.timestamp),
        "phone_confidence": float(phone_confidence),
        "gaze_off_seconds": float(gaze_off_seconds),
        "extra_person_detected": bool(extra_person_detected),
        "fused_score": float(flag.score),
    }

    with sidecar_path.open("w", encoding="utf-8") as sidecar_file:
        json.dump(sidecar_data, sidecar_file, indent=2)


def _encode_frames_to_mp4(
    frames: list[FrameEntry],
    *,
    clip_path: Path,
    fps: float,
    bitrate: str,
) -> None:
    """
    Encode BGR frames to an H.264 MP4 file
    """
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
        f"{width}x{height}",
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

    try:
        for _, frame in frames:

            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame dimensions {frame.shape[:2]} do not match expected "
                    f"dimensions {(height, width)}"
                )

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Frames must be BGR images with 3 channels")

            if frame.dtype != np.uint8:
                raise ValueError("Frames must use uint8 pixel values")

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
    phone_confidence: float,
    gaze_off_seconds: float,
    extra_person_detected: bool,
    # Bitrate should be moved to the config
    clip_bitrate: str = "500k",
    clip_dir: Path = Path("data/staged_clips"),
) -> tuple[Path, Path]:
    """
    Encode extracted frames to MP4, write a JSON sidecar, and persist the upload queue.

    Returns:
        (clip_path, sidecar_path)
    """
    stage_dir = clip_dir
    stage_dir.mkdir(parents=True, exist_ok=True)

    timestamp_unix = int(flag.timestamp)
    filename_base = f"{config.exam_id}_{config.student_id}_{timestamp_unix}"

    clip_path = stage_dir / f"{filename_base}.mp4"
    sidecar_path = stage_dir / f"{filename_base}.json"
    queue_path = stage_dir / "upload_queue.json"

    _encode_frames_to_mp4(
        frames,
        clip_path=clip_path,
        fps=config.sampling_fps,
        bitrate=clip_bitrate,
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
