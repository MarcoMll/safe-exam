"""Session lifecycle orchestrator (startup, loop, shutdown). Owned by #39."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from threading import Event

import cv2  # pylint: disable=no-member
import requests

from safe_exam.agent.buffer import RingBuffer
from safe_exam.agent.config import Config, build_capture_config
from safe_exam.agent.fusion import SignalFuser
from safe_exam.agent.network import (
    QUEUE_FILENAME,
    ClipUploadThread,
    MetadataStreamThread,
    load_pending_uploads,
    stage_clip,
)
from safe_exam.capture.capture import capture_frames
from safe_exam.detectors.face_gaze import FaceGazeConfig, FaceGazeDetector
from safe_exam.detectors.object import ObjectDetector
from safe_exam.processor.frame_processor import process_frame

DISK_SPACE_FLOOR_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
logger = logging.getLogger(__name__)


class ChecklistError(RuntimeError):
    """Pre-exam check failed. Message is safe to print to the user"""


def check_server(server_url: str) -> None:
    """GET /health must return 200"""
    url = server_url.rstrip("/") + "/health"

    try:
        response = requests.get(url, timeout=5)
    except requests.RequestException as exc:
        raise ChecklistError(f"Server is not reachable at {url}") from exc
    if response.status_code != 200:
        raise ChecklistError(
            f"Server health check failed ({response.status_code}) at {url}"
        )


def check_auth(server_url: str, auth_token: str) -> None:
    """GET /auth/check must accept the Bearer token"""
    url = server_url.rstrip("/") + "/auth/check"

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=5,
        )
    except requests.RequestException as exc:
        raise ChecklistError(f"Auth check could not reach {url}") from exc
    if response.status_code == 401:
        raise ChecklistError("Auth token was rejected by the server")
    if response.status_code != 200:
        raise ChecklistError(f"Auth check failed ({response.status_code}) at {url}")


def check_disk_space(path: Path, *, floor_bytes: int = DISK_SPACE_FLOOR_BYTES) -> None:
    """Need at least 2 GB free for the ring buffer and staged clips"""
    usage = shutil.disk_usage(path)
    if usage.free < floor_bytes:
        free_gb = usage.free / (1024**3)
        raise ChecklistError(
            f"Not enough disk space ({free_gb:.1f} GB free, need 2 GB)"
        )


def check_camera(camera_index: int) -> None:
    """Open the camera and read one frame"""
    cap = cv2.VideoCapture(camera_index)

    try:
        if not cap.isOpened():
            raise ChecklistError(f"Camera {camera_index} is not accessible")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ChecklistError(
                f"Camera {camera_index} opened but did not return a frame"
            )
    finally:
        cap.release()


def run_checklist(config: Config, *, camera_index: int = 0) -> None:
    """Run all pre-exam checks in order. Raises ChecklistError on first failure."""
    logger.info("Running pre-exam checklist...")
    check_server(config.server_url)
    check_auth(config.server_url, config.auth_token)
    check_disk_space(config.clip_dir)
    check_camera(camera_index)
    logger.info("Pre-exam checklist passed")


class SessionError(RuntimeError):
    """Session start or end failed. Message is safe to print to the user."""


def start_session(config: Config) -> str:
    """POST /session/start and return the server-issued session_id"""
    url = config.server_url.rstrip("/") + "/session/start"

    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {config.auth_token}"},
            json={
                "exam_id": config.exam_id,
                "student_id": config.student_id,
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        raise SessionError(f"Could not start session at {url}") from exc

    if response.status_code != 200:
        raise SessionError(f"Session start failed ({response.status_code}) at {url}")

    try:
        payload = response.json()
        session_id = payload["session_id"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SessionError("Session start response was missing session_id") from exc

    if not isinstance(session_id, str) or not session_id:
        raise SessionError("Session start response had an empty session_id")

    logger.info("Session started: %s", session_id)
    return session_id


def end_session(
    config: Config,
    session_id: str,
    *,
    duration_seconds: float,
    flag_count: int,
    clips_uploaded: int,
    clips_pending: int,
) -> None:
    """POST /session/end with the shutdown summary. Best-effort: log, don't crash."""
    url = config.server_url.rstrip("/") + "/session/end"

    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {config.auth_token}"},
            json={
                "session_id": session_id,
                "duration_seconds": duration_seconds,
                "flag_count": flag_count,
                "clips_uploaded": clips_uploaded,
                "clips_pending": clips_pending,
            },
            timeout=10,
        )
    except requests.RequestException:
        logger.exception("Could not reach %s to end session %s", url, session_id)
        return

    if response.status_code != 200:
        logger.warning(
            "Session end failed (%s) for %s",
            response.status_code,
            session_id,
        )
        return

    logger.info("Session closed: %s", session_id)


class ExamSession:
    """One Exam run: checklist, threads, detectors, shutdown"""

    def __init__(self, config: Config, *, camera_index: int = 0) -> None:
        self.config = config
        self.camera_index = camera_index
        self.session_id: str | None = None
        self.started_at: float | None = None
        self.flag_count: int = 0
        self._stop = Event()

        self.ring_buffer: RingBuffer | None = None
        self.metadata_stream: MetadataStreamThread | None = None
        self.clip_uploader: ClipUploadThread | None = None
        self.fuser: SignalFuser | None = None
        self.object_detector: ObjectDetector | None = None
        self.face_gaze_detector: FaceGazeDetector | None = None

    def start(self) -> None:
        """Checklist, register with server, then start threads and detectors"""
        run_checklist(self.config, camera_index=self.camera_index)
        self.session_id = start_session(self.config)
        self.started_at = time.time()

        self.ring_buffer = RingBuffer(
            ring_buffer_seconds=self.config.ring_buffer_seconds,
            sampling_fps=self.config.sampling_fps,
            clip_before_flag_seconds=self.config.clip_before_flag_seconds,
            clip_after_flag_seconds=self.config.clip_after_flag_seconds,
        )
        self.metadata_stream = MetadataStreamThread(
            server_url=self.config.server_url,
            session_id=self.session_id,
            auth_token=self.config.auth_token,
            batch_store_dir=self.config.clip_dir.parent / "metadata_batches",
        )
        self.clip_uploader = ClipUploadThread(
            server_url=self.config.server_url,
            auth_token=self.config.auth_token,
            clip_dir=self.config.clip_dir,
        )
        self.fuser = SignalFuser.from_config(self.config)
        self.object_detector = ObjectDetector()
        self.face_gaze_detector = FaceGazeDetector(
            config=FaceGazeConfig(
                draw_landmarks=False,
                mirror_preview=False,
                refine_landmarks=True,
            )
        )

        self.metadata_stream.start()
        self.clip_uploader.start()
        logger.info("Exam session running: %s", self.session_id)

    def request_stop(self) -> None:
        """SIGTERM / Ctrl+C land here. The loop will see this flag"""
        self._stop.set()

    def run(self) -> None:
        """Capture -> Process -> Fuse -> Record Metadata -> Flag -> Stage Clip"""
        if (
            self.ring_buffer is None
            or self.metadata_stream is None
            or self.fuser is None
            or self.object_detector is None
            or self.face_gaze_detector is None
        ):
            raise RuntimeError("ExamSession.start() must be called before run()")

        capture_config = build_capture_config(
            self.config,
            camera_index=self.camera_index,
        )
        logger.info("Detection loop started")

        for frame in capture_frames(capture_config):
            if self._stop.is_set():
                break

            output = process_frame(
                frame,
                self.object_detector,
                self.face_gaze_detector,
            )
            timestamp = output.result.timestamp
            self.ring_buffer.add_frame(timestamp, frame)

            flag = self.fuser.process_frame(output.result)

            gaze_off_seconds = 0.0
            if self.fuser.gaze_off_start_time is not None:
                gaze_off_seconds = max(
                    0.0,
                    timestamp - self.fuser.gaze_off_start_time,
                )

            self.metadata_stream.record_frame(
                output,
                gaze_off_seconds=gaze_off_seconds,
                fused_score=self.fuser.last_fused_score,
            )

            if flag is None:
                continue

            self.flag_count += 1
            logger.info(
                "Flag #%s score=%.2f reasons=%s",
                self.flag_count,
                flag.score,
                flag.reasons,
            )
            frames = self.ring_buffer.extract_clip(flag.timestamp)
            if not frames:
                logger.warning("Flag had no frames in the ring buffer")
                continue
            try:
                stage_clip(
                    frames,
                    flag=flag,
                    config=self.config,
                    gaze_off_seconds=gaze_off_seconds,
                    frame_result=output.result,
                )
            except Exception:
                logger.exception(
                    "Failed to stage clip for flag at %.3f", flag.timestamp
                )

    def shutdown(self) -> None:
        """Flush metadata, stop uploads, close session, close detectors"""
        if self.metadata_stream is not None:
            self.metadata_stream.stop()
        if self.clip_uploader is not None:
            self.clip_uploader.stop()

        duration_seconds = 0.0
        if self.started_at is not None:
            duration_seconds = time.time() - self.started_at

        clips_pending = 0
        if self.config.clip_dir is not None:
            clips_pending = len(
                load_pending_uploads(self.config.clip_dir / QUEUE_FILENAME)
            )
        clips_uploaded = (
            self.clip_uploader.clips_uploaded if self.clip_uploader is not None else 0
        )

        if self.session_id is not None:
            end_session(
                self.config,
                self.session_id,
                duration_seconds=duration_seconds,
                flag_count=self.flag_count,
                clips_uploaded=clips_uploaded,
                clips_pending=clips_pending,
            )

        if self.face_gaze_detector is not None:
            self.face_gaze_detector.close()
            self.face_gaze_detector = None

        logger.info("Exam session shut down")
