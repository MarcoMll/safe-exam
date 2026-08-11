"""Clip staging, metadata stream, and clip upload. Owned by #36–#38."""

from threading import Lock, Thread
from urllib.parse import urljoin

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
