"""Clip staging, metadata stream, and clip upload. Owned by #36–#38."""
import time
from queue import Empty, Queue
from threading import Event, Lock, Thread
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

        self._lock = Lock()
        self._signals: list[dict] = []
        self._pending_packets: Queue[dict] = Queue()

        self._thread: Thread | None = None
        self._stop_event = Event()

    def start_recording(self) -> None:
        if self.recording:
            return

        self._stop_event.clear()

        self.recording = True
        self._thread = Thread(target=self._run)

        if self._thread is not None:
            self._thread.start()

    def stop_recording(self) -> None:
        with self._lock:
            if not self.recording:
                return

            self.recording = False

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

    def record_frame(
        self,
        output: ProcessFrameOutput,
        *,
        gaze_off_seconds: float,
        fused_score: float,
    ) -> None:
        """
        Copy one processed frame into the metadata buffer.
        """
        signal = self._build_signal(
            output,
            gaze_off_seconds=gaze_off_seconds,
            fused_score=fused_score,
        )

        with self._lock:
            if not self.recording:
                return

            self._signals.append(signal)

    def _drain_signals(self) -> tuple[dict, ...]:
        """Take the current buffered entries and clear the in-memory buffer."""
        with self._lock:
            signals = tuple(self._signals)
            self._signals.clear()

        return signals

    def _create_package(self, signals: tuple[dict, ...]) -> dict:
        return  {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "signals": list(signals),
        }

    def _flush_signals(self) -> None:
        signals = self._drain_signals()

        if len(signals) > 0:
            package = self._create_package(signals)
            self._pending_packets.put(package)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            stop_requested = self._stop_event.wait(self.interval_seconds)

            if stop_requested:
                self._flush_signals()
                return

            self._flush_signals()

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

    def get_pending_packets(self) -> list[dict]:
        packets = []

        while True:
            try:
                packets.append(self._pending_packets.get_nowait())
            except Empty:
                break

        return packets
