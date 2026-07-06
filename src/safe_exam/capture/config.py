from dataclasses import dataclass


@dataclass
class CaptureConfig:
    target_fps: float = 12.0
    camera_index: int = 0
    show_debug: bool = True
    window_name: str = "safe-exam capture"
