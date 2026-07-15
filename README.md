# safe-exam

A modular proctoring layer for [Safe Exam Browser (SEB)](https://safeexambrowser.org/). The system uses webcam-based computer vision to detect suspicious behavior during exams — phone use, gaze off-screen, additional people in frame — and raises **flags with confidence scores** for professor review. It does not automatically remove students from exams.

**Current phase:** Phase 0 — local detection prototype. No SEB integration, server, or recording yet. The goal is to prove that detection runs reliably on student hardware before building infrastructure around it.

## What we're detecting

| Signal | Approach (Phase 0) | Module |
|--------|-------------------|--------|
| Phone in frame | YOLO (nano/small) | `detectors/object/` |
| Person count | YOLO | `detectors/object/` |
| Head orientation | MediaPipe Face Mesh + solvePnP | `detectors/face_gaze/` |
| Iris / eye gaze | MediaPipe iris landmarks | `detectors/face_gaze/` |
| Combined gaze | Head + eye angles | `detectors/face_gaze/` → `FrameResult` |

Detectors plug into a shared capture loop and are merged into a single `process_frame()` → `FrameResult` output for later signal fusion (Phase 1).

## Repository layout

```
safe-exam/
├── .github/workflows/        # CI (lint checks on pull requests)
├── docs/
│   └── experiments/
│       └── phone-calibration/  # Threshold findings + result CSVs (#12)
├── models/                   # Downloaded model weights (not committed)
├── scripts/                  # Standalone demos and calibration tools
│   ├── detector_test.py      # Object detection only
│   ├── face_gaze_demo.py     # Face gaze (head + iris) only
│   └── phone_calibration.py  # Phone threshold calibration (#12)
├── src/safe_exam/            # Main application package
│   ├── capture/              # Webcam capture (#7)
│   │   ├── capture_config.py
│   │   └── capture.py
│   ├── detectors/            # Detection modules (#8–#10)
│   │   ├── object/           # YOLO — phone + person
│   │   │   ├── config.py
│   │   │   ├── detector.py
│   │   │   └── overlay.py
│   │   └── face_gaze/        # MediaPipe — head pose + iris gaze
│   │       ├── config.py
│   │       ├── detector.py
│   │       ├── overlay.py
│   │       └── iris_estimation.py  # internal helper
│   ├── processor/            # Unified frame processor (#11)
│   │   ├── frame_result.py   # FrameResult schema
│   │   ├── frame_processor.py
│   │   ├── debug_overlay.py  # composite debug view
│   │   └── session_stats.py
│   └── utils/                # Shared helpers (logging, paths)
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev tools (formatting, linting, hooks)
└── pyproject.toml            # Tool configuration (Black, Ruff)
```

`src/safe_exam/` is the main application package. Import detectors from their subpackages:

```python
from safe_exam.detectors.object import ObjectDetector, ObjectDetectorConfig
from safe_exam.detectors.face_gaze import FaceGazeDetector, FaceGazeConfig
from safe_exam.processor.frame_processor import process_frame
from safe_exam.processor.frame_result import FrameResult
```

Each detector subpackage follows the same layout: `config.py`, `detector.py`, `overlay.py`. The face gaze subpackage also has `iris_estimation.py` for internal iris math.

## Setup

Requires **Python 3.10+** and a working webcam.

```bash
git clone https://github.com/MarcoMll/safe-exam.git
cd safe-exam

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

Verify the environment:

```bash
python -c "import cv2, mediapipe; from ultralytics import YOLO; print('Environment OK')"
```

On first YOLO run, model weights are downloaded automatically into `models/` (gitignored).

> **Note:** MediaPipe 0.10.31+ removed the legacy `solutions` API used by face gaze detection. `requirements.txt` pins `mediapipe` to `<0.10.30`. If you see `module 'mediapipe' has no attribute 'solutions'`, reinstall dependencies: `pip install -r requirements.txt`.

### Standalone demos

Test individual detectors without the full processor:

```bash
python scripts/detector_test.py       # YOLO object detection only
python scripts/face_gaze_demo.py      # Face gaze (head + iris) only
python scripts/phone_calibration.py   # Phone threshold calibration (#12)
```

Calibration findings and how to reproduce them: [`docs/experiments/phone-calibration/`](docs/experiments/phone-calibration/).

### Run the webcam capture loop (#7)

```bash
python -m safe_exam.capture
```

A window opens with the live feed, FPS, and frame size. Press `q` to quit.

To use a different FPS in code:

```python
from safe_exam.capture.capture_config import CaptureConfig
from safe_exam.capture.capture import capture_frames

for frame in capture_frames(CaptureConfig(target_fps=5)):
    ...  # detector code goes here
```

### Run the unified frame processor (#11)

```bash
python -m safe_exam.processor
```

This runs capture + YOLO + face gaze (head pose + iris) with the debug overlay window **on by default**. Press `q` to quit.

The overlay shows YOLO boxes, face mesh, head direction line (green), iris markers (cyan), and a summary panel with head/eye/gaze angles.

Optional flags:

```bash
python -m safe_exam.processor --no-debug
python -m safe_exam.processor --camera-index 1 --target-fps 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--no-debug` | off | Headless run — no overlay window |
| `--camera-index` | `0` | Webcam device index |
| `--target-fps` | `12` | Target capture frame rate |

Session stats (avg fps, inference time, detection counts, attention counters) are logged every 60 frames and again when the processor stops.

### Attention signals (head, eye, gaze)

Each `FrameResult` from `process_frame()` is the processor's public per-frame contract:

**Object detection (YOLO)**

| Field | Meaning |
|-------|---------|
| `phone_detected`, `phone_confidence` | Phone present and best confidence |
| `person_count`, `extra_person_detected` | Number of persons; `True` when count > 1 |

**Face / attention (MediaPipe)**

| Field | Meaning |
|-------|---------|
| `face_detected` | Whether a face was tracked this frame |
| `head_pitch`, `head_yaw` | Face orientation (solvePnP) — obvious head turns |
| `head_direction` | Text label (`Forward`, `Looking Left`, etc.) |
| `eye_pitch`, `eye_yaw` | Iris offset within the eyes — subtle glances |
| `gaze_pitch`, `gaze_yaw` | Combined look direction (`head + eye`) |
| `iris_offset_x`, `iris_offset_y` | Normalized iris position (~`-0.5..0.5`, `0` = centered) |
| `timestamp` | Unix timestamp for the frame |

Extra-person cheating signals use YOLO (`person_count`). Subtle gaze cheating (head forward, eyes down) is tracked via `eye_pitch` / `eye_yaw` and `iris_offset_*`.

**Debug overlays:** per-detector drawing lives in each subpackage (`face_gaze/overlay.py`, `object/overlay.py`). The processor composes them in `processor/debug_overlay.py` for the full pipeline view.

**Session logging:** `processor/session_stats.py` tracks frame counts including `head_off_center_frames` and `eye_off_center_frames` (logged every 60 frames).

## Development workflow

### Branch strategy

- `main` is protected — do not push directly.
- Create a feature branch for each issue: `git checkout -b feature/7-webcam-capture`
- Open a pull request to merge into `main`.
- At least **one review** is required before merge.

### Code quality (Black + Ruff)

We use [pre-commit](https://pre-commit.com/) hooks so formatting and linting run automatically before each commit.

| Tool | Role |
|------|------|
| **Black** | Code **formatter** — rewrites Python to a consistent style (line length, quotes, spacing). You don't argue about formatting; Black decides. |
| **Ruff** | Code **linter** — catches bugs and style issues (unused imports, undefined names, risky patterns) and can auto-fix many of them. Much faster than traditional Python linters. |

Run manually without committing:

```bash
black .
ruff check . --fix
```

Or run all hooks on the whole repo:

```bash
pre-commit run --all-files
```

Configuration lives in `pyproject.toml`.

> **Note:** `pre-commit` is installed inside the project `venv`, not globally. Activate the venv before running `pre-commit install` or `pre-commit run`.

### Starting on an issue

After cloning and completing [Setup](#setup):

```bash
git pull origin main
git checkout -b feature/name   # use your issue number
```

| Issue | Folder | What to build |
|-------|--------|---------------|
| #7 | `src/safe_exam/capture/` | Webcam capture loop |
| #8 | `src/safe_exam/detectors/object/` | Phone detection (YOLO) |
| #9 | `src/safe_exam/detectors/object/` | Person count (YOLO) |
| #10 | `src/safe_exam/detectors/face_gaze/` | Gaze estimation (MediaPipe head + iris) |
| #11 | `src/safe_exam/processor/` | Unified `process_frame()` → `FrameResult` |
| #12 | `docs/experiments/phone-calibration/` | Phone threshold calibration |

Put shared helpers (config, logging) in `src/safe_exam/utils/`. Do not push directly to `main`.

### Opening a pull request

When your branch is ready, open a PR to `main` and include:

1. **What changed** — brief summary and `Closes #N` (auto-closes the issue on merge)
2. **How to test** — steps another teammate can follow
3. **Checklist** — pre-commit passes (`pre-commit run --all-files`) or CI is green

## Milestones

| Phase | Focus |
|-------|-------|
| **Phase 0** (current) | Local detection prototype, threshold tuning, CPU profiling, go/no-go report |
| **Phase 1** (planned) | SEB integration, flag streaming, server-side recording, professor review UI |

See [GitHub Issues](https://github.com/MarcoMll/safe-exam/issues) for the full task breakdown.

## License

TBD.
