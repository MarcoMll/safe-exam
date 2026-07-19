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
├── .github/workflows/        # CI (lint + pytest on pull requests)
├── docs/
│   └── experiments/
│       ├── phone-calibration/  # Threshold findings + result CSVs (#12)
│       ├── gaze-calibration/   # Off-screen duration findings (#13)
│       ├── person-intrusion/   # Spatial intrusion policy findings (#14)
│       └── cpu-profiling/      # CPU/RAM profiling findings (#15)
├── models/                   # Downloaded model weights (not committed)
├── scripts/                  # Standalone tools
│   ├── detector_test.py      # One-off YOLO demo
│   ├── face_gaze_demo.py     # One-off face-gaze demo
│   └── experiments/          # Durable experiment tools (kept long-term)
│       ├── phone_calibration/
│       │   ├── __main__.py   # CLI entrypoint
│       │   ├── record.py     # Live capture session
│       │   └── analyze.py    # Summarize (no camera)
│       ├── gaze_calibration/
│       │   ├── __main__.py   # CLI entrypoint
│       │   ├── record.py     # Live capture session
│       │   └── analyze.py    # Summarize + backtest (no camera)
│       ├── person_intrusion/
│       │   ├── __main__.py   # CLI entrypoint
│       │   ├── record.py     # Live capture session
│       │   └── analyze.py    # Summarize + backtest (no camera)
│       └── cpu_profile/
│           ├── __main__.py   # CLI entrypoint
│           └── profile.py    # Timed CPU/RAM sampling loop
├── src/safe_exam/            # Main application package
│   ├── capture/              # Webcam capture (#7)
│   │   ├── capture_config.py
│   │   └── capture.py
│   ├── detectors/            # Detection modules (#8–#10)
│   │   ├── object/           # YOLO — phone + person
│   │   │   ├── config.py
│   │   │   ├── results.py    # DetectedBox
│   │   │   ├── detector.py
│   │   │   └── overlay.py
│   │   └── face_gaze/        # MediaPipe — head pose + iris gaze
│   │       ├── config.py
│   │       ├── results.py    # FaceGazeResult
│   │       ├── detector.py
│   │       ├── overlay.py
│   │       └── iris_estimation.py  # internal helper
│   ├── processor/            # Unified frame processor (#11)
│   │   ├── frame_result.py   # FrameResult schema
│   │   ├── frame_processor.py
│   │   ├── attention_policy.py  # runtime off-center interpretation
│   │   ├── intrusion_policy.py  # spatial multi-person intrusion rules
│   │   ├── session_stats.py  # session counters + summaries
│   │   ├── runner.py         # live capture loop
│   │   └── debug_overlay.py  # composite debug view
│   └── utils/                # Shared helpers (logging, paths)
├── tests/                    # Unit tests (pytest; also run in CI)
│   └── test_intrusion_policy.py
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

Each detector subpackage follows the same layout: `config.py`, `results.py`, `detector.py`, `overlay.py`. The face gaze subpackage also has `iris_estimation.py` for internal iris math.

### Processor architecture

The `processor/` package has a few clear roles:

| Module | Role |
|--------|------|
| `frame_processor.py` | Run detectors and merge one `FrameResult` per frame |
| `attention_policy.py` | Choose which signal counts as off-center (`head` / `eye` / `gaze` / `iris`) |
| `intrusion_policy.py` | Spatial rules for multi-person intrusion (ROI / area / overlap) |
| `session_stats.py` | Aggregate session counters from processed frames |
| `runner.py` | Live capture loop (orchestration) |
| `__main__.py` | CLI entrypoint only |

Detectors **compute** raw signals. The processor **interprets** them via `attention_policy.py` and `intrusion_policy.py`, and **aggregates** them via `session_stats.py`. Calibration experiments under `docs/experiments/` help choose policy values; they do not change detector internals.

Phase 1 will add flag logic (duration streaks, pattern detection, professor-facing flags) on top of this layer — not inside individual detectors.

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
```

### Experiment calibration tools

Durable tools for threshold experiments (kept long-term). Findings live under `docs/experiments/`; the tools live under `scripts/experiments/`.

Run from the `scripts/` directory so the experiment packages are importable:

```bash
cd scripts

# Phone threshold calibration (#12)
python -m experiments.phone_calibration --experiment <name>
python -m experiments.phone_calibration --summarize

# Gaze off-screen calibration (#13)
python -m experiments.gaze_calibration --experiment <name>
python -m experiments.gaze_calibration --backtest
python -m experiments.gaze_calibration --summarize

# Person intrusion policy calibration (#14)
python -m experiments.person_intrusion --experiment <name>
python -m experiments.person_intrusion --backtest
python -m experiments.person_intrusion --summarize

# CPU / RAM profiling (#15)
# Drill-down (~30s/mode) then sustained 10-min run at production FPS
python -m experiments.cpu_profile --experiment <name> --mode all --target-fps 5 --duration 30
python -m experiments.cpu_profile --experiment <name> --mode both --target-fps 5 --duration 600
```

Calibration findings:

- Phone: [`docs/experiments/phone-calibration/`](docs/experiments/phone-calibration/)
- Gaze: [`docs/experiments/gaze-calibration/`](docs/experiments/gaze-calibration/)
- Person intrusion: [`docs/experiments/person-intrusion/`](docs/experiments/person-intrusion/)
- CPU profiling: [`docs/experiments/cpu-profiling/`](docs/experiments/cpu-profiling/)

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
python -m safe_exam.processor --camera-index 1 --target-fps 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--no-debug` | off | Headless run — no overlay window |
| `--camera-index` | `0` | Webcam device index |
| `--target-fps` | `5` | Target capture frame rate (use `10` for denser local debug; see [cpu-profiling](docs/experiments/cpu-profiling/)) |

Session stats (avg fps, inference time, detection counts, attention counters, intrusion counters) are logged every 60 frames and again when the processor stops.

### Attention signals (head, eye, gaze)

Each `FrameResult` from `process_frame()` is the processor's public per-frame contract:

**Object detection (YOLO)**

| Field | Meaning |
|-------|---------|
| `phone_detected`, `phone_confidence` | Phone present and best confidence |
| `person_count`, `person_boxes` | Raw YOLO person detections (count + bounding boxes) |
| `frame_width`, `frame_height` | Frame size used by spatial policies |

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

Person count is raw YOLO output. Whether a visible classmate counts as **intrusion** is decided by `intrusion_policy.py` and aggregated as `intrusion_suspected_frames` in session stats — not by `person_count > 1`. Subtle gaze cheating (head forward, eyes down) is tracked via `eye_pitch` / `eye_yaw` and `iris_offset_*`.

**Debug overlays:** per-detector drawing lives in each subpackage (`face_gaze/overlay.py`, `object/overlay.py`). The processor composes them in `processor/debug_overlay.py` for the full pipeline view.

**Session logging:** `processor/session_stats.py` tracks frame counts including raw per-signal counters (`head_off_center_frames`, `eye_off_center_frames`, `gaze_off_center_frames`), `attention_off_center_frames` for the active attention policy, and `intrusion_suspected_frames` for the spatial intrusion policy (logged every 60 frames).

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
pytest tests/
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
| #12 | `scripts/experiments/phone_calibration/` + `docs/experiments/phone-calibration/` | Phone threshold calibration |
| #13 | `scripts/experiments/gaze_calibration/` + `docs/experiments/gaze-calibration/` | Gaze off-screen duration calibration |
| #14 | `scripts/experiments/person_intrusion/` + `docs/experiments/person-intrusion/` | Person intrusion policy calibration |
| #15 | `scripts/experiments/cpu_profile/` + `docs/experiments/cpu-profiling/` | CPU/RAM profiling |

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
