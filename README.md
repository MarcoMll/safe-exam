# safe-exam

A modular proctoring layer for [Safe Exam Browser (SEB)](https://safeexambrowser.org/). The system uses webcam-based computer vision to detect suspicious behavior during exams — phone use, gaze off-screen, additional people in frame — and raises **flags with confidence scores** for professor review. It does not automatically remove students from exams.

**Current phase:** Phase 0 — local detection prototype. No SEB integration, server, or recording yet. The goal is to prove that detection runs reliably on student hardware before building infrastructure around it.

## What we're detecting

| Signal | Approach (Phase 0) |
|--------|-------------------|
| Phone in frame | YOLO (nano/small) |
| Person count | YOLO |
| Gaze direction | MediaPipe Face Mesh |

Detectors plug into a shared capture loop and are merged into a single `process_frame()` output for later signal fusion (Phase 1).

## Repository layout

```
safe-exam/
├── .github/workflows/        # CI (lint checks on pull requests)
├── data/
│   ├── raw/                  # Local test images/videos (not committed)
│   └── processed/            # Derived datasets, exports
├── docs/                     # Findings reports, architecture notes
├── models/                   # Downloaded model weights (not committed)
├── src/safe_exam/            # Main application package
│   ├── capture/              # Webcam capture loop (#7)
│   ├── detectors/            # Phone, person, gaze modules (#8–#10)
│   ├── processor/            # Unified frame processor (#11)
│   └── utils/                # Shared helpers (config, logging, etc.)
├── requirements.txt          # Runtime dependencies
├── requirements-dev.txt      # Dev tools (formatting, linting, hooks)
└── pyproject.toml            # Tool configuration (Black, Ruff)
```

`src/safe_exam/` is the main application package — all app code and imports live here (`from safe_exam.detectors.phone import ...`). Keep detector logic isolated in `detectors/` so each module can be developed and tested independently.

## Setup

Requires **Python 3.10+** and a working webcam for later issues.

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
pre-commit install
```

Verify the environment:

```bash
python -c "import cv2, mediapipe; from ultralytics import YOLO; print('Environment OK')"
```

On first YOLO run, model weights are downloaded automatically into `models/` (gitignored).

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
git checkout -b feature/8-phone-detection   # use your issue number
```

| Issue | Folder | What to build |
|-------|--------|---------------|
| #7 | `src/safe_exam/capture/` | Webcam capture loop |
| #8 | `src/safe_exam/detectors/` | Phone detection (YOLO) |
| #9 | `src/safe_exam/detectors/` | Person count (YOLO) |
| #10 | `src/safe_exam/detectors/` | Gaze estimation (MediaPipe) |
| #11 | `src/safe_exam/processor/` | Unified `process_frame()` |

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

See [GitHub Issues](https://github.com/MarcoMll/safe-exam/issues) for the full task breakdown. Issue #6 blocks all other Phase 0 work.

## License

TBD.
