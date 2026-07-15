# acts as a single source of truth for project navigation
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def find_project_root(
    start: Path | None = None, markers: Iterable[str] = ("pyproject.toml", ".git")
) -> Path:
    p = (start or Path(__file__).resolve()).parent
    for candidate in (p, *p.parents):
        if any((candidate / m).exists() for m in markers):
            return candidate
    raise RuntimeError(f"Could not find project root (looked for: {list(markers)}).")


@dataclass(frozen=True)
class ProjectPaths:
    ROOT: Path
    MODELS_DIR: Path
    DOCS_DIR: Path
    PHONE_CALIBRATION_RESULTS_DIR: Path


def get_paths() -> ProjectPaths:
    root = find_project_root()
    docs = root / "docs"
    return ProjectPaths(
        ROOT=root,
        MODELS_DIR=root / "models",
        DOCS_DIR=docs,
        PHONE_CALIBRATION_RESULTS_DIR=docs
        / "experiments"
        / "phone-calibration"
        / "results",
    )


def verify_paths():
    paths = get_paths()
    required_dirs = (
        paths.MODELS_DIR,
        paths.DOCS_DIR,
        paths.PHONE_CALIBRATION_RESULTS_DIR,
    )
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def file_exists(file_path: Path | str) -> bool:

    file_path = Path(file_path)

    if not file_path.is_file():
        return False

    return True
