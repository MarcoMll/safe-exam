"""Offline analysis for gaze calibration: summarize + backtest."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from safe_exam.utils.paths_initializer import get_paths

CSV_FIELDS = (
    "recorded_at",
    "experiment",
    "label",
    "frame_index",
    "elapsed_s",
    "face_detected",
    "head_pitch",
    "head_yaw",
    "eye_pitch",
    "eye_yaw",
    "gaze_pitch",
    "gaze_yaw",
    "iris_offset_x",
    "iris_offset_y",
)

DEFAULT_DURATION_THRESHOLDS = [4.0, 6.0, 8.0, 12.0]

OffCenterMode = Literal["both", "yaw_only", "pitch_only"]
SignalName = Literal["head", "eye", "gaze", "iris"]


@dataclass(frozen=True)
class AngleThresholds:
    pitch_deg: float
    yaw_deg: float
    mode: OffCenterMode = "both"


@dataclass(frozen=True)
class AnalysisConfig:
    """How to classify a frame as off-center during summarize / backtest."""

    signal: SignalName = "gaze"
    angles: AngleThresholds = field(
        default_factory=lambda: AngleThresholds(pitch_deg=10.0, yaw_deg=10.0)
    )
    iris_offset_thr: float | None = None
    gap_tolerance_s: float = 0.0


@dataclass(frozen=True)
class BacktestRow:
    signal: str
    mode: str
    pitch_thr: float
    yaw_thr: float
    iris_thr: float | None
    gap_tolerance_s: float
    duration_s: float
    writing_fires: bool
    reading_fires: bool
    natural_fp_count: int
    natural_total: int
    suspicious_tp_count: int
    suspicious_total: int
    best_suspicious_label: str
    best_suspicious_streak_s: float
    score: float


@dataclass
class FrameSample:
    elapsed_s: float
    face_detected: bool
    head_pitch: float
    head_yaw: float
    eye_pitch: float
    eye_yaw: float
    gaze_pitch: float
    gaze_yaw: float
    iris_offset_x: float
    iris_offset_y: float


def slugify_experiment(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    slug = slug.strip("_")
    return slug or "experiment"


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one number.")
    return values


def default_output_path(experiment: str) -> Path:
    paths = get_paths()
    return (
        paths.GAZE_CALIBRATION_RESULTS_DIR
        / slugify_experiment(experiment)
        / "gaze_calibration.csv"
    )


def _signal_angles(sample: FrameSample, signal: str) -> tuple[float, float]:
    if signal == "head":
        return sample.head_pitch, sample.head_yaw
    if signal == "eye":
        return sample.eye_pitch, sample.eye_yaw
    if signal == "gaze":
        return sample.gaze_pitch, sample.gaze_yaw
    raise ValueError(f"Unknown signal: {signal}")


def is_off_center(pitch: float, yaw: float, pitch_thr: float, yaw_thr: float) -> bool:
    return abs(pitch) > pitch_thr or abs(yaw) > yaw_thr


def is_off_center_mode(
    pitch: float,
    yaw: float,
    *,
    pitch_thr: float,
    yaw_thr: float,
    mode: OffCenterMode,
) -> bool:
    if mode == "yaw_only":
        return abs(yaw) > yaw_thr
    if mode == "pitch_only":
        return abs(pitch) > pitch_thr
    return is_off_center(pitch, yaw, pitch_thr, yaw_thr)


def is_sample_off(sample: FrameSample, config: AnalysisConfig) -> bool:
    if not sample.face_detected:
        return False

    if config.signal == "iris":
        if config.iris_offset_thr is None:
            raise ValueError("iris_offset_thr required for signal=iris")
        return (
            abs(sample.iris_offset_x) > config.iris_offset_thr
            or abs(sample.iris_offset_y) > config.iris_offset_thr
        )

    pitch, yaw = _signal_angles(sample, config.signal)
    return is_off_center_mode(
        pitch,
        yaw,
        pitch_thr=config.angles.pitch_deg,
        yaw_thr=config.angles.yaw_deg,
        mode=config.angles.mode,
    )


def max_off_center_streak_s(
    samples: list[FrameSample],
    config: AnalysisConfig,
) -> float:
    """Longest consecutive off-center streak in seconds."""
    if len(samples) < 2:
        return 0.0

    best = 0.0
    streak_start: float | None = None
    gap_start: float | None = None

    for sample in samples:
        off = is_sample_off(sample, config)
        if off:
            gap_start = None
            if streak_start is None:
                streak_start = sample.elapsed_s
            best = max(best, sample.elapsed_s - streak_start)
            continue

        if not sample.face_detected and config.gap_tolerance_s > 0:
            if gap_start is None:
                gap_start = sample.elapsed_s
            elif sample.elapsed_s - gap_start <= config.gap_tolerance_s:
                continue
            gap_start = None

        streak_start = None

    return best


def off_center_fraction(samples: list[FrameSample], config: AnalysisConfig) -> float:
    if not samples:
        return 0.0
    hits = sum(1 for sample in samples if is_sample_off(sample, config))
    return hits / len(samples)


def _legacy_analysis_config(
    *,
    signal: str,
    pitch_thr: float,
    yaw_thr: float,
    mode: OffCenterMode = "both",
    iris_offset_thr: float | None = None,
    gap_tolerance_s: float = 0.0,
) -> AnalysisConfig:
    return AnalysisConfig(
        signal=signal,  # type: ignore[arg-type]
        angles=AngleThresholds(pitch_deg=pitch_thr, yaw_deg=yaw_thr, mode=mode),
        iris_offset_thr=iris_offset_thr,
        gap_tolerance_s=gap_tolerance_s,
    )


def append_rows(
    output_path: Path,
    *,
    experiment: str,
    label: str,
    samples: list[FrameSample],
    recorded_at: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for index, sample in enumerate(samples, start=1):
            writer.writerow(
                {
                    "recorded_at": recorded_at,
                    "experiment": experiment,
                    "label": label,
                    "frame_index": index,
                    "elapsed_s": f"{sample.elapsed_s:.4f}",
                    "face_detected": int(sample.face_detected),
                    "head_pitch": f"{sample.head_pitch:.4f}",
                    "head_yaw": f"{sample.head_yaw:.4f}",
                    "eye_pitch": f"{sample.eye_pitch:.4f}",
                    "eye_yaw": f"{sample.eye_yaw:.4f}",
                    "gaze_pitch": f"{sample.gaze_pitch:.4f}",
                    "gaze_yaw": f"{sample.gaze_yaw:.4f}",
                    "iris_offset_x": f"{sample.iris_offset_x:.4f}",
                    "iris_offset_y": f"{sample.iris_offset_y:.4f}",
                }
            )


def load_csv_by_label(
    csv_path: Path,
) -> tuple[str | None, dict[str, list[FrameSample]]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_label: dict[str, list[FrameSample]] = defaultdict(list)
    label_order: list[str] = []
    experiment: str | None = None
    for row in rows:
        label = row["label"]
        if label not in by_label:
            label_order.append(label)
        if experiment is None:
            experiment = row.get("experiment") or None
        by_label[label].append(
            FrameSample(
                elapsed_s=float(row["elapsed_s"]),
                face_detected=bool(int(float(row["face_detected"]))),
                head_pitch=float(row["head_pitch"]),
                head_yaw=float(row["head_yaw"]),
                eye_pitch=float(row["eye_pitch"]),
                eye_yaw=float(row["eye_yaw"]),
                gaze_pitch=float(row["gaze_pitch"]),
                gaze_yaw=float(row["gaze_yaw"]),
                iris_offset_x=float(row["iris_offset_x"]),
                iris_offset_y=float(row["iris_offset_y"]),
            )
        )
    # Preserve insertion order via label_order
    ordered = {label: by_label[label] for label in label_order}
    return experiment, ordered


def print_csv_summary(
    csv_path: Path,
    *,
    config: AnalysisConfig,
    duration_thresholds: list[float],
    signals: list[str] | None = None,
) -> None:
    experiment, by_label = load_csv_by_label(csv_path)
    if not by_label:
        print(f"No rows found in {csv_path}")
        return

    if signals is None:
        signals = ["gaze", "eye", "head", "iris"]

    title = experiment or csv_path.parent.name
    print(f"\nCSV summary: {title}")
    print(f"Source: {csv_path}")
    _print_config_line(config)
    print(f"Duration thresholds (s): {duration_thresholds}")
    print()

    dur_headers = " ".join(f"@{d:.0f}s".rjust(6) for d in duration_thresholds)
    header = (
        f"{'label':<36} {'sig':<5} {'frames':>6} {'off%':>6} "
        f"{'max_off_s':>9} {dur_headers}"
    )
    print(header)
    print("-" * len(header))

    for label, samples in by_label.items():
        for signal in signals:
            run_config = _config_for_signal(config, signal)
            off_frac = off_center_fraction(samples, run_config)
            max_streak = max_off_center_streak_s(samples, run_config)
            fires = " ".join(
                f"{'Y' if max_streak >= duration else 'n':>6}"
                for duration in duration_thresholds
            )
            print(
                f"{label:<36} {signal:<5} {len(samples):>6} "
                f"{off_frac:>5.0%} {max_streak:>9.2f} {fires}"
            )

    print()
    print("Interpretation:")
    print("  Raw off-center streaks feed future flag logic, not flags themselves.")
    print("  Compare yaw_only / eye / iris for side glances vs pitch-heavy writing.")
    print("  Use --backtest to sweep configs on this CSV without re-recording.")


def _config_for_signal(base: AnalysisConfig, signal: str) -> AnalysisConfig:
    if signal == "iris":
        thr = base.iris_offset_thr if base.iris_offset_thr is not None else 0.12
        return AnalysisConfig(
            signal="iris",
            angles=base.angles,
            iris_offset_thr=thr,
            gap_tolerance_s=base.gap_tolerance_s,
        )
    return AnalysisConfig(
        signal=signal,  # type: ignore[arg-type]
        angles=base.angles,
        iris_offset_thr=base.iris_offset_thr,
        gap_tolerance_s=base.gap_tolerance_s,
    )


def _print_config_line(config: AnalysisConfig) -> None:
    if config.signal == "iris":
        print(f"Signal: iris  offset_thr={config.iris_offset_thr}")
    else:
        print(
            f"Signal: {config.signal}  mode={config.angles.mode}  "
            f"pitch={config.angles.pitch_deg:.1f}  yaw={config.angles.yaw_deg:.1f}"
        )
    if config.gap_tolerance_s > 0:
        print(f"Gap tolerance (blinks): {config.gap_tolerance_s:.2f}s")


def evaluate_backtest_row(
    by_label: dict[str, list[FrameSample]],
    config: AnalysisConfig,
    duration_s: float,
) -> BacktestRow:
    natural_labels = [label for label in by_label if label.startswith("natural_")]
    suspicious_labels = [label for label in by_label if label.startswith("suspicious_")]

    writing_streak = max_off_center_streak_s(
        by_label.get("natural_writing", []), config
    )
    reading_streak = max_off_center_streak_s(
        by_label.get("natural_reading_paper", []), config
    )
    writing_fires = writing_streak >= duration_s
    reading_fires = reading_streak >= duration_s

    natural_fp = 0
    for label in natural_labels:
        streak = max_off_center_streak_s(by_label[label], config)
        if streak >= duration_s:
            natural_fp += 1

    suspicious_tp = 0
    best_label = ""
    best_streak = 0.0
    for label in suspicious_labels:
        streak = max_off_center_streak_s(by_label[label], config)
        if streak >= duration_s:
            suspicious_tp += 1
        if streak > best_streak:
            best_streak = streak
            best_label = label

    # Higher is better: catch suspicious, avoid natural FP, especially writing/reading
    score = (
        suspicious_tp * 10
        + best_streak
        - natural_fp * 5
        - (8 if writing_fires else 0)
        - (6 if reading_fires else 0)
    )

    return BacktestRow(
        signal=config.signal,
        mode=config.angles.mode,
        pitch_thr=config.angles.pitch_deg,
        yaw_thr=config.angles.yaw_deg,
        iris_thr=config.iris_offset_thr,
        gap_tolerance_s=config.gap_tolerance_s,
        duration_s=duration_s,
        writing_fires=writing_fires,
        reading_fires=reading_fires,
        natural_fp_count=natural_fp,
        natural_total=len(natural_labels),
        suspicious_tp_count=suspicious_tp,
        suspicious_total=len(suspicious_labels),
        best_suspicious_label=best_label,
        best_suspicious_streak_s=best_streak,
        score=score,
    )


def pick_conservative_profile(rows: list[BacktestRow]) -> BacktestRow | None:
    """Highest score with writing_fires=False (issue #13 style)."""
    candidates = [row for row in rows if not row.writing_fires]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.score)


def pick_sensitive_profile(rows: list[BacktestRow]) -> BacktestRow | None:
    """Highest suspicious recall among writing-safe configs."""
    candidates = [row for row in rows if not row.writing_fires]
    if not candidates:
        return None
    signal_order = {"gaze": 0, "head": 1, "eye": 2, "iris": 3}

    def sort_key(row: BacktestRow) -> tuple:
        return (
            row.suspicious_tp_count,
            row.best_suspicious_streak_s,
            -row.natural_fp_count,
            -signal_order.get(row.signal, 9),
            -row.duration_s,
        )

    return max(candidates, key=sort_key)


def print_profile(name: str, row: BacktestRow | None, *, description: str) -> None:
    print(f"\n--- {name} ---")
    print(description)
    if row is None:
        print("(no matching config)")
        return
    iris = f"{row.iris_thr:.2f}" if row.iris_thr is not None else "n/a"
    print(
        f"  signal={row.signal}  mode={row.mode}  "
        f"pitch={row.pitch_thr:.0f}  yaw={row.yaw_thr:.0f}  iris={iris}"
    )
    print(
        f"  duration={row.duration_s:.0f}s  gap_tolerance={row.gap_tolerance_s:.1f}s  "
        f"score={row.score:.1f}"
    )
    print(
        f"  writing={'FIRES' if row.writing_fires else 'ok'}  "
        f"reading={'FIRES' if row.reading_fires else 'ok'}  "
        f"natural_fp={row.natural_fp_count}/{row.natural_total}  "
        f"suspicious_tp={row.suspicious_tp_count}/{row.suspicious_total}"
    )
    print(
        f"  longest suspicious streak: {row.best_suspicious_streak_s:.1f}s "
        f"({row.best_suspicious_label})"
    )


def print_scoring_formula() -> None:
    print("\nBacktest score formula (per config × duration):")
    print("  score = suspicious_tp×10 + best_suspicious_streak_s")
    print("          - natural_fp×5 - (8 if writing fires) - (6 if reading fires)")
    print("  suspicious_tp = count of suspicious_* scenarios with max_off >= duration")
    print("  natural_fp    = count of natural_* scenarios with max_off >= duration")


def print_recommended_profiles(rows: list[BacktestRow]) -> None:
    print_scoring_formula()
    conservative = pick_conservative_profile(rows)
    sensitive = pick_sensitive_profile(rows)
    print_profile(
        "Profile A — Conservative (best score, writing-safe)",
        conservative,
        description=(
            "Optimizes the backtest score while keeping natural_writing from "
            "firing. Fewer raw signals; closer to issue #13 acceptance style."
        ),
    )
    print_profile(
        "Profile B — Sensitive (high suspicious recall)",
        sensitive,
        description=(
            "Maximizes suspicious_* firing at the chosen duration. More natural "
            "false positives expected — pair with stricter Phase 1 flag logic."
        ),
    )
    print("\nPhase 0 recommendation:")
    print(
        "  Use Profile A if you want a simple duration rule with fewer professor flags."
    )
    print("  Use Profile B if you want raw signal + pattern/duration logic in Phase 1.")


def run_backtest(
    csv_path: Path,
    *,
    gap_tolerance_s: float = 0.4,
    top_n: int = 15,
    output_path: Path | None = None,
) -> list[BacktestRow]:
    _, by_label = load_csv_by_label(csv_path)
    if not by_label:
        print(f"No rows found in {csv_path}")
        return []

    duration_thresholds = [4.0, 6.0, 8.0, 12.0]
    yaw_thresholds = [5.0, 8.0, 10.0, 12.0, 15.0]
    eye_yaw_thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    pitch_thresholds_high = [20.0, 25.0, 30.0, 99.0]
    iris_thresholds = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]

    rows: list[BacktestRow] = []

    for duration_s in duration_thresholds:
        for yaw_thr in yaw_thresholds:
            for pitch_thr in pitch_thresholds_high:
                for mode in ("yaw_only", "both"):
                    for signal in ("gaze", "head"):
                        config = AnalysisConfig(
                            signal=signal,  # type: ignore[arg-type]
                            angles=AngleThresholds(
                                pitch_deg=pitch_thr,
                                yaw_deg=yaw_thr,
                                mode=mode,  # type: ignore[arg-type]
                            ),
                            gap_tolerance_s=gap_tolerance_s,
                        )
                        rows.append(evaluate_backtest_row(by_label, config, duration_s))

        for eye_yaw in eye_yaw_thresholds:
            for eye_pitch in (99.0, 15.0, 10.0):
                for mode in ("yaw_only", "both"):
                    config = AnalysisConfig(
                        signal="eye",
                        angles=AngleThresholds(
                            pitch_deg=eye_pitch,
                            yaw_deg=eye_yaw,
                            mode=mode,  # type: ignore[arg-type]
                        ),
                        gap_tolerance_s=gap_tolerance_s,
                    )
                    rows.append(evaluate_backtest_row(by_label, config, duration_s))

        for iris_thr in iris_thresholds:
            config = AnalysisConfig(
                signal="iris",
                angles=AngleThresholds(pitch_deg=99.0, yaw_deg=99.0),
                iris_offset_thr=iris_thr,
                gap_tolerance_s=gap_tolerance_s,
            )
            rows.append(evaluate_backtest_row(by_label, config, duration_s))

    rows.sort(key=lambda row: row.score, reverse=True)
    top = rows[:top_n]

    print(f"\nBacktest: {csv_path}")
    print(f"Scenarios: {len(by_label)}  gap_tolerance={gap_tolerance_s:.2f}s")
    print(f"Configs evaluated: {len(rows)}  showing top {len(top)}")
    print()
    header = (
        f"{'rank':>4} {'score':>6} {'dur':>4} {'sig':<5} {'mode':<9} "
        f"{'p':>4} {'y':>4} {'iris':>5} {'write':>5} {'read':>5} "
        f"{'natFP':>5} {'susTP':>5} {'best_s':>6} best_label"
    )
    print(header)
    print("-" * len(header))
    for index, row in enumerate(top, start=1):
        iris = f"{row.iris_thr:.2f}" if row.iris_thr is not None else "  -"
        print(
            f"{index:>4} {row.score:>6.1f} {row.duration_s:>4.0f} "
            f"{row.signal:<5} {row.mode:<9} {row.pitch_thr:>4.0f} {row.yaw_thr:>4.0f} "
            f"{iris:>5} {'Y' if row.writing_fires else 'n':>5} "
            f"{'Y' if row.reading_fires else 'n':>5} "
            f"{row.natural_fp_count}/{row.natural_total:>2} "
            f"{row.suspicious_tp_count}/{row.suspicious_total:>2} "
            f"{row.best_suspicious_streak_s:>6.2f} {row.best_suspicious_label}"
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "score",
                    "duration_s",
                    "signal",
                    "mode",
                    "pitch_thr",
                    "yaw_thr",
                    "iris_thr",
                    "gap_tolerance_s",
                    "writing_fires",
                    "reading_fires",
                    "natural_fp_count",
                    "natural_total",
                    "suspicious_tp_count",
                    "suspicious_total",
                    "best_suspicious_streak_s",
                    "best_suspicious_label",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "score": f"{row.score:.2f}",
                        "duration_s": row.duration_s,
                        "signal": row.signal,
                        "mode": row.mode,
                        "pitch_thr": row.pitch_thr,
                        "yaw_thr": row.yaw_thr,
                        "iris_thr": row.iris_thr if row.iris_thr is not None else "",
                        "gap_tolerance_s": row.gap_tolerance_s,
                        "writing_fires": int(row.writing_fires),
                        "reading_fires": int(row.reading_fires),
                        "natural_fp_count": row.natural_fp_count,
                        "natural_total": row.natural_total,
                        "suspicious_tp_count": row.suspicious_tp_count,
                        "suspicious_total": row.suspicious_total,
                        "best_suspicious_streak_s": (
                            f"{row.best_suspicious_streak_s:.2f}"
                        ),
                        "best_suspicious_label": row.best_suspicious_label,
                    }
                )
        print(f"\nWrote full backtest grid to {output_path}")

    print_recommended_profiles(rows)

    return top
