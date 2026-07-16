"""Offline analysis for phone calibration: summarize CSV results."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from safe_exam.utils.paths_initializer import get_paths

CSV_FIELDS = (
    "recorded_at",
    "experiment",
    "label",
    "frame_index",
    "phone_confidence",
)

DEFAULT_SUMMARY_THRESHOLDS = [0.25, 0.35, 0.45, 0.5, 0.55, 0.6]


@dataclass(frozen=True)
class ScenarioSummary:
    label: str
    frame_count: int
    mean_confidence: float
    max_confidence: float
    detection_rates: dict[float, float]


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def slugify_experiment(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    slug = slug.strip("_")
    return slug or "experiment"


def default_output_path(experiment: str) -> Path:
    paths = get_paths()
    return (
        paths.PHONE_CALIBRATION_RESULTS_DIR
        / slugify_experiment(experiment)
        / "phone_calibration.csv"
    )


def detection_rate(confidences: list[float], threshold: float) -> float:
    if not confidences:
        return 0.0
    hits = sum(1 for value in confidences if value >= threshold)
    return hits / len(confidences)


def summarize_scenario(
    label: str,
    confidences: list[float],
    thresholds: list[float],
) -> ScenarioSummary:
    return ScenarioSummary(
        label=label,
        frame_count=len(confidences),
        mean_confidence=sum(confidences) / len(confidences),
        max_confidence=max(confidences),
        detection_rates={
            threshold: detection_rate(confidences, threshold)
            for threshold in thresholds
        },
    )


def format_summary(summary: ScenarioSummary, thresholds: list[float]) -> str:
    rate_lines = [
        f"  detection_rate @ {threshold:.2f}: {summary.detection_rates[threshold]:.0%}"
        for threshold in thresholds
    ]
    return (
        f"=== {summary.label} ===\n"
        f"  frames: {summary.frame_count}\n"
        f"  mean_confidence: {summary.mean_confidence:.3f}\n"
        f"  max_confidence: {summary.max_confidence:.3f}\n" + "\n".join(rate_lines)
    )


def format_table(summaries: list[ScenarioSummary], thresholds: list[float]) -> str:
    label_width = max(len("label"), *(len(summary.label) for summary in summaries))
    threshold_headers = " ".join(f"@ {t:.2f}".rjust(8) for t in thresholds)
    header = (
        f"{'label':<{label_width}} {'mean':>6} {'max':>6} {'frames':>6} "
        f"{threshold_headers}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        rates = " ".join(f"{summary.detection_rates[t]:>7.0%}" for t in thresholds)
        lines.append(
            f"{summary.label:<{label_width}} "
            f"{summary.mean_confidence:>6.3f} "
            f"{summary.max_confidence:>6.3f} "
            f"{summary.frame_count:>6} "
            f"{rates}"
        )
    return "\n".join(lines)


def load_csv_summaries(
    csv_path: Path,
    thresholds: list[float],
) -> tuple[str | None, list[ScenarioSummary]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_label: dict[str, list[float]] = defaultdict(list)
    label_order: list[str] = []
    experiment: str | None = None
    for row in rows:
        label = row["label"]
        if label not in by_label:
            label_order.append(label)
        by_label[label].append(float(row["phone_confidence"]))
        if experiment is None:
            experiment = row.get("experiment") or None

    summaries = [
        summarize_scenario(label, by_label[label], thresholds) for label in label_order
    ]
    return experiment, summaries


def format_tradeoff_table(
    summaries: list[ScenarioSummary],
    thresholds: list[float],
) -> str:
    phone = [s for s in summaries if s.label.startswith("phone_")]
    nophone = [s for s in summaries if s.label.startswith("nophone_")]
    header = f"{'threshold':>10} {'FP':>8} {'TP':>8}"
    lines = [header, "-" * len(header)]
    for threshold in thresholds:
        if nophone:
            fp_hits = sum(
                round(s.detection_rates[threshold] * s.frame_count) for s in nophone
            )
            fp_total = sum(s.frame_count for s in nophone)
            fp = fp_hits / fp_total if fp_total else 0.0
        else:
            fp = 0.0
        if phone:
            tp_hits = sum(
                round(s.detection_rates[threshold] * s.frame_count) for s in phone
            )
            tp_total = sum(s.frame_count for s in phone)
            tp = tp_hits / tp_total if tp_total else 0.0
        else:
            tp = 0.0
        lines.append(f"{threshold:>10.2f} {fp:>7.1%} {tp:>7.1%}")
    return "\n".join(lines)


def print_csv_summary(csv_path: Path, thresholds: list[float]) -> None:
    experiment, summaries = load_csv_summaries(csv_path, thresholds)
    if not summaries:
        print(f"No rows found in {csv_path}")
        return

    title = experiment or csv_path.parent.name
    print(f"\nCSV summary: {title}")
    print(f"Source: {csv_path}")
    print(f"Scenarios: {len(summaries)}")
    print()
    print(format_table(summaries, thresholds))
    print()
    print("Threshold tradeoff (nophone_ = FP, phone_ = TP)")
    print(format_tradeoff_table(summaries, thresholds))


def append_rows(
    output_path: Path,
    *,
    experiment: str,
    label: str,
    confidences: list[float],
    recorded_at: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for index, confidence in enumerate(confidences, start=1):
            writer.writerow(
                {
                    "recorded_at": recorded_at,
                    "experiment": experiment,
                    "label": label,
                    "frame_index": index,
                    "phone_confidence": f"{confidence:.4f}",
                }
            )
