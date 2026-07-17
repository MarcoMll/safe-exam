"""Offline analysis for person intrusion calibration."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from safe_exam.detectors.object import DetectedBox
from safe_exam.processor.frame_result import FrameResult
from safe_exam.processor.intrusion_policy import (
    DEFAULT_INTRUSION_POLICY,
    IntrusionPolicyConfig,
    is_intrusion_suspected_for_frame,
)

CSV_FIELDS = (
    "recorded_at",
    "experiment",
    "label",
    "frame_index",
    "frame_width",
    "frame_height",
    "person_count",
    "primary_area_pct",
    "max_secondary_area_pct",
    "max_secondary_iou",
    "any_secondary_in_roi",
    "intrusion_suspected",
    "person_boxes_json",
)

DEFAULT_ROI_FRACTIONS = [0.5, 0.6, 0.7]
DEFAULT_AREA_PCTS = [0.03, 0.05, 0.08]
DEFAULT_IOU_THRESHOLDS = [0.05, 0.10, 0.15]
DEFAULT_RULES_TO_MATCH = [2, 3]


@dataclass(frozen=True)
class ScenarioSummary:
    """Summary of a scenario."""

    label: str
    frame_count: int
    intrusion_rate: float
    mean_person_count: float


@dataclass(frozen=True)
class BacktestRow:
    """Row of a backtest."""

    roi_center_fraction: float
    min_secondary_area_pct: float
    primary_overlap_iou: float
    min_rules_to_match: int
    negative_fp_count: int
    negative_total: int
    intrusion_tp_count: int
    intrusion_total: int
    score: float


def slugify_experiment(name: str) -> str:
    """Slugify an experiment name."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    slug = slug.strip("_")
    return slug or "experiment"


def default_output_path(experiment: str) -> Path:
    """Default output path for a person intrusion calibration experiment."""
    from safe_exam.utils.paths_initializer import get_paths

    paths = get_paths()
    return (
        paths.PERSON_INTRUSION_CALIBRATION_RESULTS_DIR
        / slugify_experiment(experiment)
        / "person_intrusion.csv"
    )


def boxes_to_json(boxes: list[DetectedBox]) -> str:
    """Convert a list of detected boxes to a JSON string."""
    payload = [
        {
            "x1": box.x1,
            "y1": box.y1,
            "x2": box.x2,
            "y2": box.y2,
            "confidence": box.confidence,
            "class_id": box.class_id,
        }
        for box in boxes
    ]
    return json.dumps(payload)


def boxes_from_json(raw: str) -> list[DetectedBox]:
    """Convert a JSON string to a list of detected boxes."""
    if not raw:
        return []
    payload = json.loads(raw)
    return [
        DetectedBox(
            x1=float(item["x1"]),
            y1=float(item["y1"]),
            x2=float(item["x2"]),
            y2=float(item["y2"]),
            confidence=float(item["confidence"]),
            class_id=int(item["class_id"]),
        )
        for item in payload
    ]


def frame_result_from_row(row: dict) -> FrameResult:
    """Convert a row of a CSV file to a frame result."""
    boxes = boxes_from_json(row.get("person_boxes_json") or "")
    return FrameResult(
        person_count=int(row["person_count"]),
        person_boxes=boxes,
        frame_width=int(row["frame_width"]),
        frame_height=int(row["frame_height"]),
    )


def intrusion_rate_for_rows(rows: list[dict], policy: IntrusionPolicyConfig) -> float:
    """Calculate the intrusion rate for a list of rows."""
    if not rows:
        return 0.0
    hits = sum(
        1
        for row in rows
        if is_intrusion_suspected_for_frame(frame_result_from_row(row), policy)
    )
    return hits / len(rows)


def summarize_scenario(label: str, rows: list[dict]) -> ScenarioSummary:
    """Summarize a scenario."""
    policy = DEFAULT_INTRUSION_POLICY
    person_counts = [int(row["person_count"]) for row in rows]
    return ScenarioSummary(
        label=label,
        frame_count=len(rows),
        intrusion_rate=intrusion_rate_for_rows(rows, policy),
        mean_person_count=sum(person_counts) / len(person_counts),
    )


def format_summary(summary: ScenarioSummary) -> str:
    """Format a summary of a scenario."""
    return (
        f"=== {summary.label} ===\n"
        f"  frames: {summary.frame_count}\n"
        f"  mean_person_count: {summary.mean_person_count:.2f}\n"
        f"  intrusion_rate @ default policy: {summary.intrusion_rate:.0%}"
    )


def format_table(summaries: list[ScenarioSummary]) -> str:
    """Format a table of summaries."""
    label_width = max(len("label"), *(len(summary.label) for summary in summaries))
    header = f"{'label':<{label_width}} {'frames':>6} {'persons':>8} {'intrusion':>10}"
    lines = [header, "-" * len(header)]
    for summary in summaries:
        lines.append(
            f"{summary.label:<{label_width}} "
            f"{summary.frame_count:>6} "
            f"{summary.mean_person_count:>8.2f} "
            f"{summary.intrusion_rate:>9.0%}"
        )
    return "\n".join(lines)


def load_csv_rows(csv_path: Path) -> tuple[str | None, dict[str, list[dict]]]:
    """Load rows from a CSV file."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_label: dict[str, list[dict]] = defaultdict(list)
    label_order: list[str] = []
    experiment: str | None = None
    for row in rows:
        label = row["label"]
        if label not in by_label:
            label_order.append(label)
        by_label[label].append(row)
        if experiment is None:
            experiment = row.get("experiment") or None

    ordered = {label: by_label[label] for label in label_order}
    return experiment, ordered


def print_csv_summary(csv_path: Path) -> None:
    """Print a summary of a CSV file."""
    experiment, grouped = load_csv_rows(csv_path)
    if not grouped:
        print(f"No rows found in {csv_path}")
        return

    summaries = [summarize_scenario(label, rows) for label, rows in grouped.items()]
    title = experiment or csv_path.parent.name
    print(f"\nCSV summary: {title}")
    print(f"Source: {csv_path}")
    print(f"Scenarios: {len(summaries)}")
    print()
    print(format_table(summaries))
    print()
    print("Scenario details:")
    for summary in summaries:
        print(format_summary(summary))
        print()


def _scenario_groups(labels: list[str]) -> tuple[list[str], list[str]]:
    """Group scenarios into negative and positive groups."""
    negative = [
        label
        for label in labels
        if label.startswith("solo_") or label.startswith("background_")
    ]
    positive = [label for label in labels if label.startswith("intrusion_")]
    return negative, positive


def _score_backtest(fp_rate: float, tp_rate: float) -> float:
    """Score a backtest."""
    return tp_rate - (2.0 * fp_rate)


def run_backtest(
    csv_path: Path,
    *,
    top_n: int = 15,
    output_path: Path | None = None,
) -> list[BacktestRow]:
    """Run a backtest."""
    _, grouped = load_csv_rows(csv_path)
    if not grouped:
        print(f"No rows found in {csv_path}")
        return []

    labels = list(grouped.keys())
    negative_labels, positive_labels = _scenario_groups(labels)

    results: list[BacktestRow] = []
    for roi, area, iou, rules in product(
        DEFAULT_ROI_FRACTIONS,
        DEFAULT_AREA_PCTS,
        DEFAULT_IOU_THRESHOLDS,
        DEFAULT_RULES_TO_MATCH,
    ):
        policy = IntrusionPolicyConfig(
            roi_center_fraction=roi,
            min_secondary_area_pct=area,
            primary_overlap_iou=iou,
            min_rules_to_match=rules,
        )
        negative_rows = [
            row for label in negative_labels for row in grouped.get(label, [])
        ]
        positive_rows = [
            row for label in positive_labels for row in grouped.get(label, [])
        ]
        fp_count = sum(
            1
            for row in negative_rows
            if is_intrusion_suspected_for_frame(frame_result_from_row(row), policy)
        )
        tp_count = sum(
            1
            for row in positive_rows
            if is_intrusion_suspected_for_frame(frame_result_from_row(row), policy)
        )
        fp_rate = fp_count / len(negative_rows) if negative_rows else 0.0
        tp_rate = tp_count / len(positive_rows) if positive_rows else 0.0
        results.append(
            BacktestRow(
                roi_center_fraction=roi,
                min_secondary_area_pct=area,
                primary_overlap_iou=iou,
                min_rules_to_match=rules,
                negative_fp_count=fp_count,
                negative_total=len(negative_rows),
                intrusion_tp_count=tp_count,
                intrusion_total=len(positive_rows),
                score=_score_backtest(fp_rate, tp_rate),
            )
        )

    results.sort(key=lambda row: row.score, reverse=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "roi_center_fraction",
                    "min_secondary_area_pct",
                    "primary_overlap_iou",
                    "min_rules_to_match",
                    "negative_fp_count",
                    "negative_total",
                    "intrusion_tp_count",
                    "intrusion_total",
                    "score",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow(
                    {
                        "roi_center_fraction": row.roi_center_fraction,
                        "min_secondary_area_pct": row.min_secondary_area_pct,
                        "primary_overlap_iou": row.primary_overlap_iou,
                        "min_rules_to_match": row.min_rules_to_match,
                        "negative_fp_count": row.negative_fp_count,
                        "negative_total": row.negative_total,
                        "intrusion_tp_count": row.intrusion_tp_count,
                        "intrusion_total": row.intrusion_total,
                        "score": f"{row.score:.4f}",
                    }
                )

    print(f"\nBacktest grid: {csv_path}")
    print(f"Configs evaluated: {len(results)}")
    if not negative_labels:
        print("Warning: no solo_* or background_* scenarios found.")
    if not positive_labels:
        print("Warning: no intrusion_* scenarios found.")

    def _print_profile(title: str, row: BacktestRow) -> None:
        """Print a profile of a backtest."""
        fp = row.negative_fp_count / row.negative_total if row.negative_total else 0.0
        tp = (
            row.intrusion_tp_count / row.intrusion_total if row.intrusion_total else 0.0
        )
        print(f"\n{title}")
        print(
            f"  roi={row.roi_center_fraction:.2f} "
            f"area={row.min_secondary_area_pct:.2f} "
            f"iou={row.primary_overlap_iou:.2f} "
            f"rules={row.min_rules_to_match}"
        )
        print(
            "  FP on solo/background: "
            f"{fp:.1%} ({row.negative_fp_count}/{row.negative_total})"
        )
        print(
            "  TP on intrusion: "
            f"{tp:.1%} ({row.intrusion_tp_count}/{row.intrusion_total})"
        )
        print(f"  score: {row.score:.3f}")

    conservative = next(
        (
            row
            for row in results
            if row.negative_total
            and row.intrusion_total
            and (row.negative_fp_count / row.negative_total) <= 0.05
        ),
        None,
    )
    sensitive = results[0] if results else None

    print("\nTop configs:")
    for row in results[:top_n]:
        fp = row.negative_fp_count / row.negative_total if row.negative_total else 0.0
        tp = (
            row.intrusion_tp_count / row.intrusion_total if row.intrusion_total else 0.0
        )
        print(
            f"  score={row.score:.3f} FP={fp:.0%} TP={tp:.0%} "
            f"roi={row.roi_center_fraction:.2f} "
            f"area={row.min_secondary_area_pct:.2f} "
            f"iou={row.primary_overlap_iou:.2f} "
            f"rules={row.min_rules_to_match}"
        )

    if conservative:
        _print_profile("Profile A (conservative, FP <= 5%)", conservative)
    if sensitive:
        _print_profile("Profile B (best score)", sensitive)

    if output_path is not None:
        print(f"\nWrote {output_path}")

    return results


def append_rows(
    output_path: Path,
    *,
    experiment: str,
    label: str,
    rows: list[dict],
    recorded_at: str,
) -> None:
    """Append rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "recorded_at": recorded_at,
                    "experiment": experiment,
                    "label": label,
                    "frame_index": index,
                    **row,
                }
            )
