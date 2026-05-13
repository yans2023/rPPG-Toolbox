#!/usr/bin/env python3
"""Compare 1 minute HR indicators for aligned rPPG and fiber HR data."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rppg_1min_collect_analyze import (  # noqa: E402
    FIELDNAMES,
    HrSample,
    calculate_window_metrics,
    fmt_float,
)


DEFAULT_INPUT = Path("test_output/20260420-rPPG-fiber/aligned_rppg_fiber_hr.csv")
DEFAULT_OUTPUT_DIR = Path("test_output/20260420-rPPG-fiber")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and compare 1 minute indicators for aligned rPPG and fiber HR."
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT),
        help="Aligned CSV containing timestamp, rppg_hr_bpm, fiber_hr_bpm.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Number of 1 minute windows to compare. Default: 5.",
    )
    parser.add_argument(
        "--window-size-sec",
        type=int,
        default=60,
        help="Window size in seconds. Default: 60.",
    )
    parser.add_argument(
        "--start-time",
        default="",
        help="Optional start time, format YYYY-mm-dd HH:MM:SS. Defaults to first comparable timestamp.",
    )
    parser.add_argument(
        "--min-hr",
        type=float,
        default=40.0,
        help="Minimum valid HR in BPM. Default: 40.",
    )
    parser.add_argument(
        "--max-hr",
        type=float,
        default=180.0,
        help="Maximum valid HR in BPM. Default: 180.",
    )
    parser.add_argument(
        "--max-jump-bpm",
        type=float,
        default=25.0,
        help="Adjacent valid sample jump threshold counted as outlier. Default: 25.",
    )
    parser.add_argument(
        "--delta-segment-sec",
        type=int,
        default=10,
        help="First/last segment length for heart_rate_delta. Default: 10.",
    )
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(str(value).strip())
    except Exception:
        return float("nan")


def load_aligned(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "rppg_hr_bpm", "fiber_hr_bpm"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            timestamp = parse_time(row["timestamp"])
            rows.append(
                {
                    "timestamp": timestamp,
                    "rppg_hr_bpm": parse_float(row.get("rppg_hr_bpm")),
                    "fiber_hr_bpm": parse_float(row.get("fiber_hr_bpm")),
                }
            )
    rows.sort(key=lambda x: x["timestamp"])
    return rows


def first_comparable_time(rows: list[dict[str, Any]], min_hr: float, max_hr: float) -> datetime:
    for row in rows:
        rppg = row["rppg_hr_bpm"]
        fiber = row["fiber_hr_bpm"]
        if math.isfinite(rppg) and math.isfinite(fiber) and min_hr <= rppg <= max_hr and min_hr <= fiber <= max_hr:
            return row["timestamp"]
    raise ValueError("No comparable rows found with both valid rPPG and fiber HR.")


def build_samples(
    rows: list[dict[str, Any]],
    *,
    start_time: datetime,
    end_time: datetime,
    source_column: str,
) -> list[HrSample]:
    samples: list[HrSample] = []
    for row in rows:
        timestamp = row["timestamp"]
        if timestamp < start_time or timestamp > end_time:
            continue
        elapsed_sec = int((timestamp - start_time).total_seconds())
        samples.append(
            HrSample(
                elapsed_sec=elapsed_sec,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                hr_bpm=row[source_column],
            )
        )
    return samples


def comparison_fields() -> list[str]:
    excluded = {
        "source",
        "window_start",
        "window_end",
        "window_start_elapsed_sec",
        "window_end_elapsed_sec",
        "window_size_sec",
        "sampling_interval_sec",
        "quality_flag",
    }
    metric_fields = [field for field in FIELDNAMES if field not in excluded]
    fields = [
        "window_index",
        "window_start",
        "window_end",
        "rppg_quality_flag",
        "fiber_quality_flag",
    ]
    for field in metric_fields:
        fields.append(f"rppg_{field}")
        fields.append(f"fiber_{field}")
        if field not in {"expected_count", "valid_count", "missing_count", "hr_outlier_count", "quality_flag"}:
            fields.append(f"diff_{field}")
            fields.append(f"abs_diff_{field}")
    return fields


def metric_diff(left: Any, right: Any) -> tuple[float, float]:
    try:
        left_f = float(left)
        right_f = float(right)
        if not math.isfinite(left_f) or not math.isfinite(right_f):
            return float("nan"), float("nan")
        diff = left_f - right_f
        return diff, abs(diff)
    except Exception:
        return float("nan"), float("nan")


def make_comparison_rows(rppg_rows: list[dict[str, Any]], fiber_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = {
        "source",
        "window_start",
        "window_end",
        "window_start_elapsed_sec",
        "window_end_elapsed_sec",
        "window_size_sec",
        "sampling_interval_sec",
        "quality_flag",
    }
    metric_fields = [field for field in FIELDNAMES if field not in excluded]
    rows: list[dict[str, Any]] = []
    for idx, (rppg, fiber) in enumerate(zip(rppg_rows, fiber_rows), start=1):
        row: dict[str, Any] = {
            "window_index": idx,
            "window_start": rppg["window_start"],
            "window_end": rppg["window_end"],
            "rppg_quality_flag": rppg["quality_flag"],
            "fiber_quality_flag": fiber["quality_flag"],
        }
        for field in metric_fields:
            row[f"rppg_{field}"] = rppg.get(field, "")
            row[f"fiber_{field}"] = fiber.get(field, "")
            if field not in {"expected_count", "valid_count", "missing_count", "hr_outlier_count", "quality_flag"}:
                diff, abs_diff = metric_diff(rppg.get(field), fiber.get(field))
                row[f"diff_{field}"] = diff
                row[f"abs_diff_{field}"] = abs_diff
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: fmt_float(row.get(key), 6)
                    if isinstance(row.get(key), float)
                    else row.get(key, "")
                    for key in fieldnames
                }
            )


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def write_summary(path: Path, comparison_rows: list[dict[str, Any]], input_csv: Path) -> None:
    avg_hr_diffs = [float(row["diff_avg_heart_rate"]) for row in comparison_rows if math.isfinite(float(row["diff_avg_heart_rate"]))]
    median_hr_diffs = [float(row["diff_median_heart_rate"]) for row in comparison_rows if math.isfinite(float(row["diff_median_heart_rate"]))]
    rmssd_diffs = [float(row["diff_RMSSD_HR"]) for row in comparison_rows if math.isfinite(float(row["diff_RMSSD_HR"]))]
    sdnn_diffs = [float(row["diff_SDNN"]) for row in comparison_rows if math.isfinite(float(row["diff_SDNN"]))]

    lines = [
        "Aligned rPPG vs fiber 1 minute indicator comparison",
        "",
        f"input_csv: {input_csv}",
        f"windows: {len(comparison_rows)}",
        "",
        "Per-window key results:",
    ]
    for row in comparison_rows:
        lines.append(
            (
                f"  window {row['window_index']} ({row['window_start']} -> {row['window_end']}): "
                f"rPPG avg HR {float(row['rppg_avg_heart_rate']):.2f}, "
                f"fiber avg HR {float(row['fiber_avg_heart_rate']):.2f}, "
                f"diff {float(row['diff_avg_heart_rate']):+.2f} BPM; "
                f"rPPG RMSSD {float(row['rppg_RMSSD_HR']):.2f}, "
                f"fiber RMSSD {float(row['fiber_RMSSD_HR']):.2f}"
            )
        )

    lines.extend(
        [
            "",
            "Aggregate differences (rPPG - fiber):",
            f"  avg_heart_rate mean diff: {mean(avg_hr_diffs):+.2f} BPM",
            f"  median_heart_rate mean diff: {mean(median_hr_diffs):+.2f} BPM",
            f"  SDNN mean diff: {mean(sdnn_diffs):+.2f} ms",
            f"  RMSSD_HR mean diff: {mean(rmssd_diffs):+.2f} ms",
            "",
            "Notes:",
            "  HRV-like metrics are HR-derived proxy HRV, not beat-level HRV.",
            "  Fiber samples are not present on every wall-clock second, so quality flags should be checked.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_aligned(input_csv)
    start_time = parse_time(args.start_time) if args.start_time else first_comparable_time(rows, args.min_hr, args.max_hr)
    total_seconds = args.windows * args.window_size_sec
    end_time = start_time + timedelta(seconds=total_seconds - 1)

    rppg_samples = build_samples(rows, start_time=start_time, end_time=end_time, source_column="rppg_hr_bpm")
    fiber_samples = build_samples(rows, start_time=start_time, end_time=end_time, source_column="fiber_hr_bpm")

    rppg_metric_rows: list[dict[str, Any]] = []
    fiber_metric_rows: list[dict[str, Any]] = []
    for idx in range(args.windows):
        window_start_sec = idx * args.window_size_sec
        common_kwargs = {
            "window_start_sec": window_start_sec,
            "window_size_sec": args.window_size_sec,
            "min_hr": args.min_hr,
            "max_hr": args.max_hr,
            "max_jump_bpm": args.max_jump_bpm,
            "delta_segment_sec": args.delta_segment_sec,
        }
        rppg_metric_rows.append(
            calculate_window_metrics(rppg_samples, source="rppg_hr_bpm", **common_kwargs)
        )
        fiber_metric_rows.append(
            calculate_window_metrics(fiber_samples, source="fiber_hr_bpm", **common_kwargs)
        )

    comparison_rows = make_comparison_rows(rppg_metric_rows, fiber_metric_rows)

    rppg_metrics_csv = output_dir / "rppg_1min_metrics_from_aligned.csv"
    fiber_metrics_csv = output_dir / "fiber_1min_metrics_from_aligned.csv"
    comparison_csv = output_dir / "rppg_fiber_1min_metrics_comparison.csv"
    summary_txt = output_dir / "rppg_fiber_1min_metrics_comparison_summary.txt"

    write_csv(rppg_metrics_csv, rppg_metric_rows, FIELDNAMES)
    write_csv(fiber_metrics_csv, fiber_metric_rows, FIELDNAMES)
    write_csv(comparison_csv, comparison_rows, comparison_fields())
    write_summary(summary_txt, comparison_rows, input_csv)

    print(f"[INFO] window start: {start_time:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] window end: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] rPPG metrics: {rppg_metrics_csv}")
    print(f"[INFO] fiber metrics: {fiber_metrics_csv}")
    print(f"[INFO] comparison: {comparison_csv}")
    print(f"[INFO] summary: {summary_txt}")


if __name__ == "__main__":
    main()
