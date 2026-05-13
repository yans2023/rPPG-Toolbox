#!/usr/bin/env python3
"""Align rPPG heart-rate CSV files with fiber sensor JSONL logs.

The script reads all rPPG CSV files and fiber TXT/JSONL files in one directory,
aggregates both sources to one row per wall-clock second, writes an aligned CSV,
and writes a concise comparison summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Iterable


RPPG_SESSION_RE = re.compile(r"hr_(\d{8}_\d{6})_part\d+\.csv$")


@dataclass
class SeriesStats:
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float


@dataclass
class DiffStats:
    count: int
    bias: float
    mae: float
    rmse: float
    median_abs_error: float
    std_diff: float
    min_diff: float
    max_diff: float
    p95_abs_error: float
    corr: float | None
    within_3_bpm_pct: float
    within_5_bpm_pct: float
    within_10_bpm_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align rPPG CSV heart-rate output with fiber sensor TXT logs."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="test_output/20260420-rPPG-fiber",
        help="Directory containing rPPG *.csv files and fiber *.txt/*.jsonl files.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Aligned CSV path. Defaults to <input_dir>/aligned_rppg_fiber_hr.csv.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary TXT path. Defaults to <input_dir>/rppg_fiber_hr_summary.txt.",
    )
    parser.add_argument(
        "--fiber-time-offset-sec",
        type=float,
        default=0.0,
        help=(
            "Seconds added to fiber deviceSendTime before alignment. "
            "Use this if device and computer clocks are known to differ."
        ),
    )
    parser.add_argument(
        "--min-hr",
        type=float,
        default=30.0,
        help="Minimum valid heart rate in BPM.",
    )
    parser.add_argument(
        "--max-hr",
        type=float,
        default=220.0,
        help="Maximum valid heart rate in BPM.",
    )
    return parser.parse_args()


def parse_rppg_timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")


def parse_fiber_timestamp(value: object, offset_sec: float) -> datetime:
    timestamp = int(str(value).strip())
    return datetime.fromtimestamp(timestamp) + timedelta(seconds=offset_sec)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def corrcoef(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_avg = sum(xs) / len(xs)
    y_avg = sum(ys) / len(ys)
    cov = sum((x - x_avg) * (y - y_avg) for x, y in zip(xs, ys))
    x_var = sum((x - x_avg) ** 2 for x in xs)
    y_var = sum((y - y_avg) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return cov / math.sqrt(x_var * y_var)


def series_stats(values: list[float]) -> SeriesStats | None:
    if not values:
        return None
    return SeriesStats(
        count=len(values),
        mean=sum(values) / len(values),
        median=median(values),
        minimum=min(values),
        maximum=max(values),
        std=sample_std(values),
    )


def diff_stats(pairs: list[tuple[float, float]]) -> DiffStats | None:
    if not pairs:
        return None
    rppg_values = [pair[0] for pair in pairs]
    fiber_values = [pair[1] for pair in pairs]
    diffs = [rppg - fiber for rppg, fiber in pairs]
    abs_diffs = [abs(diff) for diff in diffs]
    count = len(diffs)
    return DiffStats(
        count=count,
        bias=sum(diffs) / count,
        mae=sum(abs_diffs) / count,
        rmse=math.sqrt(sum(diff**2 for diff in diffs) / count),
        median_abs_error=median(abs_diffs),
        std_diff=sample_std(diffs),
        min_diff=min(diffs),
        max_diff=max(diffs),
        p95_abs_error=percentile(abs_diffs, 95),
        corr=corrcoef(rppg_values, fiber_values),
        within_3_bpm_pct=sum(value <= 3 for value in abs_diffs) * 100.0 / count,
        within_5_bpm_pct=sum(value <= 5 for value in abs_diffs) * 100.0 / count,
        within_10_bpm_pct=sum(value <= 10 for value in abs_diffs) * 100.0 / count,
    )


def is_valid_hr(value: float, min_hr: float, max_hr: float) -> bool:
    return min_hr <= value <= max_hr


def session_id_from_file(path: Path) -> str:
    match = RPPG_SESSION_RE.match(path.name)
    return match.group(1) if match else path.stem


def read_rppg(input_dir: Path, min_hr: float, max_hr: float) -> tuple[dict[datetime, dict], dict]:
    by_second: dict[datetime, dict] = defaultdict(
        lambda: {"hr": [], "raw_count": 0, "valid_count": 0, "files": set(), "sessions": set()}
    )
    meta = {
        "files": [],
        "raw_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "range": None,
    }

    for path in sorted(input_dir.glob("*.csv")):
        meta["files"].append(path.name)
        session_id = session_id_from_file(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta["raw_rows"] += 1
                timestamp = parse_rppg_timestamp(row["timestamp"])
                bucket = by_second[timestamp]
                bucket["raw_count"] += 1
                bucket["files"].add(path.name)
                bucket["sessions"].add(session_id)

                value = float(row["hr_bpm"])
                if is_valid_hr(value, min_hr, max_hr):
                    bucket["hr"].append(value)
                    bucket["valid_count"] += 1
                    meta["valid_rows"] += 1
                else:
                    meta["invalid_rows"] += 1

                current_range = meta["range"]
                if current_range is None:
                    meta["range"] = [timestamp, timestamp]
                else:
                    current_range[0] = min(current_range[0], timestamp)
                    current_range[1] = max(current_range[1], timestamp)

    return dict(by_second), meta


def read_fiber(
    input_dir: Path, min_hr: float, max_hr: float, offset_sec: float
) -> tuple[dict[datetime, dict], dict]:
    by_second: dict[datetime, dict] = defaultdict(
        lambda: {
            "hr": [],
            "raw_count": 0,
            "valid_count": 0,
            "files": set(),
            "heart_rate_doc": [],
            "body_motion": [],
        }
    )
    meta = {
        "files": [],
        "raw_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "range": None,
    }

    for path in sorted([*input_dir.glob("*.txt"), *input_dir.glob("*.jsonl")]):
        if path.name == "rppg_fiber_hr_summary.txt":
            continue
        meta["files"].append(path.name)
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    data = payload.get("data", {})
                    timestamp = parse_fiber_timestamp(data["deviceSendTime"], offset_sec)
                    value = float(data["heartRate"])
                except json.JSONDecodeError:
                    continue
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Cannot parse {path}:{line_no}: {exc}") from exc

                bucket = by_second[timestamp]
                bucket["raw_count"] += 1
                bucket["files"].add(path.name)
                meta["raw_rows"] += 1

                if data.get("heartRateDoc") is not None:
                    bucket["heart_rate_doc"].append(float(data["heartRateDoc"]))
                if data.get("bodyMotion") is not None:
                    bucket["body_motion"].append(float(data["bodyMotion"]))

                if is_valid_hr(value, min_hr, max_hr):
                    bucket["hr"].append(value)
                    bucket["valid_count"] += 1
                    meta["valid_rows"] += 1
                else:
                    meta["invalid_rows"] += 1

                current_range = meta["range"]
                if current_range is None:
                    meta["range"] = [timestamp, timestamp]
                else:
                    current_range[0] = min(current_range[0], timestamp)
                    current_range[1] = max(current_range[1], timestamp)

    return dict(by_second), meta


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_time_range(value: list[datetime] | None) -> str:
    if not value:
        return "N/A"
    return f"{value[0]:%Y-%m-%d %H:%M:%S} -> {value[1]:%Y-%m-%d %H:%M:%S}"


def format_series_stats(label: str, stats: SeriesStats | None) -> list[str]:
    if stats is None:
        return [f"{label}: N/A"]
    return [
        (
            f"{label}: n={stats.count}, mean={stats.mean:.2f}, median={stats.median:.2f}, "
            f"std={stats.std:.2f}, min={stats.minimum:.2f}, max={stats.maximum:.2f}"
        )
    ]


def format_diff_stats(label: str, stats: DiffStats | None) -> list[str]:
    if stats is None:
        return [f"{label}: N/A"]
    corr = "N/A" if stats.corr is None else f"{stats.corr:.3f}"
    return [
        f"{label}: n={stats.count}",
        f"  bias(rPPG-fiber)={stats.bias:.2f} BPM",
        f"  MAE={stats.mae:.2f} BPM, RMSE={stats.rmse:.2f} BPM, median_abs_error={stats.median_abs_error:.2f} BPM",
        f"  std_diff={stats.std_diff:.2f} BPM, diff_range=[{stats.min_diff:.2f}, {stats.max_diff:.2f}] BPM",
        f"  p95_abs_error={stats.p95_abs_error:.2f} BPM, corr={corr}",
        (
            f"  within: <=3 BPM {stats.within_3_bpm_pct:.1f}%, "
            f"<=5 BPM {stats.within_5_bpm_pct:.1f}%, <=10 BPM {stats.within_10_bpm_pct:.1f}%"
        ),
    ]


def build_rows(rppg_by_second: dict[datetime, dict], fiber_by_second: dict[datetime, dict]) -> list[dict]:
    timestamps = sorted(set(rppg_by_second) | set(fiber_by_second))
    rows = []
    for timestamp in timestamps:
        rppg = rppg_by_second.get(timestamp)
        fiber = fiber_by_second.get(timestamp)
        rppg_hr = mean(rppg["hr"]) if rppg and rppg["hr"] else None
        fiber_hr = mean(fiber["hr"]) if fiber and fiber["hr"] else None
        diff = rppg_hr - fiber_hr if rppg_hr is not None and fiber_hr is not None else None
        abs_diff = abs(diff) if diff is not None else None
        rows.append(
            {
                "timestamp": timestamp,
                "rppg_hr_bpm": rppg_hr,
                "fiber_hr_bpm": fiber_hr,
                "diff_bpm": diff,
                "abs_diff_bpm": abs_diff,
                "rppg_raw_count": rppg["raw_count"] if rppg else 0,
                "rppg_valid_count": rppg["valid_count"] if rppg else 0,
                "fiber_raw_count": fiber["raw_count"] if fiber else 0,
                "fiber_valid_count": fiber["valid_count"] if fiber else 0,
                "rppg_files": ";".join(sorted(rppg["files"])) if rppg else "",
                "rppg_sessions": ";".join(sorted(rppg["sessions"])) if rppg else "",
                "fiber_files": ";".join(sorted(fiber["files"])) if fiber else "",
                "fiber_heart_rate_doc_mean": (
                    mean(fiber["heart_rate_doc"]) if fiber and fiber["heart_rate_doc"] else None
                ),
                "fiber_body_motion_max": (
                    max(fiber["body_motion"]) if fiber and fiber["body_motion"] else None
                ),
            }
        )
    return rows


def write_aligned_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "timestamp",
        "rppg_hr_bpm",
        "fiber_hr_bpm",
        "diff_bpm",
        "abs_diff_bpm",
        "rppg_raw_count",
        "rppg_valid_count",
        "fiber_raw_count",
        "fiber_valid_count",
        "rppg_files",
        "rppg_sessions",
        "fiber_files",
        "fiber_heart_rate_doc_mean",
        "fiber_body_motion_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "rppg_hr_bpm": fmt_float(row["rppg_hr_bpm"], 3),
                    "fiber_hr_bpm": fmt_float(row["fiber_hr_bpm"], 3),
                    "diff_bpm": fmt_float(row["diff_bpm"], 3),
                    "abs_diff_bpm": fmt_float(row["abs_diff_bpm"], 3),
                    "rppg_raw_count": row["rppg_raw_count"],
                    "rppg_valid_count": row["rppg_valid_count"],
                    "fiber_raw_count": row["fiber_raw_count"],
                    "fiber_valid_count": row["fiber_valid_count"],
                    "rppg_files": row["rppg_files"],
                    "rppg_sessions": row["rppg_sessions"],
                    "fiber_files": row["fiber_files"],
                    "fiber_heart_rate_doc_mean": fmt_float(row["fiber_heart_rate_doc_mean"], 3),
                    "fiber_body_motion_max": fmt_float(row["fiber_body_motion_max"], 3),
                }
            )


def comparable_pairs(rows: list[dict]) -> list[tuple[float, float]]:
    return [
        (row["rppg_hr_bpm"], row["fiber_hr_bpm"])
        for row in rows
        if row["rppg_hr_bpm"] is not None and row["fiber_hr_bpm"] is not None
    ]


def write_summary(path: Path, rows: list[dict], rppg_meta: dict, fiber_meta: dict) -> None:
    pairs = comparable_pairs(rows)
    rppg_values = [row["rppg_hr_bpm"] for row in rows if row["rppg_hr_bpm"] is not None]
    fiber_values = [row["fiber_hr_bpm"] for row in rows if row["fiber_hr_bpm"] is not None]
    both_valid_rows = [row for row in rows if row["rppg_hr_bpm"] is not None and row["fiber_hr_bpm"] is not None]

    lines = [
        "rPPG vs fiber heart-rate alignment summary",
        "",
        f"rPPG files: {len(rppg_meta['files'])} ({', '.join(rppg_meta['files'])})",
        f"fiber files: {len(fiber_meta['files'])} ({', '.join(fiber_meta['files'])})",
        f"rPPG raw rows: {rppg_meta['raw_rows']}, valid rows: {rppg_meta['valid_rows']}, invalid rows: {rppg_meta['invalid_rows']}",
        f"fiber raw rows: {fiber_meta['raw_rows']}, valid rows: {fiber_meta['valid_rows']}, invalid rows: {fiber_meta['invalid_rows']}",
        f"rPPG time range: {fmt_time_range(rppg_meta['range'])}",
        f"fiber time range: {fmt_time_range(fiber_meta['range'])}",
        f"aligned union seconds: {len(rows)}",
        f"comparable seconds with both valid HR: {len(both_valid_rows)}",
        "",
        "Heart-rate distribution after per-second aggregation:",
    ]
    lines.extend(format_series_stats("  rPPG", series_stats(rppg_values)))
    lines.extend(format_series_stats("  fiber", series_stats(fiber_values)))
    lines.extend(["", "Difference statistics on comparable seconds:"])
    lines.extend(format_diff_stats("  overall", diff_stats(pairs)))

    sessions = sorted(
        {
            session
            for row in rows
            for session in row["rppg_sessions"].split(";")
            if session and row["rppg_hr_bpm"] is not None
        }
    )
    if sessions:
        lines.extend(["", "Per-rPPG-session difference statistics:"])
        for session in sessions:
            session_pairs = [
                (row["rppg_hr_bpm"], row["fiber_hr_bpm"])
                for row in rows
                if session in row["rppg_sessions"].split(";")
                and row["rppg_hr_bpm"] is not None
                and row["fiber_hr_bpm"] is not None
            ]
            lines.extend(format_diff_stats(f"  {session}", diff_stats(session_pairs)))

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "aligned_rppg_fiber_hr.csv"
    summary_path = Path(args.summary) if args.summary else input_dir / "rppg_fiber_hr_summary.txt"

    rppg_by_second, rppg_meta = read_rppg(input_dir, args.min_hr, args.max_hr)
    fiber_by_second, fiber_meta = read_fiber(
        input_dir, args.min_hr, args.max_hr, args.fiber_time_offset_sec
    )
    if not rppg_by_second:
        raise SystemExit(f"No rPPG CSV rows found in {input_dir}")
    if not fiber_by_second:
        raise SystemExit(f"No fiber TXT/JSONL rows found in {input_dir}")

    rows = build_rows(rppg_by_second, fiber_by_second)
    write_aligned_csv(output_csv, rows)
    write_summary(summary_path, rows, rppg_meta, fiber_meta)

    pairs = comparable_pairs(rows)
    stats = diff_stats(pairs)
    print(f"wrote aligned CSV: {output_csv}")
    print(f"wrote summary: {summary_path}")
    print(f"comparable seconds: {len(pairs)}")
    if stats:
        print(
            "overall: "
            f"bias={stats.bias:.2f} BPM, MAE={stats.mae:.2f} BPM, "
            f"RMSE={stats.rmse:.2f} BPM, corr={stats.corr:.3f}"
        )


if __name__ == "__main__":
    main()
