#!/usr/bin/env python3
"""Collect and analyze 1 minute rPPG heart-rate indicators.

Default mode:
    1. Run realtime_hr.py for 60 seconds.
    2. Save the raw HR CSV.
    3. Compute one 1 minute indicator row.

Analysis-only mode:
    Pass --input-csv to compute 1 minute indicators from an existing rPPG HR CSV.

The script intentionally uses only the current rPPG CSV output fields:
elapsed_sec, timestamp, hr_bpm.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_output" / "rppg_1min"


@dataclass
class HrSample:
    elapsed_sec: int
    timestamp: str
    hr_bpm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 60s rPPG HR data and compute 1 minute indicators."
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Analyze an existing rPPG HR CSV instead of collecting new data.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for collected HR CSV and indicator outputs.",
    )
    parser.add_argument(
        "--raw-csv",
        default="",
        help="Raw HR CSV path for collection mode. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--metrics-csv",
        default="",
        help="Indicator CSV output path. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Collection duration in seconds. Default: 60.",
    )
    parser.add_argument(
        "--window-size-sec",
        type=int,
        default=60,
        help="Analysis window size in seconds. Default: 60.",
    )
    parser.add_argument(
        "--step-sec",
        type=int,
        default=60,
        help="Window step in seconds for analysis-only mode. Default: 60.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index passed to realtime_hr.py.",
    )
    parser.add_argument(
        "--rppg-window",
        type=float,
        default=10.0,
        help="Sliding rPPG calculation window passed to realtime_hr.py.",
    )
    parser.add_argument(
        "--method",
        choices=["POS", "CHROM"],
        default="POS",
        help="rPPG method passed to realtime_hr.py.",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=0.6,
        help="Face ROI scale passed to realtime_hr.py.",
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
        help="Adjacent valid sample jump threshold counted as an outlier. Default: 25.",
    )
    parser.add_argument(
        "--delta-segment-sec",
        type=int,
        default=10,
        help="First/last segment length for heart_rate_delta. Default: 10.",
    )
    return parser.parse_args()


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = timestamp_tag()

    if args.input_csv:
        raw_csv = Path(args.input_csv)
    elif args.raw_csv:
        raw_csv = Path(args.raw_csv)
    else:
        raw_csv = output_dir / f"rppg_1min_hr_{tag}.csv"

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output_dir / f"rppg_1min_metrics_{tag}.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    return raw_csv, metrics_csv


def run_collection(args: argparse.Namespace, raw_csv: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "realtime_hr.py"),
        "--camera",
        str(args.camera),
        "--window",
        str(args.rppg_window),
        "--method",
        args.method,
        "--roi-scale",
        str(args.roi_scale),
        "--duration",
        str(args.duration),
        "--csv",
        str(raw_csv),
    ]
    print("[INFO] Starting rPPG collection:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw HR CSV was not created: {raw_csv}")


def parse_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_float(value: Any) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(str(value).strip())
    except Exception:
        return float("nan")


def read_hr_csv(path: Path) -> list[HrSample]:
    samples: list[HrSample] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"elapsed_sec", "timestamp", "hr_bpm"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")

        for row in reader:
            elapsed_sec = parse_int(row.get("elapsed_sec"))
            if elapsed_sec is None:
                continue
            samples.append(
                HrSample(
                    elapsed_sec=elapsed_sec,
                    timestamp=str(row.get("timestamp", "")).strip(),
                    hr_bpm=parse_float(row.get("hr_bpm")),
                )
            )

    samples.sort(key=lambda x: x.elapsed_sec)
    return samples


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        if not math.isfinite(float(value)):
            return ""
    except Exception:
        return ""
    return f"{float(value):.{digits}f}"


def sample_entropy(data: np.ndarray, m: int = 2, r: float | None = None) -> float:
    try:
        data = np.asarray(data, dtype=float)
        if data.size < 10:
            return float("nan")
        if r is None:
            r = 0.2 * float(np.std(data))
        if r == 0:
            return float("nan")

        n = int(data.size)

        def maxdist(x_i: list[float], x_j: list[float]) -> float:
            return max(abs(a - b) for a, b in zip(x_i, x_j))

        def phi(mm: int) -> int:
            patterns = [[data[j] for j in range(i, i + mm)] for i in range(n - mm + 1)]
            counts = [
                len([1 for x_j in patterns if maxdist(x_i, x_j) <= r]) - 1
                for x_i in patterns
            ]
            return int(sum(counts))

        phi_m = phi(m)
        phi_mp1 = phi(m + 1)
        if phi_m <= 0 or phi_mp1 <= 0:
            return float("nan")
        return float(np.log(phi_m / phi_mp1))
    except Exception:
        return float("nan")


def quality_flag(valid_ratio: float) -> str:
    if valid_ratio >= 0.80:
        return "good"
    if valid_ratio >= 0.60:
        return "fair"
    return "poor"


def valid_hr_mask(values: np.ndarray, min_hr: float, max_hr: float) -> np.ndarray:
    return np.isfinite(values) & (values >= min_hr) & (values <= max_hr)


def linear_slope_bpm_per_min(elapsed: np.ndarray, values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    x = elapsed.astype(float)
    y = values.astype(float)
    x = x - float(np.mean(x))
    denom = float(np.sum(x**2))
    if denom == 0:
        return float("nan")
    slope_bpm_per_sec = float(np.sum(x * (y - float(np.mean(y)))) / denom)
    return slope_bpm_per_sec * 60.0


def calculate_window_metrics(
    samples: list[HrSample],
    *,
    window_start_sec: int,
    window_size_sec: int,
    min_hr: float,
    max_hr: float,
    max_jump_bpm: float,
    delta_segment_sec: int,
    source: str,
) -> dict[str, Any]:
    window_end_sec = window_start_sec + window_size_sec
    by_elapsed = {sample.elapsed_sec: sample for sample in samples}
    expected_elapsed = list(range(window_start_sec, window_end_sec))
    expected_count = len(expected_elapsed)

    hr_values = np.asarray(
        [by_elapsed[sec].hr_bpm if sec in by_elapsed else float("nan") for sec in expected_elapsed],
        dtype=float,
    )
    valid_mask = valid_hr_mask(hr_values, min_hr, max_hr)
    valid_values = hr_values[valid_mask]
    valid_elapsed = np.asarray(expected_elapsed, dtype=float)[valid_mask]

    below_or_above_count = int(np.sum(np.isfinite(hr_values) & ~valid_mask))
    if valid_values.size > 1:
        jump_outlier_count = int(np.sum(np.abs(np.diff(valid_values)) > max_jump_bpm))
    else:
        jump_outlier_count = 0
    hr_outlier_count = below_or_above_count + jump_outlier_count

    valid_count = int(valid_values.size)
    valid_ratio = valid_count / expected_count if expected_count else 0.0
    missing_count = expected_count - valid_count

    first_sample = next((by_elapsed[sec] for sec in expected_elapsed if sec in by_elapsed), None)
    last_sample = next((by_elapsed[sec] for sec in reversed(expected_elapsed) if sec in by_elapsed), None)

    result: dict[str, Any] = {
        "window_start": first_sample.timestamp if first_sample else "",
        "window_end": last_sample.timestamp if last_sample else "",
        "window_start_elapsed_sec": window_start_sec,
        "window_end_elapsed_sec": window_end_sec - 1,
        "window_size_sec": window_size_sec,
        "source": source,
        "sampling_interval_sec": 1,
        "expected_count": expected_count,
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "missing_count": missing_count,
        "hr_outlier_count": hr_outlier_count,
        "quality_flag": quality_flag(valid_ratio),
        "avg_heart_rate": float("nan"),
        "median_heart_rate": float("nan"),
        "min_heart_rate": float("nan"),
        "max_heart_rate": float("nan"),
        "heart_rate_range": float("nan"),
        "heart_rate_slope": float("nan"),
        "heart_rate_delta": float("nan"),
        "SD_HR": float("nan"),
        "CV_HR": float("nan"),
        "MeanJJ": float("nan"),
        "SDNN": float("nan"),
        "RMSSD_HR": float("nan"),
        "HRV_diff": float("nan"),
        "SD1": float("nan"),
        "SD2": float("nan"),
        "SD1_SD2_ratio": float("nan"),
        "PNN50_experimental": float("nan"),
        "SampEn_HR_experimental": float("nan"),
        "VAI_experimental": float("nan"),
        "VLI_experimental": float("nan"),
        "ANS_activity_experimental": float("nan"),
    }

    if valid_count == 0:
        return result

    result["avg_heart_rate"] = float(np.mean(valid_values))
    result["median_heart_rate"] = float(median(valid_values.tolist()))
    result["min_heart_rate"] = float(np.min(valid_values))
    result["max_heart_rate"] = float(np.max(valid_values))
    result["heart_rate_range"] = float(result["max_heart_rate"] - result["min_heart_rate"])
    result["heart_rate_slope"] = linear_slope_bpm_per_min(valid_elapsed, valid_values)

    segment = max(1, min(delta_segment_sec, window_size_sec // 2))
    first_segment_mask = valid_mask[:segment]
    last_segment_mask = valid_mask[-segment:]
    first_segment_values = hr_values[:segment][first_segment_mask]
    last_segment_values = hr_values[-segment:][last_segment_mask]
    if first_segment_values.size > 0 and last_segment_values.size > 0:
        result["heart_rate_delta"] = float(np.mean(last_segment_values) - np.mean(first_segment_values))

    if valid_count >= 2:
        sd_hr = float(np.std(valid_values, ddof=1))
        result["SD_HR"] = sd_hr
        mean_hr = float(np.mean(valid_values))
        result["CV_HR"] = float(sd_hr / mean_hr) if mean_hr != 0 else float("nan")

        rr_intervals = 60.0 / valid_values
        rr_ms = rr_intervals * 1000.0
        rr_diff_ms = np.diff(rr_intervals) * 1000.0
        result["MeanJJ"] = float(np.mean(rr_ms))
        result["SDNN"] = float(np.std(rr_intervals, ddof=1) * 1000.0)
        if rr_diff_ms.size > 0:
            result["RMSSD_HR"] = float(np.sqrt(np.mean(rr_diff_ms**2)))
            result["HRV_diff"] = float(np.max(np.abs(rr_diff_ms)))
            result["PNN50_experimental"] = float(np.sum(np.abs(rr_diff_ms) > 50.0) / rr_diff_ms.size * 100.0)

        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        if rr_n.size > 1:
            diff_rr = rr_n1 - rr_n
            sum_rr = rr_n1 + rr_n
            sd1 = float(np.std(diff_rr, ddof=1) / np.sqrt(2.0) * 1000.0)
            sd2 = float(np.std(sum_rr, ddof=1) / np.sqrt(2.0) * 1000.0)
            result["SD1"] = sd1
            result["SD2"] = sd2
            result["SD1_SD2_ratio"] = float(sd1 / sd2) if sd2 > 0 else float("nan")

            center_x = float(np.mean(rr_n))
            center_y = float(np.mean(rr_n1))
            angles = np.arctan2(rr_n1 - center_y, rr_n - center_x)
            distances = np.sqrt((rr_n - center_x) ** 2 + (rr_n1 - center_y) ** 2)
            result["VAI_experimental"] = float(np.std(angles - np.pi / 4.0, ddof=1))
            result["VLI_experimental"] = float(np.mean(distances) * 1000.0)

        result["ANS_activity_experimental"] = result["SDNN"]

    result["SampEn_HR_experimental"] = sample_entropy(valid_values)
    return result


FIELDNAMES = [
    "window_start",
    "window_end",
    "window_start_elapsed_sec",
    "window_end_elapsed_sec",
    "window_size_sec",
    "source",
    "sampling_interval_sec",
    "expected_count",
    "valid_count",
    "valid_ratio",
    "missing_count",
    "hr_outlier_count",
    "quality_flag",
    "avg_heart_rate",
    "median_heart_rate",
    "min_heart_rate",
    "max_heart_rate",
    "heart_rate_range",
    "heart_rate_slope",
    "heart_rate_delta",
    "SD_HR",
    "CV_HR",
    "MeanJJ",
    "SDNN",
    "RMSSD_HR",
    "HRV_diff",
    "SD1",
    "SD2",
    "SD1_SD2_ratio",
    "PNN50_experimental",
    "SampEn_HR_experimental",
    "VAI_experimental",
    "VLI_experimental",
    "ANS_activity_experimental",
]


def window_starts(samples: list[HrSample], window_size_sec: int, step_sec: int) -> list[int]:
    if not samples:
        return []
    min_elapsed = min(sample.elapsed_sec for sample in samples)
    max_elapsed = max(sample.elapsed_sec for sample in samples)
    starts = []
    current = min_elapsed
    while current <= max_elapsed:
        starts.append(current)
        current += step_sec
    if len(starts) == 1:
        return starts
    return [start for start in starts if start + window_size_sec - 1 <= max_elapsed]


def collection_window_starts(duration_sec: int, window_size_sec: int, step_sec: int) -> list[int]:
    if duration_sec < window_size_sec:
        return [0]
    return list(range(0, duration_sec - window_size_sec + 1, step_sec))


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: fmt_float(row.get(key), 6)
                    if isinstance(row.get(key), (float, np.floating))
                    else row.get(key, "")
                    for key in FIELDNAMES
                }
            )


def print_summary(rows: list[dict[str, Any]], metrics_csv: Path) -> None:
    print(f"[INFO] Metrics CSV: {metrics_csv}")
    print(f"[INFO] Computed windows: {len(rows)}")
    if not rows:
        return
    row = rows[0]
    print("[INFO] First window summary:")
    print(
        "       "
        f"quality={row['quality_flag']}, valid={row['valid_count']}/{row['expected_count']} "
        f"({row['valid_ratio']:.1%})"
    )
    print(
        "       "
        f"HR mean={fmt_float(row['avg_heart_rate'], 2)} BPM, "
        f"median={fmt_float(row['median_heart_rate'], 2)} BPM, "
        f"range={fmt_float(row['heart_rate_range'], 2)} BPM"
    )
    print(
        "       "
        f"SDNN={fmt_float(row['SDNN'], 2)} ms, "
        f"RMSSD={fmt_float(row['RMSSD_HR'], 2)} ms, "
        f"SampEn(exp)={fmt_float(row['SampEn_HR_experimental'], 3)}"
    )


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.window_size_sec <= 0:
        raise SystemExit("--window-size-sec must be positive")
    if args.step_sec <= 0:
        raise SystemExit("--step-sec must be positive")

    raw_csv, metrics_csv = default_paths(args)

    if not args.input_csv:
        run_collection(args, raw_csv)

    samples = read_hr_csv(raw_csv)
    if not samples:
        raise SystemExit(f"No HR samples found in {raw_csv}")

    source = raw_csv.name
    starts = (
        window_starts(samples, args.window_size_sec, args.step_sec)
        if args.input_csv
        else collection_window_starts(args.duration, args.window_size_sec, args.step_sec)
    )
    rows = [
        calculate_window_metrics(
            samples,
            window_start_sec=start,
            window_size_sec=args.window_size_sec,
            min_hr=args.min_hr,
            max_hr=args.max_hr,
            max_jump_bpm=args.max_jump_bpm,
            delta_segment_sec=args.delta_segment_sec,
            source=source,
        )
        for start in starts
    ]
    write_metrics_csv(metrics_csv, rows)
    print(f"[INFO] Raw HR CSV: {raw_csv}")
    print_summary(rows, metrics_csv)


if __name__ == "__main__":
    main()
