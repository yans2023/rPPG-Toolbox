from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import signal
from scipy.stats import mode, pearsonr


BASE_INDICATOR_COLUMNS: tuple[str, ...] = (
    # 心率基础
    "avg_heart_rate",
    "baseline_heart_rate",
    "min_heart_rate",
    "max_heart_rate",
    # 心率时域
    "HRV_diff",
    "MeanJJ",
    "SDNN",
    "RMSSD_HR",
    "PNN50",
    # 心率频域
    "VLF_HR",
    "LF_HR",
    "HF_HR",
    "TP_HR",
    "LF_HF_HR",
    "LF_norm_HR",
    "HF_norm_HR",
    # 心率衍生
    "CV_HR",
    "SampEn_HR",
    # 呼吸基础
    "avg_breath_rate",
    "baseline_breath_rate",
    "min_breath_rate",
    "max_breath_rate",
    # 呼吸时域
    "RRV_diff",
    "RRV",
    "SDRRI",
    "RMSSD_R",
    "Ve",
    # 呼吸频域
    "VLF_R",
    "LF_R",
    "HF_R",
    "LF_HF_R",
    # 呼吸衍生 + 心肺耦合
    "CV_R",
    "SampEn_R",
    "RRCC",
    # Poincaré
    "SD1",
    "SD2",
    "VAI",
    "VLI",
    # 自主神经评估
    "ANS_activity",
    "ANS_balance",
    "ANS_adaptability",
    "vagal_tone",
    "vagal_modulation",
    "sympathetic_tone",
    "sympathetic_modulation",
)


def parse_sequence(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return None
        return arr
    try:
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            return np.asarray([float(x) for x in s.split(",") if str(x).strip() != ""], dtype=float)
    except Exception:
        return None
    return None


def _rr_intervals_seconds(heart_rates_bpm: np.ndarray | None) -> np.ndarray | None:
    if heart_rates_bpm is None or heart_rates_bpm.size < 2:
        return None
    ok = heart_rates_bpm > 0
    if not np.all(ok):
        heart_rates_bpm = heart_rates_bpm[ok]
    if heart_rates_bpm.size < 2:
        return None
    return 60.0 / heart_rates_bpm


def _breath_intervals_seconds(breath_rates_bpm: np.ndarray | None) -> np.ndarray | None:
    if breath_rates_bpm is None or breath_rates_bpm.size < 2:
        return None
    ok = breath_rates_bpm > 0
    if not np.all(ok):
        breath_rates_bpm = breath_rates_bpm[ok]
    if breath_rates_bpm.size < 2:
        return None
    return 60.0 / breath_rates_bpm


def sample_entropy(data: np.ndarray | None, m: int = 2, r: float | None = None) -> float:
    try:
        if data is None:
            return float("nan")
        data = np.asarray(data, dtype=float)
        if data.size < 10:
            return float("nan")
        if r is None:
            r = 0.2 * float(np.std(data))
        if r == 0:
            return float("nan")

        N = int(data.size)

        def _maxdist(x_i: Iterable[float], x_j: Iterable[float]) -> float:
            return max(abs(a - b) for a, b in zip(x_i, x_j))

        def _phi(mm: int) -> int:
            x = [[data[j] for j in range(i, i + mm)] for i in range(N - mm + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1 for x_i in x]
            return int(sum(C))

        phi_m = _phi(m)
        phi_mp1 = _phi(m + 1)
        if phi_m <= 0 or phi_mp1 <= 0:
            return float("nan")
        return float(np.log(phi_m / phi_mp1))
    except Exception:
        return float("nan")


def _cv(data: np.ndarray | None) -> float:
    if data is None or data.size < 2:
        return float("nan")
    mean_val = float(np.mean(data))
    if mean_val == 0:
        return float("nan")
    return float(np.std(data, ddof=1) / mean_val)


def calculate_base_indicators(
    heart_rates_bpm: np.ndarray | None,
    breath_rates_bpm: np.ndarray | None,
    *,
    fs_hz: float = 1.0,
    tidal_volume_l: float = 0.5,
) -> dict[str, float]:
    """Notebook-aligned 45 indicator set (same as `complete_indicators_with_thresholds.ipynb`)."""

    indicators: dict[str, float] = {k: float("nan") for k in BASE_INDICATOR_COLUMNS}

    # ============== 心率基础指标 ==============
    if heart_rates_bpm is not None and heart_rates_bpm.size > 0:
        indicators["avg_heart_rate"] = float(np.mean(heart_rates_bpm))
        try:
            indicators["baseline_heart_rate"] = float(mode(np.round(heart_rates_bpm).astype(int), keepdims=True)[0][0])
        except Exception:
            indicators["baseline_heart_rate"] = float("nan")
        indicators["min_heart_rate"] = float(np.min(heart_rates_bpm))
        indicators["max_heart_rate"] = float(np.max(heart_rates_bpm))

    # ============== 心率时域指标 ==============
    rr_intervals = _rr_intervals_seconds(heart_rates_bpm)
    rr_diff_ms: np.ndarray | None = None
    if rr_intervals is not None and rr_intervals.size > 1:
        rr_diff_ms = np.diff(rr_intervals) * 1000.0
        indicators["HRV_diff"] = float(np.max(np.abs(rr_diff_ms))) if rr_diff_ms.size else float("nan")
        indicators["MeanJJ"] = float(np.mean(rr_intervals) * 1000.0)
        indicators["SDNN"] = float(np.std(rr_intervals, ddof=1) * 1000.0)
        indicators["RMSSD_HR"] = float(np.sqrt(np.mean(rr_diff_ms**2))) if rr_diff_ms.size else float("nan")
        nn50 = int(np.sum(np.abs(rr_diff_ms) > 50.0))
        indicators["PNN50"] = float((nn50 / rr_diff_ms.size * 100.0) if rr_diff_ms.size else np.nan)

    # ============== 心率频域指标 ==============
    if rr_intervals is not None and rr_intervals.size > 10:
        try:
            hrv = rr_intervals - float(np.mean(rr_intervals))
            freqs, psd = signal.welch(hrv, fs=fs_hz, nperseg=min(256, int(hrv.size)))
            vlf_power = float(np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)], freqs[(freqs >= 0.003) & (freqs < 0.04)]))
            lf_power = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)]))
            hf_power = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)]))
            total_power = vlf_power + lf_power + hf_power

            # 转换为 ms^2（rr_intervals 单位是秒，积分得到 sec^2；1 sec^2 = 1e6 ms^2）
            indicators["VLF_HR"] = vlf_power * 1_000_000.0
            indicators["LF_HR"] = lf_power * 1_000_000.0
            indicators["HF_HR"] = hf_power * 1_000_000.0
            indicators["TP_HR"] = total_power * 1_000_000.0
            indicators["LF_HF_HR"] = float(lf_power / hf_power) if hf_power > 0 else float("nan")
            indicators["LF_norm_HR"] = float(lf_power / total_power) if total_power > 0 else float("nan")
            indicators["HF_norm_HR"] = float(hf_power / total_power) if total_power > 0 else float("nan")
        except Exception:
            pass

    # ============== 心率衍生指标 ==============
    indicators["CV_HR"] = _cv(heart_rates_bpm)
    indicators["SampEn_HR"] = sample_entropy(heart_rates_bpm)

    # ============== 呼吸基础指标 ==============
    if breath_rates_bpm is not None and breath_rates_bpm.size > 0:
        indicators["avg_breath_rate"] = float(np.mean(breath_rates_bpm))
        try:
            indicators["baseline_breath_rate"] = float(mode(np.round(breath_rates_bpm).astype(int), keepdims=True)[0][0])
        except Exception:
            indicators["baseline_breath_rate"] = float("nan")
        indicators["min_breath_rate"] = float(np.min(breath_rates_bpm))
        indicators["max_breath_rate"] = float(np.max(breath_rates_bpm))

    # ============== 呼吸时域指标 ==============
    breath_intervals = _breath_intervals_seconds(breath_rates_bpm)
    if breath_intervals is not None and breath_intervals.size > 1:
        br_diff = np.diff(breath_intervals)
        indicators["RRV_diff"] = float(np.max(np.abs(br_diff))) if br_diff.size else float("nan")
        indicators["RRV"] = float(np.mean(np.abs(br_diff))) if br_diff.size else float("nan")
        indicators["SDRRI"] = float(np.std(breath_intervals, ddof=1))
        indicators["RMSSD_R"] = float(np.sqrt(np.mean(br_diff**2))) if br_diff.size else float("nan")
        indicators["Ve"] = float(np.mean(breath_rates_bpm) * tidal_volume_l) if breath_rates_bpm is not None else float("nan")

    # ============== 呼吸频域指标 ==============
    if breath_intervals is not None and breath_intervals.size > 10:
        try:
            rrv = breath_intervals - float(np.mean(breath_intervals))
            freqs, psd = signal.welch(rrv, fs=fs_hz, nperseg=min(256, int(rrv.size)))
            indicators["VLF_R"] = float(np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)], freqs[(freqs >= 0.003) & (freqs < 0.04)]))
            indicators["LF_R"] = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)]))
            indicators["HF_R"] = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)], freqs[(freqs >= 0.15) & (freqs < 0.4)]))
            hf_r = indicators["HF_R"]
            indicators["LF_HF_R"] = float(indicators["LF_R"] / hf_r) if np.isfinite(hf_r) and hf_r > 0 else float("nan")
        except Exception:
            pass

    # ============== 呼吸衍生指标 + RRCC ==============
    indicators["CV_R"] = _cv(breath_rates_bpm)
    indicators["SampEn_R"] = sample_entropy(breath_rates_bpm)

    if heart_rates_bpm is not None and breath_rates_bpm is not None:
        min_len = int(min(heart_rates_bpm.size, breath_rates_bpm.size))
        if min_len > 3:
            try:
                corr, _ = pearsonr(heart_rates_bpm[:min_len], breath_rates_bpm[:min_len])
                indicators["RRCC"] = float(corr)
            except Exception:
                indicators["RRCC"] = float("nan")

    # ============== Poincaré散点图参数 ==============
    if rr_intervals is not None and rr_intervals.size > 1:
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        diff_rr = rr_n1 - rr_n
        sum_rr = rr_n1 + rr_n

        indicators["SD1"] = float(np.std(diff_rr, ddof=1) / np.sqrt(2) * 1000.0)
        indicators["SD2"] = float(np.std(sum_rr, ddof=1) / np.sqrt(2) * 1000.0)

        center_x = float(np.mean(rr_n))
        center_y = float(np.mean(rr_n1))
        angles = np.arctan2(rr_n1 - center_y, rr_n - center_x)
        indicators["VAI"] = float(np.std(angles - np.pi / 4, ddof=1))

        distances = np.sqrt((rr_n - center_x) ** 2 + (rr_n1 - center_y) ** 2)
        indicators["VLI"] = float(np.mean(distances) * 1000.0)

    # ============== 自主神经评估 ==============
    indicators["ANS_activity"] = float(indicators.get("SDNN", np.nan))
    indicators["ANS_balance"] = float(indicators.get("LF_HF_HR", np.nan))
    rrcc = indicators.get("RRCC", np.nan)
    sampen_r = indicators.get("SampEn_R", np.nan)
    indicators["ANS_adaptability"] = float(sampen_r * (1.0 + abs(rrcc))) if np.isfinite(sampen_r) and np.isfinite(rrcc) else float("nan")

    vagal_tone = indicators.get("HF_norm_HR", np.nan)
    indicators["vagal_tone"] = float(vagal_tone)
    rmssd_hr = indicators.get("RMSSD_HR", np.nan)
    indicators["vagal_modulation"] = float(rmssd_hr * vagal_tone) if np.isfinite(rmssd_hr) and np.isfinite(vagal_tone) else float("nan")

    symp_tone = indicators.get("LF_norm_HR", np.nan)
    indicators["sympathetic_tone"] = float(symp_tone)
    sampen_hr = indicators.get("SampEn_HR", np.nan)
    indicators["sympathetic_modulation"] = float(sampen_hr * symp_tone) if np.isfinite(sampen_hr) and np.isfinite(symp_tone) else float("nan")

    return indicators


def calculate_chri_score(
    *,
    rmssd_ms: float | None,
    sampen_hr: float | None,
    hf_norm_ratio: float | None,
) -> float:
    """CHRI-Opt (Over60_Indicators_Distribution.ipynb), 0-100 (higher = lower risk)."""
    if rmssd_ms is None or sampen_hr is None or hf_norm_ratio is None:
        return float("nan")
    if not (np.isfinite(rmssd_ms) and np.isfinite(sampen_hr) and np.isfinite(hf_norm_ratio)):
        return float("nan")

    rmssd = float(max(rmssd_ms, 0.0))
    hfnorm = float(hf_norm_ratio * 100.0)  # convert 0-1 to percentage scale

    S_rmssd = float(100.0 * (1.0 - np.exp(-0.08 * rmssd)))
    S_rmssd = float(np.clip(S_rmssd, 0.0, 100.0))

    if sampen_hr < 0.3:
        S_en = 10.0
    elif sampen_hr < 0.7:
        S_en = 10.0 + (float(sampen_hr) - 0.3) * (70.0 / 0.4)
    else:
        S_en = 80.0 + (float(sampen_hr) - 0.7) * (20.0 / 0.3)
    S_en = float(np.clip(S_en, 0.0, 100.0))

    Z_hf = (hfnorm - 35.0) / 1.5
    S_hf = float(np.clip(50.0 + 20.0 * Z_hf, 0.0, 100.0))

    chri = 0.5 * S_rmssd + 0.35 * S_en + 0.15 * S_hf
    return float(np.clip(chri, 0.0, 100.0))


def calculate_row_indicators(row: dict[str, Any] | Any) -> dict[str, Any]:
    """Compute: 45 base indicators + 1 CHRI score (total 46 computed columns, excluding ids)."""
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    heart_rates = parse_sequence(row.get("heart_datas"))
    breath_rates = parse_sequence(row.get("breath_datas"))
    base = calculate_base_indicators(heart_rates, breath_rates)
    base["CHRI_score"] = calculate_chri_score(
        rmssd_ms=base.get("RMSSD_HR"),
        sampen_hr=base.get("SampEn_HR"),
        hf_norm_ratio=base.get("HF_norm_HR"),
    )
    return base
