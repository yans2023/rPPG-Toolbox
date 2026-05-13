#!/usr/bin/env python3
"""
rPPG 1分钟即时健康快检报告生成器

基于 rppg_1min_metrics_comparison.csv 中的 rPPG 指标，
生成五维度即时健康快照：
  ① 心理健康即时评估
  ② 运动健康即时评估
  ③ 慢病风险即时筛查
  ④ 自主神经即时评估
  ⑤ 中医体质即时辨识

设计文档: healthModel/Docs/rppg_1min_quick_report_design.md

用法:
  python rppg_1min_quick_report.py <input_csv> [--output <output_dir>] [--age <age>]

输入 CSV 可以是:
  - rppg_fiber_1min_metrics_comparison.csv (取 rppg_ 前缀列)
  - 或单源 1min 指标 CSV (直接列名，无前缀)
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from statistics import median


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clip(v, lo, hi):
    return max(lo, min(hi, v))


def normalize(v, lo, hi):
    """线性归一化到 0~100"""
    if hi == lo:
        return 50.0
    return clip((v - lo) / (hi - lo) * 100, 0, 100)


# ---------------------------------------------------------------------------
# HR 序列预处理 (v2)
# ---------------------------------------------------------------------------

def preprocess_hr_sequence(hr_values):
    """对原始 HR 序列做预处理，返回 (cleaned_hrs, noise_info)。

    步骤:
      1. MAD (Median Absolute Deviation) 异常检测
      2. 3点中值滤波平滑跳点
      3. 用邻近插值替代被标记的异常点

    输入: list[float|None]，None 表示缺失
    输出: (list[float|None], dict) — 清洗后序列 + 噪声诊断信息
    """
    valid = [h for h in hr_values if h is not None]
    n_total = len(hr_values)
    n_valid = len(valid)

    if n_valid < 5:
        return hr_values, {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_mad_outliers": 0,
            "n_jump_outliers": 0,
            "mad_value": 0,
            "noise_ratio": 1.0 if n_valid == 0 else 0.0,
            "noise_discount": 0.3 if n_valid < 3 else 0.7,
        }

    med = median(valid)
    abs_devs = [abs(h - med) for h in valid]
    mad = median(abs_devs) if abs_devs else 0
    # MAD threshold: 3 * MAD / 0.6745 ≈ 4.45 * MAD (robust z-score > 3)
    mad_threshold = max(4.45 * mad, 5.0)  # 至少 5 BPM 容差

    # --- Pass 1: MAD outlier marking ---
    outlier_mask = [False] * n_total
    mad_outlier_count = 0
    for i, h in enumerate(hr_values):
        if h is not None and abs(h - med) > mad_threshold:
            outlier_mask[i] = True
            mad_outlier_count += 1

    # --- Pass 2: Jump outlier detection (连续差 > 15 BPM) ---
    jump_outlier_count = 0
    prev_valid = None
    for i, h in enumerate(hr_values):
        if h is None:
            continue
        if prev_valid is not None and abs(h - prev_valid) > 15 and not outlier_mask[i]:
            outlier_mask[i] = True
            jump_outlier_count += 1
        if not outlier_mask[i]:
            prev_valid = h

    total_outliers = mad_outlier_count + jump_outlier_count
    noise_ratio = total_outliers / n_valid if n_valid > 0 else 0

    # --- Pass 3: 替换异常点 (邻近有效值插值) ---
    cleaned = list(hr_values)
    for i in range(n_total):
        if not outlier_mask[i]:
            continue
        # 找前后最近的非异常有效值
        left = right = None
        for j in range(i - 1, -1, -1):
            if cleaned[j] is not None and not outlier_mask[j]:
                left = cleaned[j]
                break
        for j in range(i + 1, n_total):
            if cleaned[j] is not None and not outlier_mask[j]:
                right = cleaned[j]
                break
        if left is not None and right is not None:
            cleaned[i] = (left + right) / 2
        elif left is not None:
            cleaned[i] = left
        elif right is not None:
            cleaned[i] = right
        # else: 保持原值

    # --- Pass 4: 3点中值滤波 ---
    smoothed = list(cleaned)
    for i in range(1, n_total - 1):
        vals = [cleaned[j] for j in (i - 1, i, i + 1) if cleaned[j] is not None]
        if len(vals) == 3:
            smoothed[i] = sorted(vals)[1]

    # --- 噪声降权系数 ---
    if noise_ratio >= 0.20:
        discount = 0.3
    elif noise_ratio >= 0.10:
        discount = 0.5
    elif noise_ratio >= 0.05:
        discount = 0.7
    else:
        discount = 1.0

    noise_info = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_mad_outliers": mad_outlier_count,
        "n_jump_outliers": jump_outlier_count,
        "mad_value": round(mad, 2),
        "noise_ratio": round(noise_ratio, 4),
        "noise_discount": discount,
    }

    return smoothed, noise_info


def compute_metrics_from_hr(hr_values, noise_info, expected=60):
    """从预处理后的 HR 序列计算全部 1min 指标，返回 dict。"""
    valid = [h for h in hr_values if h is not None]
    n = len(valid)
    missing = expected - n

    # 基础质量
    valid_ratio = n / expected if expected > 0 else 0
    if valid_ratio >= 0.80:
        qflag = "good"
    elif valid_ratio >= 0.60:
        qflag = "fair"
    else:
        qflag = "poor"

    result = {
        "quality_flag": qflag,
        "valid_ratio": valid_ratio,
        "valid_count": n,
        "missing_count": missing,
        "hr_outlier_count": noise_info["n_mad_outliers"] + noise_info["n_jump_outliers"],
    }

    if n < 5:
        # 填充零值避免下游 KeyError
        for k in ["avg_heart_rate", "median_heart_rate", "min_heart_rate",
                   "max_heart_rate", "heart_rate_range", "heart_rate_slope",
                   "heart_rate_delta", "SD_HR", "CV_HR", "MeanJJ", "SDNN",
                   "RMSSD_HR", "HRV_diff", "SD1", "SD2", "SD1_SD2_ratio",
                   "PNN50_experimental", "SampEn_HR_experimental"]:
            result[k] = 0
        return result

    import numpy as np
    hr = np.array(valid)

    avg_hr = float(np.mean(hr))
    median_hr = float(np.median(hr))
    min_hr = float(np.min(hr))
    max_hr = float(np.max(hr))
    hr_range = max_hr - min_hr
    sd_hr = float(np.std(hr, ddof=1)) if n > 1 else 0
    cv_hr = sd_hr / avg_hr if avg_hr > 0 else 0

    x = np.arange(n)
    slope = float(np.polyfit(x, hr, 1)[0]) * 60 if n > 1 else 0

    seg = max(1, n // 6)
    delta = float(np.mean(hr[-seg:]) - np.mean(hr[:seg]))

    rr = 60000.0 / hr
    mean_jj = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1)) if n > 1 else 0

    if n > 1:
        diff_rr = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
        hrv_diff = float(np.max(np.abs(diff_rr)))
        pnn50 = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100)
    else:
        rmssd = hrv_diff = pnn50 = 0

    if n > 1:
        diff_rr_arr = np.diff(rr)
        sd1 = float(np.std(diff_rr_arr, ddof=1) / math.sqrt(2))
        var_rr = np.std(rr, ddof=1) ** 2
        sd2_sq = 2 * var_rr - sd1 ** 2
        sd2 = float(math.sqrt(sd2_sq)) if sd2_sq > 0 else 0
        sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0
    else:
        sd1 = sd2 = sd1_sd2 = 0

    def sample_entropy(data, m=2, r_factor=0.2):
        N = len(data)
        if N < m + 2:
            return 0
        r = r_factor * np.std(data)
        if r == 0:
            return 0
        def count_matches(tl):
            count = 0
            templates = [data[i:i + tl] for i in range(N - tl)]
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 1
            return count
        A = count_matches(m + 1)
        B = count_matches(m)
        if B == 0:
            return 0
        return -math.log(A / B) if A > 0 else 0

    sampen = sample_entropy(hr)

    result.update({
        "avg_heart_rate": round(avg_hr, 2),
        "median_heart_rate": round(median_hr, 2),
        "min_heart_rate": round(min_hr, 2),
        "max_heart_rate": round(max_hr, 2),
        "heart_rate_range": round(hr_range, 2),
        "heart_rate_slope": round(slope, 4),
        "heart_rate_delta": round(delta, 4),
        "SD_HR": round(sd_hr, 4),
        "CV_HR": round(cv_hr, 6),
        "MeanJJ": round(mean_jj, 4),
        "SDNN": round(sdnn, 4),
        "RMSSD_HR": round(rmssd, 4),
        "HRV_diff": round(hrv_diff, 4),
        "SD1": round(sd1, 4),
        "SD2": round(sd2, 4),
        "SD1_SD2_ratio": round(sd1_sd2, 6),
        "PNN50_experimental": round(pnn50, 4),
        "SampEn_HR_experimental": round(sampen, 6),
    })
    return result


# ---------------------------------------------------------------------------
# CSV Reader — 兼容 comparison CSV (rppg_ 前缀) 和单源 CSV
# ---------------------------------------------------------------------------

def read_windows(csv_path):
    """读取 CSV，返回 list[dict]，每个 dict 的 key 为无前缀的指标名"""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        # 判断是否为 comparison CSV
        is_comparison = any(k.startswith("rppg_") for k in fieldnames)
        for r in reader:
            if not r.get("window_index", "").strip():
                continue
            d = {}
            for k, v in r.items():
                if is_comparison:
                    if k.startswith("rppg_"):
                        key = k[len("rppg_"):]
                        d[key] = v
                    elif k in ("window_index", "window_start", "window_end"):
                        d[k] = v
                else:
                    d[k] = v
            rows.append(d)
    return rows


def fval(row, key):
    return float(row[key])


# ---------------------------------------------------------------------------
# 评分引擎
# ---------------------------------------------------------------------------

def compute_report(row, age=None, noise_info=None):
    """对单个窗口计算五维度报告，返回结构化 dict。

    参数:
      row: dict — 指标行 (来自 CSV 或 compute_metrics_from_hr)
      age: int|None — 用户年龄
      noise_info: dict|None — 来自 preprocess_hr_sequence 的噪声诊断。
                  若提供则使用其 noise_discount；否则回退到旧的 outlier count 逻辑。
    """

    result = {}

    # --- 元数据 ---
    window_index = int(row.get("window_index", 0))
    window_start = row.get("window_start", "")
    window_end = row.get("window_end", "")
    qflag = row.get("quality_flag", "good")
    valid_ratio = fval(row, "valid_ratio")

    result["meta"] = {
        "window_index": window_index,
        "window_start": window_start,
        "window_end": window_end,
        "quality_flag": qflag,
        "valid_ratio": valid_ratio,
    }

    # 质量门控
    if qflag == "poor":
        result["blocked"] = True
        result["block_reason"] = "数据质量不足 (valid_ratio < 0.60)，无法输出健康评估"
        return result
    result["blocked"] = False

    # --- 读取指标 ---
    avg_hr = fval(row, "avg_heart_rate")
    median_hr = fval(row, "median_heart_rate")
    min_hr = fval(row, "min_heart_rate")
    max_hr = fval(row, "max_heart_rate")
    hr_range = fval(row, "heart_rate_range")
    slope = fval(row, "heart_rate_slope")
    delta = fval(row, "heart_rate_delta")
    sd_hr = fval(row, "SD_HR")
    cv_hr = fval(row, "CV_HR")
    sdnn = fval(row, "SDNN")
    rmssd = fval(row, "RMSSD_HR")
    hrv_diff = fval(row, "HRV_diff")
    sd1 = fval(row, "SD1")
    sd2 = fval(row, "SD2")
    sd1_sd2 = fval(row, "SD1_SD2_ratio")
    pnn50 = fval(row, "PNN50_experimental")
    sampen = fval(row, "SampEn_HR_experimental")
    outlier = fval(row, "hr_outlier_count")

    result["vitals"] = {
        "avg_heart_rate": round(avg_hr, 1),
        "median_heart_rate": round(median_hr, 1),
        "min_heart_rate": round(min_hr, 1),
        "max_heart_rate": round(max_hr, 1),
        "heart_rate_range": round(hr_range, 1),
        "heart_rate_slope": round(slope, 2),
    }

    # --- 噪声降权 (v2: 优先使用 MAD-based noise_info) ---
    if noise_info is not None:
        noise_discount = noise_info["noise_discount"]
        noise_ratio = noise_info["noise_ratio"]
        n_outliers = noise_info["n_mad_outliers"] + noise_info["n_jump_outliers"]
    else:
        # 回退: 旧逻辑 (兼容已有 CSV 输入)
        n_outliers = int(outlier)
        noise_ratio = n_outliers / max(fval(row, "valid_count"), 1)
        if noise_ratio >= 0.20:
            noise_discount = 0.3
        elif noise_ratio >= 0.10:
            noise_discount = 0.5
        elif noise_ratio >= 0.05:
            noise_discount = 0.7
        elif n_outliers >= 1:
            noise_discount = 0.8
        else:
            noise_discount = 1.0

    eff_sdnn = sdnn * noise_discount
    eff_rmssd = rmssd * noise_discount
    eff_sd1 = sd1 * noise_discount
    eff_sd2 = sd2 * noise_discount
    eff_hrv_diff = hrv_diff * noise_discount
    eff_hr_range = hr_range * noise_discount

    result["noise"] = {
        "outlier_count": n_outliers,
        "noise_ratio": round(noise_ratio, 4),
        "noise_discount": noise_discount,
    }

    # ===================================================================
    # ① 心理健康即时评估
    # ===================================================================

    # 1.1 即时紧张度
    hr_deviation = max(0, median_hr - 72)
    slope_act = max(0, slope)
    tension = clip(
        normalize(hr_deviation, 0, 25) * 0.35
        + normalize(slope_act, 0, 10) * 0.30
        + normalize(sd_hr, 1, 12) * 0.35,
        0, 100,
    )
    tension_labels = [(30, "放松"), (55, "平稳"), (75, "轻度紧张"), (100, "明显紧张")]
    tension_label = next(l for t, l in tension_labels if tension <= t)

    # 1.2 即时放松度
    hr_calm = clip(100 - normalize(abs(median_hr - 62), 0, 20) * 100 / 100, 0, 100)
    slope_calm = normalize(-slope, -3, 5)
    if sd_hr < 3 and 55 <= median_hr <= 72:
        var_calm = 80  # 极稳定 + 正常心率 = 深度平静
    else:
        var_calm = normalize(eff_rmssd, 8, 45)
    relaxation = clip(hr_calm * 0.35 + slope_calm * 0.30 + var_calm * 0.35, 0, 100)
    relax_labels = [(25, "紧绷"), (50, "一般"), (75, "较放松"), (100, "深度放松")]
    relax_label = next(l for t, l in relax_labels if relaxation <= t)

    # 1.3 心率波动敏感性
    cv_norm = normalize(cv_hr, 0.01, 0.12)
    hvd_norm = normalize(eff_hrv_diff, 30, 400)
    fluct = (cv_norm + hvd_norm) / 2
    fluct_label = "低" if fluct <= 33 else ("中" if fluct <= 66 else "高")

    result["mental"] = {
        "tension_score": round(tension, 1),
        "tension_label": tension_label,
        "relaxation_score": round(relaxation, 1),
        "relaxation_label": relax_label,
        "fluctuation_level": fluct_label,
    }

    # ===================================================================
    # ② 运动健康即时评估
    # ===================================================================

    # 2.1 心率储备利用率
    est_max = (208 - 0.7 * age) if age else 185
    resting_est = min_hr
    if est_max > resting_est:
        reserve_util = clip(
            (avg_hr - resting_est) / (est_max - resting_est) * 100, 0, 100
        )
    else:
        reserve_util = 0
    reserve_labels = [
        (20, "静息态"), (50, "轻度活动"), (70, "中度活动"),
        (85, "高强度"), (100, "极限区"),
    ]
    reserve_label = next(l for t, l in reserve_labels if reserve_util <= t)

    # 2.2 心率恢复趋势
    if delta < -3 and slope < 0:
        recovery_label = "恢复中"
    elif delta > 3 and slope > 0:
        recovery_label = "激活中"
    else:
        recovery_label = "稳定"

    # 2.3 运动耐受提示
    sdnn_tol = normalize(eff_sdnn, 15, 70) * 0.50
    if eff_hr_range <= 25:
        range_tol = normalize(eff_hr_range, 3, 25) * 0.30
    else:
        range_tol = normalize(50 - eff_hr_range, 0, 25) * 0.30
    hr_zone = normalize(100 - abs(avg_hr - 68), 70, 100) * 0.20
    tolerance = clip(sdnn_tol + range_tol + hr_zone, 0, 100)
    tol_label = "良好" if tolerance >= 50 else ("一般" if tolerance >= 25 else "需关注")

    result["fitness"] = {
        "reserve_utilization": round(reserve_util, 1),
        "reserve_label": reserve_label,
        "recovery_tendency": recovery_label,
        "tolerance_label": tol_label,
    }

    # ===================================================================
    # ③ 慢病风险即时筛查
    # ===================================================================

    # 3.1 心血管应激指数
    hr_stress = normalize(avg_hr, 60, 100) * 0.40
    sd_stress = normalize(sd_hr, 2, 12) * 0.30
    range_stress = normalize(eff_hr_range, 5, 35) * 0.30
    cv_stress = clip(hr_stress + sd_stress + range_stress, 0, 100)
    cvs_labels = [(30, "低应激"), (55, "正常范围"), (75, "轻度应激"), (100, "明显应激")]
    cvs_label = next(l for t, l in cvs_labels if cv_stress <= t)

    # 3.2 代谢负荷提示
    if reserve_util <= 20:
        if avg_hr > 80 and eff_rmssd < 15:
            metab_label = "需关注"
        elif avg_hr > 75 and eff_rmssd < 20:
            metab_label = "偏高"
        else:
            metab_label = "正常"
    else:
        metab_label = "—（非静息态）"

    # 3.3 节律异常提示
    anomaly_score = 0
    if outlier >= 3:
        anomaly_score += 2
    elif outlier >= 1:
        anomaly_score += 1
    if eff_hrv_diff > 250:
        anomaly_score += 2
    elif eff_hrv_diff > 120:
        anomaly_score += 1
    if pnn50 > 15:
        anomaly_score += 1
    rhythm_label = (
        "建议复查" if anomaly_score >= 3
        else ("轻度波动" if anomaly_score >= 1 else "未见异常")
    )

    chronic_reliable = qflag == "good"
    result["chronic"] = {
        "cv_stress_score": round(cv_stress, 1),
        "cv_stress_label": cvs_label,
        "metabolic_label": metab_label,
        "rhythm_label": rhythm_label,
        "reliable": chronic_reliable,
    }

    # ===================================================================
    # ④ 自主神经即时评估
    # ===================================================================

    # 4.1 即时 ANS 活性
    if eff_sdnn > 0:
        ans_activity = clip(math.log(eff_sdnn) / math.log(160) * 100, 0, 100)
    else:
        ans_activity = 0
    ans_labels = [
        (25, "活性偏低"), (50, "活性一般"), (75, "活性良好"), (100, "活性旺盛"),
    ]
    ans_act_label = next(l for t, l in ans_labels if ans_activity <= t)

    # 4.2 即时神经平衡趋势
    eff_ratio = (eff_sd1 / eff_sd2) if eff_sd2 > 0 else sd1_sd2
    if eff_ratio > 0.50:
        bal_label = "副交感偏强"
    elif eff_ratio < 0.20:
        bal_label = "交感偏强"
    else:
        bal_label = "平衡"

    # 4.3 即时调节弹性
    rmssd_res = normalize(eff_rmssd, 5, 50) * 0.65
    range_res = normalize(eff_hr_range, 3, 25) * 0.35
    resilience = clip(rmssd_res + range_res, 0, 100)
    if sd_hr < 3 and 55 <= median_hr <= 72 and resilience < 40:
        resilience = 40  # 平静稳定态兜底
    res_labels = [
        (30, "弹性偏低"), (55, "弹性一般"), (75, "弹性良好"), (100, "弹性优秀"),
    ]
    res_label = next(l for t, l in res_labels if resilience <= t)

    result["ans"] = {
        "activity_score": round(ans_activity, 1),
        "activity_label": ans_act_label,
        "balance_label": bal_label,
        "resilience_score": round(resilience, 1),
        "resilience_label": res_label,
    }

    # ===================================================================
    # ⑤ 中医体质即时辨识
    # ===================================================================

    TCM_RULES = {
        "平和质": lambda: sum([
            58 <= avg_hr <= 75,
            1 <= sd_hr <= 5,
            10 <= eff_rmssd <= 45,
            abs(slope) < 3,
        ]),
        "气虚质": lambda: sum([
            avg_hr < 62,
            eff_sdnn < 25,
            eff_rmssd < 15,
            cv_hr < 0.03,
        ]),
        "阳虚质": lambda: sum([
            avg_hr < 62,
            eff_hr_range < 8,
            eff_sd1 < 10,
            eff_sdnn < 25,
        ]),
        "阴虚质": lambda: sum([
            avg_hr > 75,
            eff_rmssd < 20,
            eff_ratio < 0.20,
            slope > 2,
        ]),
        "痰湿质": lambda: sum([
            avg_hr > 65,
            sd_hr > 5,
            eff_hrv_diff > 120,
            eff_hr_range > 15,
        ]),
        "湿热质": lambda: sum([
            avg_hr > 72,
            slope > 2,
            eff_hr_range > 15,
            eff_rmssd < 25,
        ]),
        "血瘀质": lambda: sum([
            58 <= avg_hr <= 75,
            eff_rmssd > 35,
            pnn50 > 8,
            eff_sd1 > 25,
        ]),
        "气郁质": lambda: sum([
            58 <= avg_hr <= 80,
            sd_hr > 5,
            sampen > 0.5,
            abs(slope) > 3,
        ]),
    }

    TCM_TIPS = {
        "平和质": "保持规律作息，适度运动，均衡饮食",
        "气虚质": "适度有氧运动，避免过度劳累，注意保暖",
        "阳虚质": "注意保暖，适当温补，避免生冷食物",
        "阴虚质": "避免熬夜，注意补充水分，少食辛辣",
        "痰湿质": "加强运动，饮食清淡，少食油腻甜食",
        "湿热质": "清淡饮食，避免辛辣油腻，适当排汗运动",
        "血瘀质": "适度运动促进循环，避免久坐，注意保暖",
        "气郁质": "保持心情舒畅，适当户外运动，规律作息",
    }

    matches = {name: fn() for name, fn in TCM_RULES.items()}
    sorted_m = sorted(matches.items(), key=lambda x: (-x[1], x[0]))
    primary = sorted_m[0]
    secondary = sorted_m[1] if sorted_m[1][1] >= 2 else None

    def match_level(score):
        return "高" if score >= 4 else ("中" if score >= 3 else "低")

    tcm_reliable = qflag == "good"
    result["tcm"] = {
        "primary": primary[0],
        "primary_match": primary[1],
        "primary_level": match_level(primary[1]),
        "secondary": secondary[0] if secondary else None,
        "secondary_match": secondary[1] if secondary else None,
        "secondary_level": match_level(secondary[1]) if secondary else None,
        "tip": TCM_TIPS[primary[0]],
        "reliable": tcm_reliable,
    }

    # ===================================================================
    # 置信度标注
    # ===================================================================
    # 每个维度的置信度由三个因素决定:
    #   1. 数据质量 (quality_flag)
    #   2. 噪声水平 (noise_discount)
    #   3. 该维度对 HRV 指标的依赖程度
    #
    # 依赖程度:
    #   ① 心理健康: 中 (紧张度主要用 HR 水平+slope, 放松度用 RMSSD)
    #   ② 运动健康: 低 (主要用 HR 水平指标)
    #   ③ 慢病风险: 中-高 (节律异常依赖 HRV_diff, 代谢依赖 RMSSD)
    #   ④ 自主神经: 高 (全部依赖 SDNN/RMSSD/SD1/SD2)
    #   ⑤ 中医体质: 高 (多条规则依赖 RMSSD/SDNN/SD1)

    def _confidence(hrv_dependency_weight):
        """hrv_dependency_weight: 0~1, 该维度对 HRV 指标的依赖比例"""
        # 基础分: quality_flag
        base = {"good": 1.0, "fair": 0.7, "poor": 0.3}.get(qflag, 0.5)
        # 噪声惩罚: 仅按 HRV 依赖比例施加
        noise_penalty = (1.0 - noise_discount) * hrv_dependency_weight
        score = base - noise_penalty
        if score >= 0.75:
            return "高"
        elif score >= 0.50:
            return "中"
        else:
            return "低"

    result["confidence"] = {
        "mental": _confidence(0.4),
        "fitness": _confidence(0.15),
        "chronic": _confidence(0.5),
        "ans": _confidence(0.85),
        "tcm": _confidence(0.7),
    }

    return result


# ---------------------------------------------------------------------------
# 文本报告格式化
# ---------------------------------------------------------------------------

def _conf_icon(level):
    """置信度等级 → 图标"""
    return {"高": "●", "中": "◐", "低": "○"}[level]


def format_text_report(r):
    """将结构化 dict 格式化为文本报告"""
    m = r["meta"]
    stars = {"good": "★★★★☆", "fair": "★★★☆☆", "poor": "★★☆☆☆"}[m["quality_flag"]]

    lines = []
    lines.append("┌──────────────────────────────────────────────────────────────┐")
    lines.append(f"│  rPPG 1分钟即时健康快照  [窗口 {m['window_index']}]")
    lines.append(f"│  采集: {m['window_start']} ~ {m['window_end']}")
    lines.append(f"│  数据质量: {stars} ({m['quality_flag']})  有效率: {m['valid_ratio']:.1%}")
    lines.append("├──────────────────────────────────────────────────────────────┤")

    if r["blocked"]:
        lines.append(f"│  ⚠ {r['block_reason']}")
        lines.append("│  建议重新采集，保持面部稳定、光线充足")
        lines.append("└──────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)

    v = r["vitals"]
    n = r["noise"]
    conf = r["confidence"]

    lines.append("│  基础体征")
    lines.append(f"│    平均心率: {v['avg_heart_rate']:5.1f} BPM   中位心率: {v['median_heart_rate']:5.1f} BPM")
    lines.append(f"│    最低心率: {v['min_heart_rate']:5.1f} BPM   最高心率: {v['max_heart_rate']:5.1f} BPM")
    lines.append(f"│    心率极差: {v['heart_rate_range']:5.1f} BPM  心率斜率: {v['heart_rate_slope']:+6.2f} BPM/min")

    if n["outlier_count"] >= 1:
        lines.append(f"│  ℹ 检测到 {n['outlier_count']} 个异常跳点 (噪声比: {n['noise_ratio']:.1%})，降权系数: {n['noise_discount']}")

    lines.append("├──────────────────────────────────────────────────────────────┤")

    # ① 心理健康
    me = r["mental"]
    ci = _conf_icon(conf["mental"])
    lines.append(f"│  ① 心理健康即时评估  {ci} 置信度: {conf['mental']}")
    lines.append(f"│    即时紧张度:     {me['tension_score']:5.1f} / 100  [{me['tension_label']}]")
    lines.append(f"│    即时放松度:     {me['relaxation_score']:5.1f} / 100  [{me['relaxation_label']}]")
    lines.append(f"│    心率波动敏感性: {me['fluctuation_level']}")
    lines.append("│")

    # ② 运动健康
    fi = r["fitness"]
    ci = _conf_icon(conf["fitness"])
    lines.append(f"│  ② 运动健康即时评估  {ci} 置信度: {conf['fitness']}")
    lines.append(f"│    心率储备利用率: {fi['reserve_utilization']:5.1f}%     [{fi['reserve_label']}]")
    lines.append(f"│    心率恢复趋势:   {fi['recovery_tendency']}")
    lines.append(f"│    运动耐受提示:   {fi['tolerance_label']}")
    lines.append("│")

    # ③ 慢病风险
    ch = r["chronic"]
    ci = _conf_icon(conf["chronic"])
    lines.append(f"│  ③ 慢病风险即时筛查  {ci} 置信度: {conf['chronic']}")
    lines.append(f"│    心血管应激指数: {ch['cv_stress_score']:5.1f} / 100  [{ch['cv_stress_label']}]")
    lines.append(f"│    代谢负荷提示:   {ch['metabolic_label']}")
    lines.append(f"│    节律异常提示:   {ch['rhythm_label']}")
    lines.append("│")

    # ④ 自主神经
    an = r["ans"]
    ci = _conf_icon(conf["ans"])
    lines.append(f"│  ④ 自主神经即时评估  {ci} 置信度: {conf['ans']}")
    lines.append(f"│    即时ANS活性:    {an['activity_score']:5.1f} / 100  [{an['activity_label']}]")
    lines.append(f"│    神经平衡趋势:   {an['balance_label']}")
    lines.append(f"│    即时调节弹性:   {an['resilience_score']:5.1f} / 100  [{an['resilience_label']}]")
    lines.append("│")

    # ⑤ 中医体质
    tc = r["tcm"]
    ci = _conf_icon(conf["tcm"])
    lines.append(f"│  ⑤ 中医体质即时辨识  {ci} 置信度: {conf['tcm']}")
    lines.append(f"│    主要倾向: {tc['primary']} (匹配度: {tc['primary_level']}, {tc['primary_match']}/4项)")
    if tc["secondary"]:
        lines.append(f"│    次要倾向: {tc['secondary']} (匹配度: {tc['secondary_level']}, {tc['secondary_match']}/4项)")
    else:
        lines.append("│    次要倾向: 无明显次要倾向")
    lines.append(f"│    调养提示: {tc['tip']}")

    lines.append("├──────────────────────────────────────────────────────────────┤")
    lines.append("│  置信度图例: ● 高  ◐ 中  ○ 低")
    lines.append("│  ⚠ 本报告为1分钟即时状态快照，非临床诊断")
    lines.append("│    HRV相关指标为HR推导的代理值，非心电级别")
    lines.append("└──────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="rPPG 1分钟即时健康快检报告")
    parser.add_argument("input_csv", help="1min 指标 CSV 文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录 (默认: 输入文件同目录)")
    parser.add_argument("--age", type=int, default=None, help="用户年龄 (用于最大心率估算)")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 格式")
    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"错误: 文件不存在 {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or os.path.dirname(os.path.abspath(args.input_csv))
    os.makedirs(output_dir, exist_ok=True)

    windows = read_windows(args.input_csv)
    if not windows:
        print("错误: CSV 中无有效窗口数据", file=sys.stderr)
        sys.exit(1)

    reports = []
    text_parts = []

    for row in windows:
        r = compute_report(row, age=args.age)
        reports.append(r)
        text_parts.append(format_text_report(r))

    # 输出文本报告
    text_output = "\n\n".join(text_parts)
    txt_path = os.path.join(output_dir, "rppg_1min_quick_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_output + "\n")
    print(text_output)
    print(f"\n报告已保存: {txt_path}")

    # 输出 JSON
    if args.json:
        json_path = os.path.join(output_dir, "rppg_1min_quick_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
