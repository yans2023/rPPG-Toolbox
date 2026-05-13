#!/usr/bin/env python3
"""根据 `healthModel/1minReport/1min_HRV_Indicator_Design_with_formula.xlsx` 中的伪代码，
从 1 分钟（N=60，每秒 1 点）的 BPM 序列计算各产品指标。

说明:
  - 输入为长度 60 的 hr_bpm 数组（与 Excel 一致）。
  - 「即时活力指数」依赖「即时放松度」；Excel 表内未给出放松度闭式公式，本脚本按
    `healthModel/Docs/rppg_1min_quick_report_design.md` 1.2 的思路，用 RMSSD、心率斜率、
    SD1/SD2 比三项线性归一化后加权（0.35 / 0.35 / 0.30）得到 relaxation_score（0~100）。

用法示例:
  python healthModel/scripts/hrv_1min_excel_indicators.py \\
    --data-dir test_output/20260420-rPPG-fiber \\
    --n-samples 10 \\
    --seed 42 \\
    --out test_output/20260420-rPPG-fiber/hrv_excel_indicator_sample10.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def normalize_linear(v: float, lo: float, hi: float) -> float:
    """线性映射到 0~100（与项目中 quick_report.normalize 一致）。"""
    if hi == lo:
        return 50.0
    return clip((v - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def std_ddof1(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size < 2:
        return float("nan")
    return float(np.std(a, ddof=1))


def linregress_slope_hr_per_sec(hr: np.ndarray) -> float:
    hr = np.asarray(hr, dtype=float)
    n = hr.size
    if n < 2:
        return float("nan")
    t = np.arange(n, dtype=float)
    return float(np.polyfit(t, hr, 1)[0])


def rr_proxy_ms(hr_bpm: np.ndarray) -> np.ndarray:
    h = np.asarray(hr_bpm, dtype=float)
    return 60000.0 / h


def relaxation_score_excel_upstream(
    hr: np.ndarray, slope_bpm_per_sec: float, rmssd_ms: float, sd1_sd2_ratio: float
) -> float:
    """Excel 未给公式时的放松度兜底（与设计文档 1.2 对齐的可计算版本）。"""
    slope_bpm_per_min = slope_bpm_per_sec * 60.0
    return clip(
        0.35 * normalize_linear(rmssd_ms, 8.0, 45.0)
        + 0.35 * normalize_linear(-slope_bpm_per_min, -3.0, 5.0)
        + 0.30 * normalize_linear(sd1_sd2_ratio, 0.15, 0.55),
        0.0,
        100.0,
    )


def compute_sd1_sd2_poincare(rr_ms: np.ndarray) -> tuple[float, float, float]:
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 2:
        return float("nan"), float("nan"), float("nan")
    diff_rr = np.diff(rr)
    sd1 = std_ddof1(diff_rr) / math.sqrt(2.0)
    sdnn = std_ddof1(rr)
    sd2_sq = max(2.0 * sdnn**2 - sd1**2, 0.0)
    sd2 = math.sqrt(sd2_sq)
    ratio = sd1 / sd2 if sd2 > 0 else 0.0
    return float(sd1), float(sd2), float(ratio)


# =========================================================================
# 指标解释文案库 — 对照 Excel「解释说明（用户视角）」「当日行动建议」
# 每个指标: {level_key: {"explanation": ..., "advice": ...}}
# =========================================================================

_READINESS_TEXT: dict[str, dict[str, str]] = {
    "优秀": {
        "explanation": "您当前身心状态极佳，身体的自我调节能力处于高水平，精力充沛。",
        "advice": "适合进行高强度工作、运动挑战或重要决策。",
    },
    "良好": {
        "explanation": "您的即时身心状态不错，身体具备良好的应对能力。",
        "advice": "可正常安排各类活动，适合中高强度任务。",
    },
    "一般": {
        "explanation": "您当前处于正常状态，身体调节能力中等。",
        "advice": "保持当前节奏即可，避免同时叠加多项高压任务。",
    },
    "偏低": {
        "explanation": "您当前的身心活力偏低，身体恢复和调节的余量不足。",
        "advice": "建议适当休息或进行轻度活动（散步、拉伸），避免重要决策和高强度运动。",
    },
}

_STRESS_TEXT: dict[str, dict[str, str]] = {
    "放松": {
        "explanation": "您此刻处于放松平静的状态，身心压力很低。",
        "advice": "可能处于低唤醒状态，适合创意性或轻松的工作。",
    },
    "平稳": {
        "explanation": "您当前处于正常的工作/生活状态，压力水平适中。",
        "advice": "这是日常活动的正常状态，无需特别干预。",
    },
    "轻度紧张": {
        "explanation": "您的身体出现了一定的紧张反应，心率有波动或上升趋势。",
        "advice": "建议适当放松，可尝试几次深呼吸或短暂休息。",
    },
    "明显紧张": {
        "explanation": "您此刻处于较高的紧张或兴奋状态，身体激活程度明显。",
        "advice": "建议暂停高压任务，进行5分钟深呼吸或冥想，让身体缓和下来。",
    },
}

_FLUCTUATION_TEXT: dict[str, dict[str, str]] = {
    "低": {
        "explanation": "您的心率非常平稳，对外界刺激的即时反应较小。",
        "advice": "心率平稳，适合需要专注力的工作或安静休息。",
    },
    "中": {
        "explanation": "您的心率存在正常范围内的波动，属于健康的生理节律。",
        "advice": "属于正常表现，无需特别关注。",
    },
    "高": {
        "explanation": "您的心率波动较大，可能与情绪变化、体动或环境刺激有关。",
        "advice": "建议减少外界干扰，保持稳定姿势；如持续出现，可留意是否与情绪或身体不适相关。",
    },
}

_HRR_TEXT: dict[str, dict[str, str]] = {
    "静息态": {
        "explanation": "您当前心率接近静息水平，身体处于安静休息状态。",
        "advice": "适合放松和恢复，也适合进行冥想或阅读。",
    },
    "轻度活动": {
        "explanation": "您的心率略高于静息，相当于散步或轻度日常活动的水平。",
        "advice": "适合散步、拉伸等轻度活动。",
    },
    "中度活动": {
        "explanation": "您的心率处于有氧运动区间，身体正在进行中等强度的活动。",
        "advice": "正处于有效的有氧训练区间，可继续保持。",
    },
    "高强度": {
        "explanation": "您的心率已进入高强度区间，心脏正在承受较大负荷。",
        "advice": "注意自身感受，适时降低运动强度。",
    },
    "极限区": {
        "explanation": "您的心率接近最大值，身体处于极限负荷状态。",
        "advice": "请注意安全，建议立即降低强度或停止运动，及时补水休息。",
    },
}

_RECOVERY_TEXT: dict[str, dict[str, str]] = {
    "恢复中": {
        "explanation": "您的心率在这一分钟内呈下降趋势，身体正从之前的活动或压力中恢复。",
        "advice": "恢复进行中，无需额外干预；如刚运动完，这是正常的恢复信号。",
    },
    "稳定": {
        "explanation": "您的心率在这一分钟内保持平稳，既没有明显上升也没有下降。",
        "advice": "正常的稳态表现，身体状况平稳。",
    },
    "激活中": {
        "explanation": "您的心率在这一分钟内呈上升趋势，身体正在进入更活跃的状态。",
        "advice": "若非主动运动导致，建议关注是否有压力源或身体不适。",
    },
}

_TOLERANCE_TEXT: dict[str, dict[str, str]] = {
    "良好": {
        "explanation": "您的心脏调节能力较好，当前身体对运动负荷的承受力充足。",
        "advice": "今日可进行中高强度训练或体力活动。",
    },
    "一般": {
        "explanation": "您当前的身体承受能力中等，心脏调节余量一般。",
        "advice": "建议以有氧运动为主，避免高强度力量训练。",
    },
    "需关注": {
        "explanation": "您的心脏调节能力偏低，当前不太适合高负荷活动。",
        "advice": "建议以恢复性运动为主（散步、瑜伽），充分休息后再增加运动量。",
    },
}

_CV_STRESS_TEXT: dict[str, dict[str, str]] = {
    "低应激": {
        "explanation": "您的心血管系统当前负荷很低，心脏工作轻松。",
        "advice": "心血管状态良好，可正常安排各项活动。",
    },
    "正常范围": {
        "explanation": "您的心血管系统承受适度负荷，属于日常正常水平。",
        "advice": "维持当前作息即可，注意保持充足饮水。",
    },
    "轻度应激": {
        "explanation": "您的心血管系统当前承受一定压力，心率或波动偏高。",
        "advice": "建议减少咖啡因摄入，避免剧烈运动，保持情绪平稳。",
    },
    "明显应激": {
        "explanation": "您的心血管系统当前负荷较重，心率和波动均处于偏高水平。",
        "advice": "建议立即休息，避免一切高强度活动；若经常出现此提示，请咨询医生。",
    },
}

_METABOLIC_TEXT: dict[str, dict[str, str]] = {
    "正常": {
        "explanation": "在静息状态下，您的心率和心率变异性组合正常，未见明显代谢压力。",
        "advice": "维持当前饮食和作息即可。",
    },
    "偏高": {
        "explanation": "在静息状态下，出现心率偏高或心率调节能力下降的信号，提示身体可能有一定代谢负担。",
        "advice": "建议今日减少高糖高脂饮食，增加饮水量，餐后适当散步。",
    },
    "需关注": {
        "explanation": "在静息状态下，心率偏高且心率调节能力明显降低，提示较重的代谢压力信号。",
        "advice": "建议关注近期饮食、睡眠和血糖水平；如持续出现，请进行专业体检。",
    },
    "非静息态，本指标不适用": {
        "explanation": "当前心率未处于静息水平，代谢负荷评估需要安静状态下测量才有参考价值。",
        "advice": "如需评估，请在安静坐姿下重新测量。",
    },
}

_RHYTHM_TEXT: dict[str, dict[str, str]] = {
    "未见异常": {
        "explanation": "心率节律平稳，未检测到明显的异常跳动。",
        "advice": "节律表现正常，无需额外关注。",
    },
    "轻度波动": {
        "explanation": "检测到少量心率跳动异常，可能与体动、呼吸或一过性干扰有关。",
        "advice": "多数情况属于正常生理波动；建议减少咖啡因，保持规律作息，次日复测观察。",
    },
    "建议复查": {
        "explanation": "检测到较多心率节律异常信号，虽然不等于心律失常，但建议引起重视。",
        "advice": "建议使用专业心电设备（如 Apple Watch ECG 或医用心电图）进行确认；注意：本结果不构成医学诊断。",
    },
}

_ANS_ACTIVITY_TEXT: dict[str, dict[str, str]] = {
    "活性旺盛": {
        "explanation": "您的自主神经系统非常活跃，身体的自我调节能力处于高水平。",
        "advice": "身体状态佳，适合迎接挑战性任务或中高强度运动。",
    },
    "活性良好": {
        "explanation": "您的自主神经系统调节能力正常偏好，身体对变化有不错的适应力。",
        "advice": "可正常安排工作和运动。",
    },
    "活性一般": {
        "explanation": "您的自主神经系统活跃程度中等，调节能力在正常范围的低段。",
        "advice": "建议注意作息规律，避免过度疲劳。",
    },
    "活性偏低": {
        "explanation": "您的自主神经系统活跃程度偏低，身体的适应和恢复能力可能不足。",
        "advice": "建议今日减少高强度任务，优先保证充足睡眠和营养。",
    },
}

_ANS_BALANCE_TEXT: dict[str, dict[str, str]] = {
    "交感偏强": {
        "explanation": "您当前交感神经（负责战斗或逃跑反应）相对更活跃，身体偏向激活/应激模式。",
        "advice": "建议进行放松活动，如冥想、深呼吸或舒缓的音乐。",
    },
    "平衡": {
        "explanation": "您的交感和副交感神经处于良好的平衡状态，身体既有活力又有恢复能力。",
        "advice": "最佳状态，适合各类活动。",
    },
    "副交感偏强": {
        "explanation": "您当前副交感神经（负责休息与恢复）相对更活跃，身体偏向深度放松模式。",
        "advice": "适合休息和恢复；如需提升精力，可适度活动或进行轻度运动。",
    },
}

_RESILIENCE_TEXT: dict[str, dict[str, str]] = {
    "优秀": {
        "explanation": "您的神经系统快速响应能力极强，身体能迅速适应外界变化。",
        "advice": "适合高压、高强度的工作场景和运动挑战。",
    },
    "良好": {
        "explanation": "您的神经调节弹性不错，面对压力变化时身体能较好地适应。",
        "advice": "身体适应能力正常，可自信安排各项活动。",
    },
    "一般": {
        "explanation": "您的神经调节弹性处于中等水平，应对突然变化时可能需要更多时间调整。",
        "advice": "建议减少多任务切换，给身体留出调节空间。",
    },
    "偏低": {
        "explanation": "您的神经快速调节能力较弱，身体对突发压力的响应偏慢。",
        "advice": "建议今日避免情绪化决策，减少突发性高压任务，保持环境稳定。",
    },
}


def _level_readiness(score: float) -> str:
    if score > 70:
        return "优秀"
    if score > 50:
        return "良好"
    if score >= 40:
        return "一般"
    return "偏低"


def _level_stress(score: float) -> str:
    if score <= 30:
        return "放松"
    if score <= 55:
        return "平稳"
    if score <= 75:
        return "轻度紧张"
    return "明显紧张"


def _level_hrr(pct: float) -> str:
    if pct <= 20:
        return "静息态"
    if pct <= 50:
        return "轻度活动"
    if pct <= 70:
        return "中度活动"
    if pct <= 85:
        return "高强度"
    return "极限区"


def _level_cv_stress(score: float) -> str:
    if score <= 30:
        return "低应激"
    if score <= 55:
        return "正常范围"
    if score <= 75:
        return "轻度应激"
    return "明显应激"


def _level_ans_activity(score: float) -> str:
    if score > 75:
        return "活性旺盛"
    if score > 50:
        return "活性良好"
    if score >= 25:
        return "活性一般"
    return "活性偏低"


def _level_resilience(score: float) -> str:
    if score > 75:
        return "优秀"
    if score > 55:
        return "良好"
    if score >= 30:
        return "一般"
    return "偏低"


def _make_indicator(
    name: str,
    name_en: str,
    score: float | None,
    level: str,
    text_map: dict[str, dict[str, str]],
    *,
    unit: str = "",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建单个指标的标准化输出，包含 score/level/explanation/advice。"""
    texts = text_map.get(level, {"explanation": "", "advice": ""})
    result: dict[str, Any] = {
        "name": name,
        "name_en": name_en,
        "score": round(score, 1) if score is not None and math.isfinite(score) else None,
        "unit": unit,
        "level": level,
        "explanation": texts["explanation"],
        "advice": texts["advice"],
    }
    if detail:
        result["detail"] = detail
    return result


# =========================================================================
# 综合解读生成
# =========================================================================

def _generate_comprehensive_summary(indicators: dict[str, dict[str, Any]]) -> str:
    """根据所有指标结果，拼装 3~5 句用户可读的综合解读段落。"""
    parts: list[str] = []

    readiness = indicators["instant_readiness"]
    stress = indicators["instant_stress_index"]
    fluctuation = indicators["hr_fluctuation_sensitivity"]
    reserve = indicators["hr_reserve_utilization"]
    recovery = indicators["hr_recovery_tendency"]
    tolerance = indicators["exercise_tolerance_hint"]
    cv_stress = indicators["cv_stress_signal"]
    metabolic = indicators["metabolic_load_hint"]
    rhythm = indicators["rhythm_anomaly_hint"]
    ans_act = indicators["instant_ans_activity"]
    ans_bal = indicators["instant_ans_balance"]
    resilience = indicators["instant_regulatory_resilience"]

    # --- 第一句：整体状态定调 ---
    r_level = readiness["level"]
    s_level = stress["level"]
    if r_level in ("优秀", "良好") and s_level in ("放松", "平稳"):
        parts.append("综合来看，您当前身心状态良好，身体活力充足且压力水平较低。")
    elif r_level in ("优秀", "良好") and s_level in ("轻度紧张", "明显紧张"):
        parts.append("您当前身体活力不错，但压力水平有所升高，建议适当调节。")
    elif r_level in ("一般", "偏低") and s_level in ("放松", "平稳"):
        parts.append("您当前处于平静状态，但身体调节活力偏低，精力储备一般。")
    else:
        parts.append("您当前身体活力与调节能力偏低，且存在一定压力反应，建议优先休息。")

    # --- 第二句：自主神经 ---
    ans_lv = ans_act["level"]
    bal_lv = ans_bal["level"]
    res_lv = resilience["level"]
    if ans_lv in ("活性旺盛", "活性良好") and bal_lv == "平衡":
        parts.append(f"自主神经系统{ans_lv}且处于平衡状态，调节弹性{res_lv}。")
    elif bal_lv == "交感偏强":
        parts.append(f"自主神经{ans_lv}，但偏向交感激活（应激模式），弹性{res_lv}，建议放松。")
    elif bal_lv == "副交感偏强":
        parts.append(f"自主神经{ans_lv}，副交感偏强（深度放松模式），调节弹性{res_lv}。")
    else:
        parts.append(f"自主神经{ans_lv}、平衡状态，调节弹性{res_lv}。")

    # --- 第三句：运动/活动 ---
    rec_trend = recovery["level"]
    tol_hint = tolerance["level"]
    hrr_level = reserve["level"]
    parts.append(
        f"运动方面，当前处于{hrr_level}，心率趋势{rec_trend}，运动耐受能力{tol_hint}。"
    )

    # --- 第四句：风险筛查 ---
    cv_lv = cv_stress["level"]
    rhy_lv = rhythm["level"]
    met_lv = metabolic["level"]
    risk_items: list[str] = []
    if cv_lv in ("轻度应激", "明显应激"):
        risk_items.append(f"心血管应激{cv_lv}")
    if rhy_lv != "未见异常":
        risk_items.append(f"节律{rhy_lv}")
    if met_lv in ("偏高", "需关注"):
        risk_items.append(f"代谢负荷{met_lv}")
    if risk_items:
        parts.append("风险筛查提示：" + "、".join(risk_items) + "，建议持续关注或复测确认。")
    else:
        parts.append("风险筛查各项均在正常范围内，未见需要特别关注的信号。")

    # --- 第五句：波动敏感性补充 ---
    flu_lv = fluctuation["level"]
    if flu_lv == "高":
        parts.append("此外，您的心率波动敏感性偏高，可能受环境、情绪或测量条件影响，建议在安静环境下复测。")

    parts.append("⚠ 本报告为1分钟即时快照，非临床诊断。HRV指标基于心率推导，仅供健康参考。")

    return "".join(parts)


# =========================================================================
# 核心计算 + 解释
# =========================================================================

def compute_all(hr: np.ndarray, age: int | None = None) -> dict[str, Any]:
    """严格按 Excel 伪代码计算，并为每个指标附带用户可读解释。"""
    hr = np.asarray(hr, dtype=float)
    if hr.size != 60:
        raise ValueError(f"期望 60 个 BPM 样本，实际 {hr.size}")
    if not np.all(np.isfinite(hr)) or np.any(hr <= 0):
        raise ValueError("hr 必须全为有限正数")

    N = 60
    slope = linregress_slope_hr_per_sec(hr)

    avg_hr = float(np.mean(hr))
    sd_hr = std_ddof1(hr)
    hr_min = float(np.min(hr))
    hr_max = float(np.max(hr))
    hr_range = hr_max - hr_min

    # --- 即时压力指数 ---
    trend_score = clip((slope + 0.5) / 1.0 * 50.0, 0.0, 100.0)
    disp_score = clip(sd_hr / 10.0 * 100.0, 0.0, 100.0)
    baseline_dev = 70.0
    dev_score = clip(abs(avg_hr - baseline_dev) / 20.0 * 100.0, 0.0, 100.0)
    stress = clip(0.40 * trend_score + 0.35 * disp_score + 0.25 * dev_score, 0.0, 100.0)

    # --- 心率波动敏感性 ---
    cv_hr = sd_hr / avg_hr if avg_hr else float("nan")
    diff_hr = np.abs(np.diff(hr))
    hrv_diff_mean_bpm = float(np.mean(diff_hr))
    cv_norm = clip(cv_hr / 0.10, 0.0, 1.0)
    diff_norm = clip(hrv_diff_mean_bpm / 5.0, 0.0, 1.0)
    sensitivity_score = 0.5 * cv_norm + 0.5 * diff_norm
    if sensitivity_score < 0.33:
        fluctuation_level = "低"
    elif sensitivity_score < 0.67:
        fluctuation_level = "中"
    else:
        fluctuation_level = "高"

    # --- 即时心率储备利用率 ---
    hr_rest = float(np.percentile(hr, 5))
    if age is not None:
        hr_max_est = 208.0 - 0.7 * float(age)
    else:
        hr_max_est = 185.0
    denom = hr_max_est - hr_rest
    hrr_pct = clip((avg_hr - hr_rest) / denom * 100.0, 0.0, 100.0) if denom > 0 else float("nan")

    # --- 心率恢复趋势 ---
    hr_first = float(np.mean(hr[:20]))
    hr_last = float(np.mean(hr[40:]))
    delta_recovery = hr_last - hr_first
    if delta_recovery < -2 and slope < -0.05:
        recovery_trend = "恢复中"
    elif delta_recovery > 2 and slope > 0.05:
        recovery_trend = "激活中"
    else:
        recovery_trend = "稳定"

    # --- RR 代理 ---
    rr_ms = rr_proxy_ms(hr)
    diff_rr_ms = np.diff(rr_ms)
    rmssd_ms = float(np.sqrt(np.mean(diff_rr_ms**2)))
    sdnn_ms = std_ddof1(rr_ms)
    sd1, sd2, sd1_sd2_ratio = compute_sd1_sd2_poincare(rr_ms)

    # --- 运动耐受提示 ---
    range_norm = clip(hr_range / 30.0, 0.0, 1.0)
    sdnn_norm = clip(sdnn_ms / 80.0, 0.0, 1.0)
    rhr_score = clip(1.0 - abs(avg_hr - 65.0) / 20.0, 0.0, 1.0)
    tolerance_score = (0.35 * range_norm + 0.40 * sdnn_norm + 0.25 * rhr_score) * 100.0
    if tolerance_score >= 60:
        exercise_tolerance_hint = "良好"
    elif tolerance_score >= 35:
        exercise_tolerance_hint = "一般"
    else:
        exercise_tolerance_hint = "需关注"

    # --- 心血管应激指数 ---
    hr_high = clip((avg_hr - 60.0) / (100.0 - 60.0), 0.0, 1.0)
    sd_score = clip(sd_hr / 15.0, 0.0, 1.0)
    range_score = clip(hr_range / 40.0, 0.0, 1.0)
    cv_stress = clip(0.40 * hr_high + 0.35 * sd_score + 0.25 * range_score, 0.0, 1.0) * 100.0

    # --- 代谢负荷提示 ---
    if not math.isfinite(hrr_pct) or hrr_pct >= 20.0:
        metabolic_load_hint = "非静息态，本指标不适用"
    else:
        hr_high_flag = avg_hr > 80.0
        hrv_low_flag = rmssd_ms < 15.0
        if hr_high_flag and hrv_low_flag:
            metabolic_load_hint = "需关注"
        elif hr_high_flag or hrv_low_flag:
            metabolic_load_hint = "偏高"
        else:
            metabolic_load_hint = "正常"

    # --- 节律异常提示 ---
    mu, sigma = avg_hr, sd_hr
    outliers_mask = (hr < mu - 2 * sigma) | (hr > mu + 2 * sigma)
    outliers = int(np.sum(outliers_mask))
    outlier_ratio = outliers / N
    hrv_diff_max_bpm = float(np.max(diff_hr)) if diff_hr.size else float("nan")
    diff_rr_abs = np.abs(np.diff(rr_ms))
    pnn50_proxy = float(np.mean(diff_rr_abs > 50.0)) if diff_rr_abs.size else float("nan")
    flags = 0
    if outlier_ratio > 0.10:
        flags += 1
    if hrv_diff_max_bpm > 15.0:
        flags += 1
    if pnn50_proxy > 0.30:
        flags += 1
    if flags == 0:
        rhythm_hint = "未见异常"
    elif flags == 1:
        rhythm_hint = "轻度波动"
    else:
        rhythm_hint = "建议复查"

    # --- 即时自主神经活性 ---
    sdnn_clamped = clip(sdnn_ms, 1.0, 160.0)
    ans_activity = clip(math.log(sdnn_clamped) / math.log(160.0) * 100.0, 0.0, 100.0)

    # --- 即时神经平衡 ---
    if sd1_sd2_ratio < 0.25:
        ans_balance = "交感偏强"
    elif sd1_sd2_ratio <= 0.55:
        ans_balance = "平衡"
    else:
        ans_balance = "副交感偏强"

    # --- 即时神经调节弹性 ---
    rmssd_norm = clip(rmssd_ms / 50.0, 0.0, 1.0)
    range_norm_res = clip(hr_range / 30.0, 0.0, 1.0)
    resilience = clip(0.60 * rmssd_norm + 0.40 * range_norm_res, 0.0, 1.0) * 100.0

    # --- 放松度 + 即时活力指数 ---
    relax = relaxation_score_excel_upstream(hr, slope, rmssd_ms, sd1_sd2_ratio)
    readiness = clip(0.35 * relax + 0.35 * ans_activity + 0.30 * resilience, 0.0, 100.0)

    # ===================================================================
    # 组装带解释的标准化输出
    # ===================================================================

    stress_level = _level_stress(stress)
    hrr_level = _level_hrr(hrr_pct) if math.isfinite(hrr_pct) else "静息态"
    cv_stress_level = _level_cv_stress(cv_stress)
    ans_act_level = _level_ans_activity(ans_activity)
    readiness_level = _level_readiness(readiness)
    resilience_level = _level_resilience(resilience)

    indicators: dict[str, dict[str, Any]] = {}

    indicators["instant_readiness"] = _make_indicator(
        "即时活力指数", "Instant Readiness", readiness, readiness_level, _READINESS_TEXT,
        unit="分（0~100）",
        detail={
            "relaxation_score": round(relax, 1),
            "ans_activity_score": round(ans_activity, 1),
            "resilience_score": round(resilience, 1),
        },
    )
    indicators["instant_stress_index"] = _make_indicator(
        "即时压力指数", "Instant Stress Index", stress, stress_level, _STRESS_TEXT,
        unit="分（0~100）",
        detail={
            "trend_score": round(trend_score, 1),
            "disp_score": round(disp_score, 1),
            "dev_score": round(dev_score, 1),
        },
    )
    indicators["hr_fluctuation_sensitivity"] = _make_indicator(
        "心率波动敏感性", "HR Fluctuation Sensitivity", None, fluctuation_level, _FLUCTUATION_TEXT,
        detail={"CV_HR": round(cv_hr, 4), "HRV_diff_mean_bpm": round(hrv_diff_mean_bpm, 2)},
    )
    indicators["hr_reserve_utilization"] = _make_indicator(
        "即时心率储备利用率", "HR Reserve Utilization",
        hrr_pct if math.isfinite(hrr_pct) else None,
        hrr_level, _HRR_TEXT,
        unit="%",
        detail={"hr_rest_p5": round(hr_rest, 1), "hr_max_est": round(hr_max_est, 1)},
    )
    indicators["hr_recovery_tendency"] = _make_indicator(
        "心率恢复趋势", "HR Recovery Tendency", None, recovery_trend, _RECOVERY_TEXT,
        detail={
            "mean_first_20s": round(hr_first, 1),
            "mean_last_20s": round(hr_last, 1),
            "delta_bpm": round(delta_recovery, 1),
        },
    )
    indicators["exercise_tolerance_hint"] = _make_indicator(
        "运动耐受提示", "Exercise Tolerance Hint", tolerance_score, exercise_tolerance_hint,
        _TOLERANCE_TEXT,
        unit="分（0~100）",
    )
    indicators["cv_stress_signal"] = _make_indicator(
        "心血管应激指数", "CV Stress Signal", cv_stress, cv_stress_level, _CV_STRESS_TEXT,
        unit="分（0~100）",
    )
    indicators["metabolic_load_hint"] = _make_indicator(
        "代谢负荷提示", "Metabolic Load Hint", None, metabolic_load_hint, _METABOLIC_TEXT,
    )
    indicators["rhythm_anomaly_hint"] = _make_indicator(
        "节律异常提示", "Rhythm Anomaly Hint", None, rhythm_hint, _RHYTHM_TEXT,
        detail={
            "outlier_count": outliers,
            "outlier_ratio": round(outlier_ratio, 4),
            "HRV_diff_max_bpm": round(hrv_diff_max_bpm, 1),
            "PNN50_proxy": round(pnn50_proxy, 4),
        },
    )
    indicators["instant_ans_activity"] = _make_indicator(
        "即时自主神经活性", "Instant ANS Activity", ans_activity, ans_act_level, _ANS_ACTIVITY_TEXT,
        unit="分（0~100）",
    )
    indicators["instant_ans_balance"] = _make_indicator(
        "即时神经平衡趋势", "Instant ANS Balance", None, ans_balance, _ANS_BALANCE_TEXT,
        detail={"SD1_ms": round(sd1, 1), "SD2_ms": round(sd2, 1), "SD1_SD2_ratio": round(sd1_sd2_ratio, 4)},
    )
    indicators["instant_regulatory_resilience"] = _make_indicator(
        "即时神经调节弹性", "Instant Regulatory Resilience", resilience, resilience_level,
        _RESILIENCE_TEXT,
        unit="分（0~100）",
        detail={"RMSSD_ms": round(rmssd_ms, 1), "hr_range_bpm": round(hr_range, 1)},
    )

    summary = _generate_comprehensive_summary(indicators)

    return {
        "meta": {
            "n_samples": N,
            "age_used": age,
            "hr_max_formula": "208-0.7*age" if age is not None else "185(default)",
        },
        "vitals": {
            "mean_hr_bpm": round(avg_hr, 1),
            "std_hr_bpm": round(sd_hr, 2),
            "min_hr_bpm": round(hr_min, 1),
            "max_hr_bpm": round(hr_max, 1),
            "hr_range_bpm": round(hr_range, 1),
            "slope_bpm_per_min": round(slope * 60.0, 2),
        },
        "indicators": indicators,
        "comprehensive_summary": summary,
        "proxy_hrv_ms": {
            "SDNN": round(sdnn_ms, 1),
            "RMSSD": round(rmssd_ms, 1),
        },
        "disclaimer": "本报告为1分钟即时状态快照，非临床诊断。HRV相关指标为心率推导的代理值，非心电级别，仅供健康参考。",
    }


@dataclass
class HrRow:
    elapsed_sec: int
    hr_bpm: float


def read_hr_csv(path: Path) -> list[HrRow]:
    rows: list[HrRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        need = {"elapsed_sec", "hr_bpm"}
        if reader.fieldnames is None or not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} 需要列: {sorted(need)}")
        for r in reader:
            try:
                es = int(float(str(r["elapsed_sec"]).strip()))
                hb = float(str(r["hr_bpm"]).strip())
            except (TypeError, ValueError):
                continue
            rows.append(HrRow(elapsed_sec=es, hr_bpm=hb))
    rows.sort(key=lambda x: x.elapsed_sec)
    return rows


def iter_valid_windows(
    rows: list[HrRow],
    *,
    min_hr: float,
    max_hr: float,
) -> list[tuple[int, np.ndarray]]:
    """返回 (start_elapsed, hr_array60) 连续 60 秒窗口。"""
    if not rows:
        return []
    by_e = {r.elapsed_sec: r.hr_bpm for r in rows}
    min_e = min(by_e)
    max_e = max(by_e)
    out: list[tuple[int, np.ndarray]] = []
    for start in range(min_e, max_e - 59):
        ok = True
        vec = np.zeros(60, dtype=float)
        for k in range(60):
            e = start + k
            if e not in by_e:
                ok = False
                break
            v = by_e[e]
            if not math.isfinite(v) or v < min_hr or v > max_hr:
                ok = False
                break
            vec[k] = v
        if ok:
            out.append((start, vec))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="按 Excel 公式从 1min HR 序列计算 HRV 产品指标")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_output/20260420-rPPG-fiber"),
        help="含 hr_*.csv 的目录",
    )
    ap.add_argument("--n-samples", type=int, default=10, help="随机抽取的 1min 窗口数量")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--age", type=int, default=None, help="用户年龄（用于最大心率）；缺省用 185")
    ap.add_argument("--min-hr", type=float, default=40.0)
    ap.add_argument("--max-hr", type=float, default=180.0)
    ap.add_argument("--out", type=Path, default=None, help="JSON 输出路径")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    hr_files = sorted(data_dir.glob("hr_*.csv"))
    if not hr_files:
        raise SystemExit(f"未找到 hr_*.csv: {data_dir}")

    all_windows: list[dict[str, Any]] = []
    for fp in hr_files:
        rows = read_hr_csv(fp)
        for start, vec in iter_valid_windows(rows, min_hr=args.min_hr, max_hr=args.max_hr):
            all_windows.append({"source": fp.name, "window_start_elapsed_sec": start, "hr": vec})

    if not all_windows:
        raise SystemExit("无满足条件的连续 60 秒有效窗口")

    rng = random.Random(args.seed)
    n = min(args.n_samples, len(all_windows))
    chosen = rng.sample(all_windows, n)

    results: list[dict[str, Any]] = []
    for w in chosen:
        hr = w["hr"]
        metrics = compute_all(hr, age=args.age)
        results.append(
            {
                "source_csv": w["source"],
                "window_start_elapsed_sec": w["window_start_elapsed_sec"],
                **metrics,
            }
        )

    payload = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "n_requested": args.n_samples,
        "n_chosen": n,
        "total_valid_windows": len(all_windows),
        "samples": results,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[OK] 已写入 {out_path}")
    print(text)


if __name__ == "__main__":
    main()
