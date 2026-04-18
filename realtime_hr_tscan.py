"""
实时摄像头心率检测 — 基于 TS-CAN 深度学习方法 (支持 POS/CHROM 对比)
使用预训练 TS-CAN 模型从摄像头视频中提取 BVP 信号并计算心率。

用法:
    # TS-CAN 推理 (需要预训练模型)
    python realtime_hr_tscan.py --model ./final_model_release/PURE_TSCAN.pth
    python realtime_hr_tscan.py --model ./my_model.pth --duration 300 --csv tscan_hr.csv

    # 也支持 POS/CHROM 对比 (无需模型)
    python realtime_hr_tscan.py --method POS
    python realtime_hr_tscan.py --method CHROM
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy import sparse

import torch
from neural_methods.model.TS_CAN import TSCAN


# ───────────────── 信号处理工具 ─────────────────

def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                        (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
        input_signal)
    return filtered_signal


def bandpass_filter(sig, fs, low=0.75, high=2.5, order=1):
    """安全的带通滤波：自动将截止频率限制在 Nyquist 范围内"""
    nyq = fs / 2.0
    lo = max(low / nyq, 0.02)
    hi = min(high / nyq, 0.98)
    if lo >= hi:
        return sig
    b, a = scipy_signal.butter(order, [lo, hi], btype='bandpass')
    return scipy_signal.filtfilt(b, a, sig.astype(np.double))


# ───────────────── rPPG 无监督算法 (对比用) ─────────────────

def pos_wang(frames, fs):
    """POS 算法 (Wang et al., 2017)"""
    RGB = np.array([np.mean(f, axis=(0, 1)) for f in frames])
    N = RGB.shape[0]
    H = np.zeros(N)
    win_len = math.ceil(1.6 * fs)
    if win_len < 2:
        win_len = 2
    for n in range(win_len, N):
        m = n - win_len
        Cn = RGB[m:n, :] / (np.mean(RGB[m:n, :], axis=0) + 1e-10)
        S = np.dot(np.array([[0, 1, -1], [-2, 1, 1]]), Cn.T)
        h = S[0, :] + (np.std(S[0, :]) / (np.std(S[1, :]) + 1e-10)) * S[1, :]
        h -= np.mean(h)
        H[m:n] += h
    BVP = detrend(H, 100)
    BVP = bandpass_filter(BVP, fs, low=0.75, high=2.5)
    return BVP


def chrome_dehaan(frames, fs):
    """CHROM 算法 (De Haan & Jeanne, 2013)"""
    RGB = np.array([np.mean(f, axis=(0, 1)) for f in frames])
    FN = RGB.shape[0]
    nyq = fs / 2.0
    lo = max(0.7 / nyq, 0.02)
    hi = min(2.5 / nyq, 0.98)
    if lo >= hi:
        return np.zeros(FN)
    B, A = scipy_signal.butter(3, [lo, hi], 'bandpass')
    WinL = math.ceil(1.6 * fs)
    if WinL < 4:
        WinL = 4
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    totallen = (WinL // 2) * (NWin + 1)
    S = np.zeros(max(totallen, FN))
    WinS = 0
    for _ in range(NWin):
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
        if WinE > FN:
            break
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = RGB[WinS:WinE] / (RGBBase + 1e-10)
        Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
        Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]
        Xf = scipy_signal.filtfilt(B, A, Xs, axis=0)
        Yf = scipy_signal.filtfilt(B, A, Ys)
        Alpha = np.std(Xf) / (np.std(Yf) + 1e-10)
        SWin = Xf - Alpha * Yf
        SWin *= scipy_signal.windows.hann(WinL)
        S[WinS:WinM] += SWin[:WinL // 2]
        S[WinM:WinE] = SWin[WinL // 2:]
        WinS = WinM
    return S[:FN]


# ───────────────── TS-CAN 预处理 ─────────────────

def diff_normalize_data(data):
    """与 Toolbox BaseLoader.diff_normalize_data 保持一致
    输入: [T, H, W, 3] float  输出: [T, H, W, 3] float32
    """
    n, h, w, c = data.shape
    diff_len = n - 1
    diff = np.zeros((diff_len, h, w, c), dtype=np.float32)
    for j in range(diff_len):
        diff[j] = (data[j + 1] - data[j]) / (data[j + 1] + data[j] + 1e-7)
    std = np.std(diff)
    if std > 1e-10:
        diff = diff / std
    padding = np.zeros((1, h, w, c), dtype=np.float32)
    diff = np.append(diff, padding, axis=0)
    diff[np.isnan(diff)] = 0
    return diff


def standardize_data(data):
    """与 Toolbox BaseLoader.standardized_data 保持一致"""
    data = data.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    if std > 1e-10:
        data = (data - mean) / std
    else:
        data = data - mean
    data[np.isnan(data)] = 0
    return data


def preprocess_frames_for_tscan(frames_rgb, img_size):
    """将 RGB 帧列表转为 TSCAN 所需的 6 通道 Tensor
    Args:
        frames_rgb: list of [H, W, 3] float64 arrays (RGB)
        img_size: 模型输入尺寸 (如 72)
    Returns:
        tensor: [T, 6, img_size, img_size] float32 tensor
    """
    resized = np.array([
        cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)
        for f in frames_rgb
    ], dtype=np.float64)  # [T, H, W, 3]

    diff_norm = diff_normalize_data(resized)   # [T, H, W, 3]
    std_data = standardize_data(resized)       # [T, H, W, 3]

    # 拼接为 6 通道 [T, H, W, 6]
    combined = np.concatenate([diff_norm, std_data], axis=-1)

    # NHWC → NCHW
    combined = combined.transpose(0, 3, 1, 2)  # [T, 6, H, W]
    return torch.from_numpy(combined.astype(np.float32))


# ───────────────── 模型加载 ─────────────────

def load_tscan_model(model_path, frame_depth, img_size, device):
    """加载 TS-CAN 预训练模型，自动处理 DataParallel 前缀"""
    model = TSCAN(frame_depth=frame_depth, img_size=img_size)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # 训练时用了 DataParallel，权重 key 有 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


# ───────────────── 心率计算 ─────────────────

def calc_hr_fft(bvp, fs, low=0.75, high=2.5):
    N = len(bvp)
    if N < 4:
        return 0.0
    nfft = 2 ** (N - 1).bit_length()
    freqs, psd = scipy_signal.periodogram(bvp, fs=fs, nfft=nfft, detrend=False)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    valid_freqs = freqs[mask]
    valid_psd = psd[mask]
    peak_freq = valid_freqs[np.argmax(valid_psd)]
    return peak_freq * 60.0


# ───────────────── 主循环 ─────────────────

def main():
    parser = argparse.ArgumentParser(
        description='实时摄像头心率检测 — TS-CAN 深度学习推理 (兼容 POS/CHROM)')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头编号 (默认 0)')
    parser.add_argument('--window', type=float, default=10.0,
                        help='滑动窗口时长/秒 (默认 10)')
    parser.add_argument('--method', type=str, default='TSCAN',
                        choices=['TSCAN', 'POS', 'CHROM'],
                        help='rPPG 算法 (默认 TSCAN)')
    parser.add_argument('--model', type=str,
                        default='./final_model_release/PURE_TSCAN.pth',
                        help='TS-CAN 预训练模型路径')
    parser.add_argument('--frame-depth', type=int, default=10,
                        help='TS-CAN 时间移位窗口深度 (默认 10, 需与训练一致)')
    parser.add_argument('--img-size', type=int, default=72,
                        choices=[36, 72, 96, 128],
                        help='TS-CAN 输入图像尺寸 (默认 72, 需与训练一致)')
    parser.add_argument('--device', type=str, default='auto',
                        help='推理设备: cpu / cuda / mps / auto (默认 auto)')
    parser.add_argument('--roi-scale', type=float, default=0.6,
                        help='人脸 ROI 缩放比例 (默认 0.6)')
    parser.add_argument('--duration', type=int, default=0,
                        help='采集时长/秒 (默认 0 = 不限时)')
    parser.add_argument('--csv', type=str, default='',
                        help='CSV 输出路径 (默认自动生成文件名)')
    args = parser.parse_args()

    # ── 设备选择 ──
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    use_tscan = (args.method == 'TSCAN')
    model = None

    # ── 加载 TS-CAN 模型 ──
    if use_tscan:
        if not os.path.exists(args.model):
            print(f"[ERROR] 找不到模型文件: {args.model}")
            print()
            print("请通过以下方式获取预训练模型：")
            print("  1. 从 rPPG-Toolbox GitHub Release 下载预训练权重")
            print("  2. 或使用 Toolbox 训练自己的模型:")
            print("     python main.py --config_file ./configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml")
            print()
            print("然后运行: python realtime_hr_tscan.py --model <你的模型路径>")
            print()
            print("如果暂时没有模型，可以先用无监督方法测试:")
            print("  python realtime_hr_tscan.py --method POS")
            print("  python realtime_hr_tscan.py --method CHROM")
            sys.exit(1)

        print(f"[INFO] 加载 TS-CAN 模型: {args.model}")
        print(f"[INFO] 设备: {device} | frame_depth={args.frame_depth} | img_size={args.img_size}")
        model = load_tscan_model(args.model, args.frame_depth, args.img_size, device)
        print("[INFO] 模型加载成功!")

    # ── Haar Cascade 人脸检测器 ──
    cascade_path = 'dataset/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查摄像头编号或权限设置。")
        return

    real_fps = cap.get(cv2.CAP_PROP_FPS)
    if real_fps <= 0 or real_fps > 120:
        real_fps = 30.0
    print(f"[INFO] 摄像头 FPS: {real_fps:.1f}")

    buf_size = int(args.window * real_fps)
    buf_size = max(buf_size, 30)
    roi_buffer = deque(maxlen=buf_size)
    ts_buffer = deque(maxlen=buf_size)

    hr_display = 0.0
    hr_history = deque(maxlen=10)
    last_face = None
    no_face_count = 0

    # ── 每秒心率记录 ──
    hr_log = []
    last_log_sec = -1
    start_time = time.time()

    timed = args.duration > 0
    if timed:
        csv_path = args.csv or f"tscan_hr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"[INFO] 定时采集模式：{args.duration} 秒，每秒记录心率")
        print(f"[INFO] 数据将保存到: {csv_path}")

    print(f"[INFO] 使用 {args.method} 算法，滑动窗口 {args.window}s")
    print("[INFO] 按 'q' 退出。请保持面部正对摄像头，光线充足，尽量不要移动。\n")

    bvp = None
    last_infer_time = 0
    infer_interval = 0.3  # TSCAN 推理间隔 (秒)，避免每帧都跑模型

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time

        if timed and elapsed >= args.duration:
            print(f"\n[INFO] 已达到采集时长 {args.duration}s，停止采集。")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ts_now = time.time()

        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            last_face = (x, y, w, h)
            no_face_count = 0
        else:
            no_face_count += 1
            if no_face_count > int(real_fps * 2):
                last_face = None

        if last_face is not None:
            x, y, w, h = last_face
            s = args.roi_scale
            cx, cy = x + w // 2, y + h // 2
            rw, rh = int(w * s), int(h * s)
            rx = max(0, cx - rw // 2)
            ry = max(0, cy - rh // 2 - int(h * 0.05))
            rx2 = min(frame.shape[1], rx + rw)
            ry2 = min(frame.shape[0], ry + rh)

            roi = frame[ry:ry2, rx:rx2]
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float64)
                roi_buffer.append(roi_rgb)
                ts_buffer.append(ts_now)

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(display, (rx, ry), (rx2, ry2), (0, 255, 255), 1)
        else:
            cv2.putText(display, "No Face Detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── 心率计算 ──
        min_frames = int(buf_size * 0.7)
        now = time.time()
        should_compute = (len(roi_buffer) >= min_frames
                          and now - last_infer_time >= infer_interval)

        if should_compute:
            last_infer_time = now
            actual_duration = ts_buffer[-1] - ts_buffer[0]
            if actual_duration > 0:
                actual_fps = (len(ts_buffer) - 1) / actual_duration
            else:
                actual_fps = real_fps
            actual_fps = max(actual_fps, 4.0)

            frames_list = list(roi_buffer)

            try:
                if use_tscan:
                    bvp = tscan_inference(
                        model, frames_list, args.frame_depth,
                        args.img_size, device, actual_fps)
                elif args.method == 'POS':
                    bvp = pos_wang(frames_list, actual_fps)
                else:
                    bvp = chrome_dehaan(frames_list, actual_fps)

                hr_fft = calc_hr_fft(bvp, actual_fps)

                if 40 <= hr_fft <= 180:
                    hr_history.append(hr_fft)
                    hr_display = np.median(hr_history)
            except Exception as e:
                print(f"[WARN] 计算异常: {e}")

        # ── 每秒记录心率 ──
        cur_sec = int(elapsed)
        if timed and cur_sec != last_log_sec and hr_display > 0:
            last_log_sec = cur_sec
            hr_log.append({
                'elapsed_sec': cur_sec,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hr_bpm': round(hr_display, 1),
            })
            remaining = args.duration - cur_sec
            print(f"\r  [t={cur_sec:>4d}s] HR={hr_display:5.1f} BPM  |  剩余 {remaining}s   ",
                  end='', flush=True)

        # ── 绘制 UI ──
        fill_pct = min(len(roi_buffer) / buf_size, 1.0)
        bar_w = 200
        cv2.rectangle(display, (20, 10), (20 + bar_w, 30), (60, 60, 60), -1)
        cv2.rectangle(display, (20, 10), (20 + int(bar_w * fill_pct), 30), (0, 200, 0), -1)
        cv2.putText(display, f"Buffer: {fill_pct * 100:.0f}%", (230, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if hr_display > 0:
            color = (0, 220, 0) if 50 <= hr_display <= 120 else (0, 165, 255)
            cv2.putText(display, f"HR: {hr_display:.0f} BPM", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(display, "Measuring...", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

        method_label = args.method
        if use_tscan:
            method_label += f" ({device})"
        info_text = f"Method: {method_label} | FPS: {real_fps:.0f}"
        if timed:
            info_text += f" | {int(elapsed)}s / {args.duration}s"
        cv2.putText(display, info_text,
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # ── 绘制 BVP 波形 ──
        if bvp is not None and len(roi_buffer) >= min_frames and hr_display > 0:
            wave_h, wave_w = 80, min(300, display.shape[1] - 40)
            wave_y_start = display.shape[0] - 120
            if wave_y_start > 0 and wave_y_start + wave_h <= display.shape[0]:
                cv2.rectangle(display, (20, wave_y_start),
                              (20 + wave_w, wave_y_start + wave_h),
                              (40, 40, 40), -1)
                tail = bvp[-wave_w:] if len(bvp) >= wave_w else bvp
                if len(tail) > 1:
                    mn, mx = tail.min(), tail.max()
                    if mx - mn > 1e-10:
                        norm = (tail - mn) / (mx - mn)
                        pts = []
                        x_offset = wave_w - len(norm)
                        for i, v in enumerate(norm):
                            px = 20 + x_offset + i
                            py = (wave_y_start + wave_h - 5
                                  - int(v * (wave_h - 10)))
                            pts.append((px, py))
                        for i in range(len(pts) - 1):
                            cv2.line(display, pts[i], pts[i + 1],
                                     (0, 255, 100), 1)

        cv2.imshow('rPPG Heart Rate (TS-CAN)', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── 保存 CSV ──
    if timed and hr_log:
        csv_path = args.csv or f"tscan_hr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['elapsed_sec', 'timestamp', 'hr_bpm'])
            writer.writeheader()
            writer.writerows(hr_log)

        hrs = [r['hr_bpm'] for r in hr_log]
        print(f"\n\n{'='*50}")
        print(f"  采集完成！共记录 {len(hr_log)} 条心率数据")
        print(f"  方法: {args.method}")
        print(f"  时长: {hr_log[-1]['elapsed_sec']}s")
        print(f"  心率范围: {min(hrs):.1f} ~ {max(hrs):.1f} BPM")
        print(f"  平均心率: {np.mean(hrs):.1f} BPM")
        print(f"  数据已保存: {os.path.abspath(csv_path)}")
        print(f"{'='*50}")
    else:
        print("\n[INFO] 已退出。")


def tscan_inference(model, frames_list, frame_depth, img_size, device, fps):
    """用 TS-CAN 模型推理 BVP 信号
    Args:
        model: 已加载的 TSCAN 模型
        frames_list: ROI 帧列表 [H_i, W_i, 3] float64
        frame_depth: TSM 时间窗口深度
        img_size: 模型输入尺寸
        device: torch device
        fps: 实际帧率
    Returns:
        bvp: numpy array, 预测的 BVP 信号
    """
    input_tensor = preprocess_frames_for_tscan(frames_list, img_size)
    T = input_tensor.shape[0]

    # TSM 要求帧数是 frame_depth 的整数倍
    usable = (T // frame_depth) * frame_depth
    if usable < frame_depth:
        return np.zeros(T)
    input_tensor = input_tensor[:usable]

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        pred = model(input_tensor)            # [usable, 1]
        bvp = pred.cpu().numpy().flatten()    # [usable]

    # 补齐被截断的尾部帧
    if usable < T:
        bvp = np.pad(bvp, (0, T - usable), mode='edge')

    bvp = bandpass_filter(bvp, fps, low=0.75, high=2.5)
    return bvp


if __name__ == '__main__':
    main()
