"""
基于 realtime_hr.py 的分段 CSV 记录：固定总时长内每秒一行心率，按固定间隔轮换保存多个 CSV。

默认：采集 30 分钟（1800s），每 5 分钟（300s）写入一个文件，共 6 个 CSV。

用法:
    python realtime_hr_segmented_record.py
    python realtime_hr_segmented_record.py --duration 1800 --segment-seconds 300 --output-dir ./hr_logs
    python realtime_hr_segmented_record.py --method CHROM --camera 1
"""

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from realtime_hr import calc_hr_fft, chrome_dehaan, pos_wang


def flush_segment_csv(rows, output_dir, session_tag, segment_index):
    if not rows:
        return
    os.makedirs(output_dir, exist_ok=True)
    filename = f"hr_{session_tag}_part{segment_index:02d}.csv"
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["elapsed_sec", "timestamp", "hr_bpm"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] 已保存片段 {segment_index:02d}: {os.path.abspath(path)} ({len(rows)} 行)")


def main():
    parser = argparse.ArgumentParser(
        description="分段保存 CSV 的实时心率采集（基于 realtime_hr.py）"
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号 (默认 0)")
    parser.add_argument(
        "--window", type=float, default=10.0, help="滑动窗口时长/秒 (默认 10)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="POS",
        choices=["POS", "CHROM"],
        help="rPPG 算法",
    )
    parser.add_argument(
        "--roi-scale", type=float, default=0.6, help="人脸 ROI 缩放比例 (默认 0.6)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1800,
        help="总采集时长/秒 (默认 1800 = 30 分钟)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=300,
        help="每个 CSV 覆盖的时长/秒 (默认 300 = 5 分钟)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="CSV 输出目录 (默认当前目录)",
    )
    args = parser.parse_args()

    if args.segment_seconds <= 0:
        print("[ERROR] --segment-seconds 必须为正整数。")
        return
    if args.duration <= 0:
        print("[ERROR] --duration 必须为正整数。")
        return

    cascade_path = "dataset/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

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

    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    segment_rows = []
    segment_index = 0
    last_log_sec = -1
    start_time = time.time()

    n_segments = (args.duration + args.segment_seconds - 1) // args.segment_seconds
    print(f"[INFO] 总时长 {args.duration}s，每 {args.segment_seconds}s 一个文件，约 {n_segments} 个 CSV")
    print(f"[INFO] 会话前缀: hr_{session_tag}_partXX.csv → {os.path.abspath(args.output_dir)}")
    print(f"[INFO] 使用 {args.method} 算法，滑动窗口 {args.window}s")
    print("[INFO] 按 'q' 提前结束；结束后会保存当前未写满的片段。\n")

    bvp = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                print(f"\n[INFO] 已达到采集时长 {args.duration}s，停止采集。")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ts_now = time.time()

            faces = face_cascade.detectMultiScale(
                gray, 1.3, 5, minSize=(80, 80)
            )

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
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(
                        np.float64
                    )
                    roi_buffer.append(roi_rgb)
                    ts_buffer.append(ts_now)

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(
                    display, (rx, ry), (rx2, ry2), (0, 255, 255), 1
                )
            else:
                cv2.putText(
                    display,
                    "No Face Detected",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            min_frames = int(buf_size * 0.7)
            if len(roi_buffer) >= min_frames:
                actual_duration = ts_buffer[-1] - ts_buffer[0]
                if actual_duration > 0:
                    actual_fps = (len(ts_buffer) - 1) / actual_duration
                else:
                    actual_fps = real_fps
                actual_fps = max(actual_fps, 4.0)

                frames_list = list(roi_buffer)

                try:
                    if args.method == "POS":
                        bvp = pos_wang(frames_list, actual_fps)
                    else:
                        bvp = chrome_dehaan(frames_list, actual_fps)

                    hr_fft = calc_hr_fft(bvp, actual_fps)

                    if 40 <= hr_fft <= 180:
                        hr_history.append(hr_fft)
                        hr_display = float(np.median(hr_history))
                except Exception as e:
                    print(f"\n[WARN] 计算异常: {e}")

            cur_sec = int(elapsed)
            if cur_sec != last_log_sec:
                last_log_sec = cur_sec
                hr_val = round(hr_display, 1) if hr_display > 0 else 0.0
                segment_rows.append(
                    {
                        "elapsed_sec": cur_sec,
                        "timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "hr_bpm": hr_val,
                    }
                )
                remaining = args.duration - cur_sec
                print(
                    f"\r  [t={cur_sec:>4d}s] HR={hr_display:5.1f} BPM  |  剩余 {remaining}s   ",
                    end="",
                    flush=True,
                )

                if len(segment_rows) >= args.segment_seconds:
                    flush_segment_csv(
                        segment_rows,
                        args.output_dir,
                        session_tag,
                        segment_index,
                    )
                    segment_rows = []
                    segment_index += 1

            fill_pct = min(len(roi_buffer) / buf_size, 1.0)
            bar_w = 200
            cv2.rectangle(
                display, (20, 10), (20 + bar_w, 30), (60, 60, 60), -1
            )
            cv2.rectangle(
                display,
                (20, 10),
                (20 + int(bar_w * fill_pct), 30),
                (0, 200, 0),
                -1,
            )
            cv2.putText(
                display,
                f"Buffer: {fill_pct * 100:.0f}%",
                (230, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            if hr_display > 0:
                color = (0, 220, 0) if 50 <= hr_display <= 120 else (0, 165, 255)
                cv2.putText(
                    display,
                    f"HR: {hr_display:.0f} BPM",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    3,
                )
            else:
                cv2.putText(
                    display,
                    "Measuring...",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (200, 200, 200),
                    2,
                )

            seg_info = f"Seg: {segment_index + 1}/{n_segments} (buf {len(segment_rows)}/{args.segment_seconds})"
            cv2.putText(
                display,
                f"Method: {args.method} | FPS: {real_fps:.0f} | {int(elapsed)}s / {args.duration}s | {seg_info}",
                (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )

            if (
                bvp is not None
                and len(roi_buffer) >= min_frames
                and hr_display > 0
            ):
                wave_h, wave_w = 80, min(300, display.shape[1] - 40)
                wave_y_start = display.shape[0] - 120
                if wave_y_start > 0 and wave_y_start + wave_h <= display.shape[0]:
                    cv2.rectangle(
                        display,
                        (20, wave_y_start),
                        (20 + wave_w, wave_y_start + wave_h),
                        (40, 40, 40),
                        -1,
                    )
                    tail = bvp[-wave_w:] if len(bvp) >= wave_w else bvp
                    if len(tail) > 1:
                        mn, mx = tail.min(), tail.max()
                        if mx - mn > 1e-10:
                            norm = (tail - mn) / (mx - mn)
                            pts = []
                            x_offset = wave_w - len(norm)
                            for i, v in enumerate(norm):
                                px = 20 + x_offset + i
                                py = wave_y_start + wave_h - 5 - int(
                                    v * (wave_h - 10)
                                )
                                pts.append((px, py))
                            for i in range(len(pts) - 1):
                                cv2.line(
                                    display,
                                    pts[i],
                                    pts[i + 1],
                                    (0, 255, 100),
                                    1,
                                )

            cv2.imshow("rPPG Segmented HR Record", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if segment_rows:
            flush_segment_csv(
                segment_rows, args.output_dir, session_tag, segment_index
            )

        print("\n[INFO] 采集结束。")


if __name__ == "__main__":
    main()
