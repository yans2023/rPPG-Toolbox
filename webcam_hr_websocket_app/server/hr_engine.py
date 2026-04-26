"""
Heart-rate estimation core copied from `realtime_hr.py` (POS / CHROM + FFT HR).

This file is intentionally self-contained for deployment packaging.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy import sparse


def detrend(input_signal: np.ndarray, lambda_value: float) -> np.ndarray:
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lambda_value**2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def bandpass_filter(sig: np.ndarray, fs: float, low: float = 0.75, high: float = 2.5, order: int = 1) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(low / nyq, 0.02)
    hi = min(high / nyq, 0.98)
    if lo >= hi:
        return sig
    b, a = scipy_signal.butter(order, [lo, hi], btype="bandpass")
    return scipy_signal.filtfilt(b, a, sig.astype(np.double))


def pos_wang(frames: List[np.ndarray], fs: float) -> np.ndarray:
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


def chrome_dehaan(frames: List[np.ndarray], fs: float) -> np.ndarray:
    RGB = np.array([np.mean(f, axis=(0, 1)) for f in frames])
    FN = RGB.shape[0]
    nyq = fs / 2.0
    lo = max(0.7 / nyq, 0.02)
    hi = min(2.5 / nyq, 0.98)
    if lo >= hi:
        return np.zeros(FN)
    B, A = scipy_signal.butter(3, [lo, hi], "bandpass")
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
        S[WinS:WinM] += SWin[: WinL // 2]
        S[WinM:WinE] = SWin[WinL // 2 :]
        WinS = WinM

    return S[:FN]


def calc_hr_fft(bvp: np.ndarray, fs: float, low: float = 0.75, high: float = 2.5) -> float:
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
    return float(peak_freq * 60.0)


MethodName = Literal["POS", "CHROM"]


@dataclass
class FaceTracker:
    face_cascade: cv2.CascadeClassifier
    roi_scale: float
    last_face: Optional[Tuple[int, int, int, int]] = None
    no_face_count: int = 0

    def update(self, frame_bgr: np.ndarray, fps_hint: float) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            self.last_face = (int(x), int(y), int(w), int(h))
            self.no_face_count = 0
        else:
            self.no_face_count += 1
            if self.last_face is not None and self.no_face_count > int(max(fps_hint, 1.0) * 2):
                self.last_face = None

        if self.last_face is None:
            return None

        x, y, w, h = self.last_face
        s = float(self.roi_scale)
        cx, cy = x + w // 2, y + h // 2
        rw, rh = int(w * s), int(h * s)
        rx = max(0, cx - rw // 2)
        ry = max(0, cy - rh // 2 - int(h * 0.05))
        rx2 = min(frame_bgr.shape[1], rx + rw)
        ry2 = min(frame_bgr.shape[0], ry + rh)

        roi = frame_bgr[ry:ry2, rx:rx2]
        if roi.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float64)
        return roi_rgb


class HREstimatorWeb:
    def __init__(
        self,
        *,
        method: MethodName,
        window_seconds: float,
        fps_hint: float,
        roi_scale: float,
        cascade_xml_path: str,
    ) -> None:
        self.method: MethodName = method
        self.roi_scale = float(roi_scale)

        self.face_cascade = cv2.CascadeClassifier(cascade_xml_path)
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.tracker = FaceTracker(face_cascade=self.face_cascade, roi_scale=self.roi_scale)

        fps_hint = float(fps_hint)
        if fps_hint <= 0 or fps_hint > 120:
            fps_hint = 30.0
        self.fps_hint = fps_hint

        buf_size = int(window_seconds * fps_hint)
        buf_size = max(buf_size, 30)
        self.buf_size = int(buf_size)

        self.roi_buffer: Deque[np.ndarray] = deque(maxlen=self.buf_size)
        self.ts_buffer: Deque[float] = deque(maxlen=self.buf_size)

        self.hr_history: Deque[float] = deque(maxlen=10)
        self.hr_display: float = 0.0
        self.last_bvp: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.roi_buffer.clear()
        self.ts_buffer.clear()
        self.hr_history.clear()
        self.hr_display = 0.0
        self.last_bvp = None
        self.tracker = FaceTracker(face_cascade=self.face_cascade, roi_scale=self.roi_scale)

    def push_jpeg_bgr(self, jpeg_bytes: bytes, ts: float) -> dict:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "invalid_jpeg"}

        roi_rgb = self.tracker.update(frame, self.fps_hint)
        if roi_rgb is None:
            return {
                "ok": True,
                "face": False,
                "buffer_len": len(self.roi_buffer),
                "buffer_cap": self.buf_size,
                "hr_bpm": self.hr_display,
            }

        self.roi_buffer.append(roi_rgb)
        self.ts_buffer.append(float(ts))

        min_frames = int(self.buf_size * 0.7)
        if len(self.roi_buffer) < min_frames:
            return {
                "ok": True,
                "face": True,
                "buffer_len": len(self.roi_buffer),
                "buffer_cap": self.buf_size,
                "buffer_fill": len(self.roi_buffer) / float(self.buf_size),
                "hr_bpm": self.hr_display,
                "ready": False,
            }

        actual_duration = self.ts_buffer[-1] - self.ts_buffer[0]
        if actual_duration > 0:
            actual_fps = (len(self.ts_buffer) - 1) / actual_duration
        else:
            actual_fps = self.fps_hint
        actual_fps = max(float(actual_fps), 4.0)

        frames_list = list(self.roi_buffer)
        try:
            if self.method == "POS":
                bvp = pos_wang(frames_list, actual_fps)
            else:
                bvp = chrome_dehaan(frames_list, actual_fps)

            hr_fft = calc_hr_fft(bvp, actual_fps)
            self.last_bvp = bvp

            if 40 <= hr_fft <= 180:
                self.hr_history.append(hr_fft)
                self.hr_display = float(np.median(self.hr_history))
        except Exception as e:  # noqa: BLE001 - keep websocket alive
            return {"ok": False, "error": f"hr_compute_failed: {e}"}

        return {
            "ok": True,
            "face": True,
            "buffer_len": len(self.roi_buffer),
            "buffer_cap": self.buf_size,
            "buffer_fill": len(self.roi_buffer) / float(self.buf_size),
            "hr_bpm": self.hr_display,
            "hr_instant": hr_fft,
            "fps_effective": actual_fps,
            "ready": True,
        }
