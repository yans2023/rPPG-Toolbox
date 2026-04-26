from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .hr_engine import HREstimatorWeb


ROOT_DIR = Path(__file__).resolve().parents[1]
CASCADE_PATH = ROOT_DIR / "cascade" / "haarcascade_frontalface_default.xml"
STATIC_DIR = ROOT_DIR / "static"


def _now_ts() -> float:
    return time.time()


@dataclass
class SessionState:
    estimator: HREstimatorWeb
    log_csv: bool = False
    csv_path: Optional[Path] = None
    csv_file: Optional[Any] = None
    csv_writer: Optional[Any] = None
    last_log_sec: int = -1


def _open_csv(session_id: str) -> Tuple[Path, Any, Any]:
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))[:64] or "session"
    csv_path = log_dir / f"hr_{safe}_{int(_now_ts())}.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=["unix_time", "elapsed_sec", "hr_bpm", "face", "buffer_len", "buffer_cap"])
    writer.writeheader()
    return csv_path, f, writer


app = FastAPI(title="webcam-hr-websocket", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_hr(ws: WebSocket) -> None:
    await ws.accept()

    state: Optional[SessionState] = None
    max_bytes = int(os.getenv("MAX_JPEG_BYTES", str(2 * 1024 * 1024)))

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                jpeg = msg["bytes"]
                if len(jpeg) > max_bytes:
                    await ws.send_text(json.dumps({"type": "error", "error": "frame_too_large"}))
                    continue
                if state is None:
                    await ws.send_text(json.dumps({"type": "error", "error": "send_hello_first"}))
                    continue

                out = state.estimator.push_jpeg_bgr(jpeg, _now_ts())
                out["type"] = "metrics"
                await ws.send_text(json.dumps(out))

                if state.log_csv and state.csv_writer is not None:
                    # Best-effort: log once per wall-clock second when HR is available.
                    cur_sec = int(time.time())
                    if cur_sec != state.last_log_sec and float(out.get("hr_bpm") or 0) > 0:
                        state.last_log_sec = cur_sec
                        state.csv_writer.writerow(
                            {
                                "unix_time": time.time(),
                                "elapsed_sec": cur_sec,
                                "hr_bpm": float(out.get("hr_bpm") or 0),
                                "face": bool(out.get("face")),
                                "buffer_len": int(out.get("buffer_len") or 0),
                                "buffer_cap": int(out.get("buffer_cap") or 0),
                            }
                        )
                        state.csv_file.flush()

                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    payload: Dict[str, Any] = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                    continue

                if payload.get("type") != "hello":
                    await ws.send_text(json.dumps({"type": "error", "error": "unknown_message"}))
                    continue

                method = str(payload.get("method", "POS")).upper()
                if method not in ("POS", "CHROM"):
                    await ws.send_text(json.dumps({"type": "error", "error": "invalid_method"}))
                    continue

                fps_hint = float(payload.get("fps_hint", 30))
                window_seconds = float(payload.get("window_seconds", 10))
                roi_scale = float(payload.get("roi_scale", 0.6))
                log_csv = bool(payload.get("log_csv", False))
                session_id = str(payload.get("session_id", "default"))

                estimator = HREstimatorWeb(
                    method=method,  # type: ignore[arg-type]
                    window_seconds=window_seconds,
                    fps_hint=fps_hint,
                    roi_scale=roi_scale,
                    cascade_xml_path=str(CASCADE_PATH),
                )

                csv_path = None
                csv_file = None
                csv_writer = None
                if log_csv:
                    csv_path, csv_file, csv_writer = _open_csv(session_id)

                state = SessionState(
                    estimator=estimator,
                    log_csv=log_csv,
                    csv_path=csv_path,
                    csv_file=csv_file,
                    csv_writer=csv_writer,
                    last_log_sec=-1,
                )

                await ws.send_text(
                    json.dumps(
                        {
                            "type": "hello_ok",
                            "method": method,
                            "window_seconds": window_seconds,
                            "fps_hint": fps_hint,
                            "roi_scale": roi_scale,
                            "buffer_cap": estimator.buf_size,
                            "cascade_path": str(CASCADE_PATH),
                            "log_csv": log_csv,
                            "csv_path": str(csv_path) if csv_path else "",
                        }
                    )
                )
                continue

    except WebSocketDisconnect:
        pass
    finally:
        if state and state.csv_file:
            try:
                state.csv_file.close()
            except Exception:
                pass


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
