# Webcam HR（WebSocket 整帧 JPEG）独立部署包

这个目录是**独立可拷贝**的最小 Web 服务：浏览器通过 WebSocket 发送 **JPEG 整帧二进制**，服务端用 OpenCV（headless）解码 + Haar 人脸 ROI +（复制自 `realtime_hr.py` 的）POS/CHROM + FFT 估计心率，并把指标 JSON 回推给页面。

## 目录结构

- `server/hr_engine.py`：算法与缓冲逻辑（从 `realtime_hr.py` 复制整理）
- `server/main.py`：FastAPI + `/ws` WebSocket + 静态页面
- `static/index.html`：H5 页面（`getUserMedia` + canvas 导出 JPEG）
- `cascade/haarcascade_frontalface_default.xml`：从仓库 `dataset/` 复制进来，便于打包部署
- `requirements.txt`：仅服务所需依赖

## 本地运行（开发）

在 `webcam_hr_websocket_app/` 目录：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开：`http://127.0.0.1:8000/`

> 说明：手机浏览器通常要求 **HTTPS** 才能稳定打开摄像头；本机 `localhost`/`127.0.0.1` 例外。

## Docker 一键发布（最小 demo）

在 `webcam_hr_websocket_app/` 目录构建并运行：

```bash
docker build -t webcam-hr-demo:latest .
docker run --rm -p 8000:8000 webcam-hr-demo:latest
```

然后打开：`http://127.0.0.1:8000/`

> 手机访问请走 **HTTPS 域名**（需要你在网关/Nginx/Caddy 上配置证书并反代到容器的 8000）。

说明：`Dockerfile` 使用 **Python 3.11**，因此 `requirements.txt` 里的 `numpy` 版本需要满足 OpenCV 在 py3.11 上的约束（`numpy>=1.23.5`）。如果你本地仍在用 Python 3.8 跑同一份文件，可能需要单独一份 `requirements-py38.txt`（可选）。

## WebSocket 协议（非常简单）

1) 客户端先发送一条 **文本 JSON**（hello）：

```json
{
  "type": "hello",
  "session_id": "any-string",
  "method": "POS",
  "fps_hint": 15,
  "window_seconds": 10,
  "roi_scale": 0.6,
  "log_csv": false
}
```

2) 之后反复发送 **二进制消息**：JPEG bytes（整帧）

3) 服务端回 **文本 JSON**：

- `{"type":"hello_ok",...}`
- `{"type":"metrics",...}`（包含 `hr_bpm`、`face`、`buffer_len` 等）
- `{"type":"error",...}`

可选：当 `log_csv=true` 时，服务端会在 `logs/` 下写 CSV（每秒最多记一条，且 `hr_bpm>0`）。

## 云端 HTTPS（Nginx 反代 + WSS）

要点：

- 浏览器页面是 `https://your-domain/`
- WebSocket 必须是 `wss://your-domain/ws`
- Nginx 需要 `Upgrade` / `Connection` 头

示例（仅示意，按你的证书路径改）：

```nginx
location /ws {
  proxy_http_version 1.1;
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";
  proxy_set_header Host $host;
  proxy_pass http://127.0.0.1:8000;
}

location / {
  proxy_set_header Host $host;
  proxy_pass http://127.0.0.1:8000;
}
```

## 环境变量

- `MAX_JPEG_BYTES`：单帧 JPEG 最大字节数（默认 2097152 = 2MiB）
- `CORS_ALLOW_ORIGINS`：逗号分隔，默认 `*`（生产建议收紧到你的域名）
