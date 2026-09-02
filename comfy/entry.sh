#!/bin/bash
set -e

# Dual-process entrypoint: ComfyUI + comfy-mcp over HTTP (no extra script)
# - ComfyUI on :8188
# - comfy-mcp bridged inlined on :9000 as Streamable HTTP (/mcp) + /health /status
# init:true in docker-compose handles zombie reaping + signal forwarding;
# this script ensures if either process dies, the other is terminated for clean `docker stop`.

# Ensure comfy-cli targets the mounted workspace (/comfy) — not the image's default /root/comfy/ComfyUI
# Config lives at /root/.config/comfy-cli/config.ini (not on a volume), so set on every boot.
comfy set-default /comfy >/dev/null 2>&1 || true

python /comfy/main.py --listen 0.0.0.0 --port 8188 &
PID_COMFY=$!

python - << 'PY' &
from comfy_mcp.server import mcp
from starlette.requests import Request
from starlette.responses import JSONResponse

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request):
    return JSONResponse({"status": "ok", "service": "comfy-mcp"})

@mcp.custom_route("/status", methods=["GET"])
async def status(request: Request):
    return JSONResponse({"status": "ok", "service": "comfy-mcp"})

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=9000)
PY
PID_PROXY=$!

# Wait for either to exit, then terminate the other
wait -n
EXIT_CODE=$?

kill $PID_COMFY $PID_PROXY 2>/dev/null || true
wait || true

exit $EXIT_CODE
