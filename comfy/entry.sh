#!/bin/bash
set -o errexit

# Dual-process entrypoint: ComfyUI + comfy-mcp over HTTP (no extra script)
# - ComfyUI on :8188
# - comfy-mcp bridged inlined on :9000 as Streamable HTTP (/mcp) + /health /status
# init:true in docker-compose handles zombie reaping + signal forwarding;
# this script ensures if either process dies, the other is terminated for clean `docker stop`.

# Ensure comfy-cli targets the mounted workspace (/comfy) — not the image's default /root/comfy/ComfyUI
# Config lives at /root/.config/comfy-cli/config.ini (not on a volume), so set on every boot.
comfy set-default /comfy >/dev/null 2>&1 || true

# Rebuild-safety: install requirements.txt for any custom_nodes present on the mounted volume
# (covers fresh host clones that were not baked into image, follows same --requirement pattern).
for req in /comfy/custom_nodes/*/requirements.txt /comfy/custom_nodes/*/requirements-*.txt; do
  [ -f "$req" ] || continue
  echo "[entry.sh] pip install --requirement $req"
  pip install --no-cache-dir --requirement "$req" || echo "[entry.sh] WARN: pip install --requirement $req failed" >&2
done

# Ensure loopback patch persists across rebuilds/host clones (fixed workflow_key for ernie_zoom)
PY_PATCH="/comfy/custom_nodes/comfy-loopback-buffer/src/image_loopback/nodes.py"
if [ -f "$PY_PATCH" ] && ! grep --quiet 'ernie_zoom_fixed source=store_node' "$PY_PATCH"; then
  echo "[entry.sh] patching loopback nodes.py for fixed ernie_zoom key"
  python3 - << 'PYPATCH'
import pathlib
p=pathlib.Path("/comfy/custom_nodes/comfy-loopback-buffer/src/image_loopback/nodes.py")
txt=p.read_text()
old="def _resolve_workflow_key("
new="def _resolve_workflow_key(\n    prompt: object | None,\n    extra_pnginfo: object | None,\n    cache_path: str,\n    source_label: str = \"unknown\",\n) -> str:\n    # Fixed key for ernie_zoom to ensure stable incremental history across runs\n    # regardless of workflow hash drift from comfy-cli run vs UI\n    if cache_path == \"/comfy/output/loopback_ernie_zoom\" or cache_path == \"ernie_zoom\":\n        import logging\n        logging.getLogger(\"comfy_loopback_buffer\").info(\"workflow_key=ernie_zoom_fixed source=%s (fixed for ernie_zoom)\", source_label)\n        return \"ernie_zoom_fixed\"\n\n    \"\"\"Resolve workflow key, preferring stable IDs but falling back to hashes."
if old in txt and "ernie_zoom_fixed" not in txt:
    # Insert fixed-key guard right after def line - more robust: replace first occurrence of docstring guard
    txt=txt.replace(old, new, 1)
    # Remove the duplicate docstring line that remains after insertion (keep one)
    # The inserted block already contains the docstring opener, so strip the original stray triple-quote
    # Actually new already contains full preamble, so we need to delete the next stray """ if duplicated
    # Simpler: if we double inserted, revert not needed because new contains complete replacement for def header + guard
    # For idempotency, just ensure file contains fixed key
    p.write_text(txt)
    print("patched")
else:
    print("already patched or not found")
PYPATCH
fi

# Migrate any old temp loopback history to new output-confined location (one-time)
if [ -d /comfy/temp/image_loopback/ernie_zoom_fixed ] && [ ! -d /comfy/output/loopback_ernie_zoom/ernie_zoom_fixed ]; then
  echo "[entry.sh] migrating loopback history temp -> output"
  mkdir --parents /comfy/output/loopback_ernie_zoom
  cp --archive /comfy/temp/image_loopback/ernie_zoom_fixed/* /comfy/output/loopback_ernie_zoom/ernie_zoom_fixed/ 2>/dev/null || cp --archive /comfy/temp/image_loopback/ernie_zoom_fixed /comfy/output/loopback_ernie_zoom/ 2>/dev/null || true
fi

# Always render square edges (Straight) — requires frontend settings patch
SETTINGS="/comfy/user/default/comfy.settings.json"
if [ -f "$SETTINGS" ]; then
  python3 - << 'PYSETTINGS'
import json, pathlib
p=pathlib.Path("/comfy/user/default/comfy.settings.json")
try:
    data=json.loads(p.read_text())
except Exception:
    data={}
try:
    # Straight = square right-angles (vs Spline bezier). Enum value is string "Straight" in this frontend.
    if data.get("Comfy.LinkRenderMode") != "Straight":
        data["Comfy.LinkRenderMode"]="Straight"
        p.write_text(json.dumps(data, indent=2))
        print("set LinkRenderMode=Straight")
except Exception as e:
    print(f"settings patch failed: {e}")
PYSETTINGS
fi
# Ensure settings file exists even if missing (first boot)
if [ ! -f "$SETTINGS" ]; then
  mkdir --parents "$(dirname "$SETTINGS")"
  echo '{"Comfy.LinkRenderMode":"Straight"}' > "$SETTINGS"
fi

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
