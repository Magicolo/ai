#!/usr/bin/env bash
# Build the CUDA video worker image (long pole: torch + LongLive deps +
# flash-attn). Run in background and poll /tmp/videobuild.log.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f worker/Dockerfile.video -t voyage-video:latest .
