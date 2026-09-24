#!/usr/bin/env bash
# Stream B §137A GPU qualification driver for the longlive2 backend.
#
# Usage: ./scripts/qualify.sh <run-dir>   (e.g. ./scripts/qualify.sh ./output/qual-longlive2)
#
# Stages: idle gate -> benchmark video -> 3-segment run -> validate ->
# JSON summary (via tests/test_qualification.py:summarize_run).
# Crash recovery (kill -9 the video worker mid-segment, then resume) and
# the eyeball visual review stay MANUAL — see reports/video-backends.md.
# NEVER run under contention: the gate aborts when >2 GiB on GPU 0 is held
# by another process (repo GPU-contention rule).
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir="${1:?usage: ./scripts/qualify.sh <run-dir>}"
held_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null \
  | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}')"
if [ "$held_mib" -gt 2048 ]; then
  echo "qualify: GPU busy (${held_mib} MiB held) — backing off, run later" >&2
  exit 3
fi
if [ ! -f "$run_dir/voyage.toml" ]; then
  echo "qualify: no voyage.toml in $run_dir — init first:" >&2
  echo "  ./scripts/run.sh init --output $run_dir --run-id qual-longlive2 \\" >&2
  echo "    --style 'pastel neon line-art, peaceful' --backend longlive2 --force" >&2
  exit 2
fi
./scripts/run.sh benchmark video --run "$run_dir" --warmup 1 --measured 3
./scripts/run.sh run --run "$run_dir" --segments 3
./scripts/run.sh validate --run "$run_dir"
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/app" voyage:latest \
  python -c "import json; from tests.test_qualification import summarize_run; \
print(json.dumps(summarize_run('$run_dir'), indent=2))"
