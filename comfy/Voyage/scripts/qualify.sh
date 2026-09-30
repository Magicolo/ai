#!/usr/bin/env bash
# Stream B §137A GPU qualification driver for the longlive2 backend.
#
# Usage: ./scripts/qualify.sh <run-dir>   (e.g. ./scripts/qualify.sh /tmp/qual-longlive2)
#   <run-dir> MUST be absolute: workers spawn with CWD=run_dir, so a
#   relative dir doubles up inside payload paths (issue 064 leg b).
#
# Stages: nvidia-smi presence -> idle gate -> absolute-path gate ->
# disk preflight -> benchmark video -> 3-segment run -> validate ->
# JSON summary teed to reports/ (issue 064 legs c/d, 060 artifacts).
# Crash recovery (kill -9 the video worker mid-segment, then resume) and
# the eyeball visual review stay MANUAL — see reports/video-backends.md.
# NEVER run under contention: the gate aborts when >2 GiB on GPU 0 is held
# by another process (repo GPU-contention rule).
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir="${1:?usage: ./scripts/qualify.sh <absolute-run-dir>}"

# Fail closed with a message when there is no GPU stack at all (issue 064
# leg a): without this, the bare pipeline below aborts silently under
# `pipefail`+`set -e` (grep exits 1 on empty input) with no explanation.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "qualify: nvidia-smi not found — GPU qualification needs a CUDA host" >&2
  exit 4
fi
held_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null \
  | grep -oE '[0-9]+' | awk '{s+=$1} END {print s+0}')"
if [ "$held_mib" -gt 2048 ]; then
  echo "qualify: GPU busy (${held_mib} MiB held) — backing off, run later" >&2
  exit 3
fi
# Absolute-path enforcement (issue 064 leg b): the codebase invariant is
# absolute voyage paths (cli.resolve_run_dir); a relative dir reproduces
# the doubling stall documented twice in reports/video-backends.md.
case "$run_dir" in
  /*) ;;
  *)
    echo "qualify: <run-dir> must be absolute (got '$run_dir')" >&2
    exit 2
    ;;
esac
# Disk preflight (issue 064 leg d): a 3-segment GPU run + finalize needs
# GiBs; discover a full disk here, not mid-run. Floor mirrors the dev
# default reserve (config.DEV_MIN_FREE_SPACE_GIB), overridable via env.
min_free_gib="${QUALIFY_MIN_FREE_GIB:-5}"
avail_kib="$(df -k --output=avail "$run_dir" 2>/dev/null | tail -n 1 | tr -d ' ' || true)"
if [ -n "$avail_kib" ] && [ "$avail_kib" -lt $((min_free_gib * 1024 * 1024)) ]; then
  avail_gib="$((avail_kib / 1024 / 1024))"
  echo "qualify: only ~${avail_gib} GiB free under $run_dir (need ${min_free_gib})" >&2
  exit 5
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
# Artifact persistence (issue 064 leg c, 060): the summary used to be
# stdout-only, so every qualification evaporated. Tee to reports/ and
# print the path; the filename carries backend run id + date.
# Run dir travels via the environment (never shell-interpolated into the
# python snippet): paths with spaces/quotes would otherwise break the
# quoting or inject code (single quotes inside double quotes do not expand).
artifact="reports/qual-$(basename "$run_dir")-$(date +%F).json"
docker run --rm --user="$(id -u):$(id -g)" -e PYTHONDONTWRITEBYTECODE=1 -w /app -v "$PWD:/app" -e RUN_DIR="$run_dir" voyage:latest \
  python -c 'import json, os; from tests.test_qualification import summarize_run; \
print(json.dumps(summarize_run(os.environ["RUN_DIR"]), indent=2))' \
  | tee "$artifact"
echo "qualify: summary saved to $artifact" >&2
