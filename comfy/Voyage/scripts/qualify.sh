#!/usr/bin/env bash
# GPU qualification driver (Stream B §137A; issue 090 generalized).
#
# Usage: ./scripts/qualify.sh [--backend ltxv|causvid|ltx25|ltx23] [--segments N] <run-dir>
#   e.g. ./scripts/qualify.sh /tmp/qual-ltxv
#        ./scripts/qualify.sh --backend causvid --segments 2 /tmp/qual-causvid
#   <run-dir> MUST be absolute AND equal to $PWD/output/<basename>:
#   the two-verb CLI addresses runs by NAME (`output/<name>` under the
#   repo root — workers spawn with CWD=run_dir, so a relative dir
#   doubles up inside payload paths (issue 064 leg b)). Default backend
#   is ltx25 (the config default since 2026-10-02).
#   The helper is
#   backend-agnostic — it configures, generates, and tees the JSON
#   summary for whatever backend the run dir was generated with.
#
# Stages: nvidia-smi presence -> idle gate -> absolute-path gate ->
# disk preflight -> configure -> N-segment generate (validates +
# finalizes inline) -> JSON summary teed to reports/ (issue 064 legs
# c/d, 060 artifacts).
# Crash recovery (kill -9 the video worker mid-segment, then resume) and
# the eyeball visual review stay MANUAL — see reports/video-backends.md.
# NEVER run under contention: the gate aborts when >2 GiB on GPU 0 is held
# by another process (repo GPU-contention rule).
set -euo pipefail
# NOTE: no external commands before the nvidia-smi gate below — the
# fail-closed test runs this script with an empty PATH, so SCRIPT_DIR
# resolves via builtins only (parameter expansion + cd + pwd).
SCRIPT_DIR="$(cd "${0%/*}" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
cd "$SCRIPT_DIR/.."

backend="ltx25"
segments="3"
run_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --backend)
      backend="${2:?--backend needs a value (ltxv|causvid|ltx25|ltx23)}"
      shift 2
      ;;
    --backend=*)
      backend="${1#--backend=}"
      shift
      ;;
    --segments)
      segments="${2:?--segments needs a value}"
      shift 2
      ;;
    --segments=*)
      segments="${1#--segments=}"
      shift
      ;;
    -h|--help)
      echo "usage: ./scripts/qualify.sh [--backend ltxv|causvid|ltx25|ltx23] [--segments N] <absolute-run-dir>" >&2
      exit 0
      ;;
    *)
      if [ -n "$run_dir" ]; then
        echo "qualify: unexpected extra arg '$1'" >&2
        exit 2
      fi
      run_dir="$1"
      shift
      ;;
  esac
done
if [ -z "$run_dir" ]; then
  echo "usage: ./scripts/qualify.sh [--backend ltxv|causvid|ltx25|ltx23] [--segments N] <absolute-run-dir>" >&2
  exit 2
fi
case "$backend" in
  ltxv|causvid|ltx25|ltx23) ;;
  *)
     echo "qualify: --backend must be ltxv|causvid|ltx25|ltx23 (got '$backend')" >&2
    exit 2
    ;;
esac
case "$segments" in
  ''|*[!0-9]*|0)
    echo "qualify: --segments must be a positive integer (got '$segments')" >&2
    exit 2
    ;;
esac

# Fail closed with a message when there is no GPU stack at all (issue 064
# leg a): without this, the bare pipeline below aborts silently under
# `pipefail`+`set -e` (grep exits 1 on empty input) with no explanation.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "qualify: nvidia-smi not found — GPU qualification needs a CUDA host" >&2
  exit 4
fi
held_mib="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null \
  | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) s += $i} END {print s + 0}')"
if [ "$held_mib" -gt 2048 ]; then
  echo "qualify: GPU busy (${held_mib} MiB held) — backing off, run later" >&2
  exit 3
fi
echo "qualify: GPU idle (${held_mib} MiB held) — proceeding" >&2
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
if [ ! -f "$run_dir/manifest.json" ]; then
  echo "qualify: no manifest.json in $run_dir — configure first:" >&2
  echo "  ./scripts/run.sh configure <name> --backend ${backend} --segments <N> \\" >&2
  echo "    --style 'pastel neon line-art, peaceful'" >&2
  exit 2
fi
# Two-verb CLI: the run dir must be the generate-addressable
# output/<name> (configure + generate take NAME, not --run).
./scripts/run.sh generate "$(basename "$run_dir")"
# Artifact persistence (issue 064 leg c, 060): the summary used to be
# stdout-only, so every qualification evaporated. Tee to reports/ and
# print the path; the filename carries backend run id + date.
# Run dir travels via the environment (never shell-interpolated into the
# python snippet): paths with spaces/quotes would otherwise break the
# quoting or inject code (single quotes inside double quotes do not expand).
artifact="reports/qual-$(basename "$run_dir")-$(date +%F).json"
docker run --rm "$(voyage_user_args)" "${VOYAGE_CACHE_ENV[@]}" -w /app -v "$PWD:/app" -e RUN_DIR="$run_dir" voyage:latest \
  python -c 'import json, os; from tests.test_qualification import summarize_run; \
print(json.dumps(summarize_run(os.environ["RUN_DIR"]), indent=2))' \
  | tee "$artifact"
echo "qualify: summary saved to $artifact" >&2
