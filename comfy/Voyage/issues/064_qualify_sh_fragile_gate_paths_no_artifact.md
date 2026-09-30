# 064 — `qualify.sh` idle gate is fragile, accepts known-bad relative paths, saves no artifact, has no disk preflight

**Severity:** MEDIUM

**File:line:** `scripts/qualify.sh:15-36`; `reports/video-backends.md:85-90,111-117`

**Overlaps with:** 060 (benchmark env/artifacts — sibling harness gap; not a duplicate)
- **Description:** (a) The contention gate `held_mib="$(nvidia-smi … | grep -oE … | awk …)"` yields `0` when `nvidia-smi` is absent (empty pipe → `awk … {print s+0}`), so a GPU-less box silently *passes* the gate and fails minutes later in `benchmark video`. Under `set -u`, an unset-but-empty `held_mib` is also a latent trip. (b) The script accepts a relative `<run-dir>` although the report documents the relative-path doubling bug twice as still-open (host-relative `--run` doubles paths; procedural note on dry runs). (c) The final `summarize_run` JSON is printed to stdout only — no `tee` to `logs/` or `reports/`, so the qualification evaporates. (d) No disk-space preflight despite 3-segment GPU runs + finalize needing GiBs.
- **Rationale:** A qualification harness that can run under contention, on a known-bad path shape, and leave no artifact behind produces the exact unreproducible numbers 060 complains about. The repo's own GPU-contention rule ("NEVER run under contention … aborts when >2 GiB held") deserves a gate that fails closed, not open.
- **Evidence (re-verified live 2026-09-30):** `qualify.sh:16-21` (gate), `:28-30` (benchmark/run/validate with `"$run_dir"` unvalidated), `:34-36` (bare `print(json.dumps(…))`) — all match the 36-line live file. `bash -n` passes (syntax ≠ correctness). Cache-env grep shows `qualify.sh` sets only `PYTHONDONTWRITEBYTECODE=1`, unlike the gate scripts — harmless here (no ruff/mypy/pytest in this path) but confirms it was written outside the cache-hygiene contract.
- **Repro:** `PATH=/usr/bin:/bin ./scripts/qualify.sh ./output/q` on a box without `nvidia-smi` → gate passes with `held_mib=0`, then `benchmark video` fails late. `./scripts/qualify.sh output/rel-path` → reproduces the doubling stall from the report.
- **Fix candidates:** `command -v nvidia-smi || exit 4` before the gate; `realpath`/absolute-path enforcement (`--run` must be absolute, matching the codebase invariant); `tee "reports/qual-$(date +%F)-$backend.json"`; `check_free_space` preflight mirroring `generate`.
- **Refs:** `reports/video-backends.md:85-117`; `scripts/run.sh` (absolute-path invariant); `docs/BENCHMARKING.md:8-18`.
- **Correction (second pass, 105):** leg (a) above is wrong on the idle-GPU case — when `nvidia-smi` prints nothing, `grep -oE` exits 1 and `pipefail`+`set -e` aborts silently (fail-closed, no message), not fail-open with `held_mib=0`. See 105 for the live probe. Legs (b)-(d) stand.

### Resolution log (2026-09-30, Rank-2 batch)

- **Verdict: legs (b)-(d) LIVE and fixed; leg (a) as-written superseded
  by the 105 correction, fix direction unchanged.** Re-verified
  CPU-only 2026-09-30: (a) with a stub `nvidia-smi` exiting 127/empty,
  the bare pipeline aborts rc=1 with NO message under `pipefail`+`set -e`
  (fail-closed-but-silent — the "passes with held_mib=0" reading is
  wrong per 105; the fragility stands, the mechanism differs); (b) no
  path validation before `"$run_dir"`; (c) `grep -c tee` = 0; (d) no
  disk/`df` reference. `bash -n` passed throughout (syntax ≠ correctness).
- **Fix (`scripts/qualify.sh`, stages otherwise untouched):**
  - `command -v nvidia-smi || exit 4` with an explanatory message before
    the gate (fail-closed, loudly).
  - `case "$run_dir" in /*)` absolute-path enforcement → exit 2
    (matches the `resolve_run_dir` codebase invariant).
  - Disk preflight: `df -k` avail under the run dir vs
    `${QUALIFY_MIN_FREE_GIB:-5}` (mirrors `DEV_MIN_FREE_SPACE_GIB`) →
    exit 5; unknown space skips, never aborts.
  - Final `summarize_run` JSON teed to
    `reports/qual-<run-basename>-<date>.json` + stderr path line.
- **Live-caught bug during verification:** the first preflight draft
  (`avail_kib="$(df ...)"` on a nonexistent dir) tripped `set -e` via
  the failing substitution and died rc=1 before the voyage.toml check.
  Fixed with `|| true`; regression test pins exit 2 + the init hint.
- **Tests:** `bash -n`; `test_ops_visibility_rank2.py` qualify tests —
  no-`nvidia-smi` → exit 4 (restricted-PATH probe, no docker), relative
  path → exit 2 + `absolute` message (stub `nvidia-smi` echoing 0, no
  docker), missing absolute dir → exit 2 + init hint, plus static
  `tee`/`df`/`absolute` assertions.
- **Residual/DESIGN proposal (text only):** `reports/*.json` artifacts
  are untracked by policy choice (gitignore covers media extensions
  only) — decide commit-vs-gitignore per `reports/` policy; the GPU leg
  itself still needs an idle-4060 Ti run (`qualify.sh` on real hardware).
