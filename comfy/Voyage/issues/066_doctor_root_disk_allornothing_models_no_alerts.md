# 066 — `doctor.probe()` measures disk at `/`, reports `models_ok` all-or-nothing, and alerts on nothing

**Severity:** MEDIUM

**File:line:** `voyage/doctor.py:135-165` (`disk_free_gib: _disk_free_gib(Path("/"))`, `models_ok` conjunction); consumers `voyage/cli.py:698-704` (`status`)
- **Description:** Disk-full is a first-class state (`PAUSED_DISK_FULL`, finalize preflight, `min_free_space_gib`), but doctor measures free space at filesystem root `/` — not the run dir, not `/models`, not `/tmp` (all three are distinct mounts in the standard `run.sh`: `$PWD:/app`, `/tmp:/tmp`, `$models:/models`). `models_ok` is `exists and all(checks ok)` — one missing *optional* stack (inspector VLM) flips the whole flag red. There are no thresholds anywhere: no VRAM pressure line, no compute-capability / CUDA-runtime / temp / throttle facts (all listed as "remaining gaps" in the docstring and TROUBLESHOOTING), no comparison against `min_free_space_gib`.
- **Rationale:** Health checks should answer "can I start / should I pause" per mount with thresholds, not report one root number. Monitoring guidance converges on: disk-by-mount with >85–90% alerts, VRAM used/pressure + temp, service reachability — doctor covers none as alerts, and `status`/`generate` consume its numbers without thresholds.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/doctor.py:149-164 (live)
models_ok = (
    bool(model_facts["exists"]) and all(check["ok"] for check in model_facts["checks"].values())
    if model_facts["checks"]
    else False
)
"disk_free_gib": _disk_free_gib(Path("/")),
```
`grep -in "vram\|temperature\|throttle\|capability\|cuda_runtime" voyage/doctor.py` → no VRAM/temp/throttle/capability/runtime facts (words appear only in gap documentation). `status` prints `Free: X GiB` with no reserve comparison.
- **Repro:** Mount a small `/tmp`, fill `/models` past 90% while `/` stays roomy → `voyage doctor` still reports healthy `disk_free_gib`. Delete only the optional inspector snapshot → `models_ok: false` although every required stack verifies.
- **Fix candidates:** Report `disk_free_gib` per path (`run_dir`, `/models`, `/tmp`) + `meets_reserve: bool` vs `min_free_space_gib`; split `models_ok` into `models_ok_required` / `models_ok_all`; add VRAM free/total + compute capability + CUDA runtime + driver from `nvidia-smi` (cheap, already shelled); emit WARN lines (and non-zero exit tiers) at 85%/90% disk and VRAM-pressure thresholds.
- **Refs:** `docs/TROUBLESHOOTING.md` (gap list + disk-full policy); `docs/STATE_AND_RECOVERY.md:40-42` (reserve knob); local-AI monitoring guidance ("disk space by mount", "VRAM used", "alert >85% disk / VRAM pressure / temp ≥84 °C sustained").

### Track-E scope note (structure-sweep items 7–12 triage)

Structure-sweep items 8 (scoreboard), 9 (console), 10a (rotation-blind metrics) and 10b (fps divergence) are merged into 062/063/055/061 respectively. Item 7 (`voyage/prompts.py:27-35,64-80,158-199` — staged-prompt truncation/repetition + 7-substring style-override blocklist) and item 11 (`voyage/workers/loop.py:91-168` — fd-level stdout quarantine, `checked_request` extra-key blindness, all-`VoyageError`-fatal taxonomy) were reviewed and left out of Track E as out-of-scope: prompts/director belongs to the prompts track, worker RPC framing/taxonomy to the workers/RPC track (the stdout-quarantine half is cross-referenced in 056). No finding dropped silently — route those two to their owning tracks.

### Resolution log (2026-09-30, Rank-2 batch)

- **Verdict: LIVE, fixed.** Re-verified in-container CPU-only 2026-09-30:
  `probe()` returned `disk_free_gib` from `/` only (279.5 GiB while
  `/models` was absent entirely), `models_ok: false` from the single
  conjunction, and no vram/temperature/capability/cuda-runtime keys
  (only gap documentation). All three legs confirmed as-read.
- **Fix (`voyage/doctor.py`, `models_ok`/`disk_free_gib` keys kept for
  backward compatibility):**
  - `disk_mount_facts()` + `probe()["disk_by_mount"]` = per-mount
    `{free_gib, total_gib, used_fraction}` for root/models(`/models` via
    `models_dir()` env mirror)/tmp; unknown mounts report None, never
    healthy-zero. `meets_reserve(free, min)` returns True/False/None
    (None = unknown, printed as unknown — consumed by `status` reserve).
  - `models_ok_required` (all stacks except `inspector-qwen35`) vs
    `models_ok_all` (= legacy conjunction, `models_ok` aliases it);
    optional set is the named `_OPTIONAL_MODEL_STACKS` constant.
  - `nvidia-smi` query extended to
    `compute_cap,temperature.gpu`; `_parse_gpu_details()` yields per-GPU
    VRAM total/free (MiB→GiB), driver, compute cap, temp (short rows
    degrade per-field); `driver`, `compute_cap`, `cuda_runtime` (lazy
    torch, None off-GPU) are top-level facts.
  - `health_alerts(facts, *, min_free_gib)` pure function: CRIT ≥90% /
    WARN ≥85% disk per mount, below-reserve WARN, VRAM free <15% WARN,
    temp ≥84 C WARN, missing *required* stacks WARN. Unknown facts stay
    silent. Slim-box live probe yields exactly the missing-stacks WARN.
- **Tests:** rank-2 module (per-mount keys, reserve tri-state, required/
  all split, fact presence off-GPU, synthetic-threshold alerts incl. the
  empty-facts silence case, `voyage doctor` exit path); `test_doctor.py`
  untouched and green; related suites 150 passed; ruff + format + mypy
  strict clean.
- **Residual/DESIGN proposal (text only):** `cmd_doctor` does not yet
  print `health_alerts` lines (outside this batch's owned cli regions —
  benchmark-env + status only); wiring `health_alerts(probe(),
  min_free_gib=<run reserve or None>)` into `cmd_doctor` output with the
  existing exit-0/1 ffmpeg semantics is the one-line follow-up.
  Sustained-temp (≥84 C over time, needs sampling) and service
  reachability stay future per TROUBLESHOOTING gaps.
