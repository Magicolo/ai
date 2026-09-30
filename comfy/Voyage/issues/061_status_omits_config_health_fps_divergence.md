# 061 — `status` omits load-bearing config and health; divergent `fps=0` fallbacks (`status` shows 0s, `validate` assumes 24)

**Severity:** MEDIUM

**File:line:** `voyage/cli.py:669-705` (`cmd_status`); `voyage/models.py:204-225` (`RunState.fps` unguarded, comment admits it); `voyage/cli.py:681` (`timeline_frames / fps if fps else 0`); `voyage/cli.py:922` (`fps if >0 else 24`)

**Overlaps with:** 029 (inspect/metrics rotation/fps divergence — sibling fps-fallback half; not a duplicate) / 055 (inspect metrics rotation-blind — sibling reader gap; not a duplicate)
- **Description:** `status` prints Video (backend/render/timeline/segments/blocks/GPU), World, Audio (backend/music/energy/buffered), Workers (hardcoded `idle` × 3), Stages (last commit + slowest), Storage (free GiB), last error. It never shows `quantization` (fp8/bf16 decides the 16 GB fit), SFX backend, `[augment]` floors, inspector on/off, beats/BPM/drift-every-n, take-seconds/ahead invariant, `resource_gauges` VRAM/disk trend, restart/circuit-breaker counts, or rotation state. `GPU:` is a live `doctor.probe()` at status time, not the run's hardware; `Workers:` is static text with no health check; `Free:` has no comparison against `min_free_space_gib`. Merged from the structure sweep (item 10b): `RunState` accepts `fps=0` (legacy tolerance); `status` then shows `Timeline: 0.00s` (division guarded to 0 — hides corruption), while `validate_run` silently substitutes 24 for the SFX-ledger timeline check. Same stored value, two meanings, neither flags the corruption.
- **Rationale:** `docs/OPERATIONS.md:222-223` sells `status` as the §59 long-run monitor. An operator asking "will the next segment OOM / is disk critical / what precision is this run / is SFX on" gets no answer. Observability readers must share one log-source helper; divergent fallbacks for the same corrupt field turn a detectable error into two plausible-looking lies.
- **Evidence (re-verified live 2026-09-30):**
```python
# voyage/cli.py:698-704,681 (live; sweep cited 659-665/642, drifted +~40 by concurrent growth)
gpus = probe().get("gpus")                      # live host probe, not run hardware
for name in ("video", "audio", "director"):
    print(f"  {name}: idle ({backend} backend; workers run during `voyage run`)")
free_gib = shutil.disk_usage(run_dir).free / (1024**3)
print(f"  Free: {free_gib:.1f} GiB")            # no min_free_space_gib comparison
# voyage/cli.py:681 vs :922 (live; sweep cited 642/883, drifted +~40 by concurrent growth)
seconds = state.timeline_frames / state.fps if state.fps else 0
fps = state.fps if isinstance(state.fps, int) and state.fps > 0 else 24
```
`grep -n "sfx\|augment\|inspector\|quantization\|beats\|drift" voyage/cli.py | grep print` → zero hits inside `cmd_status`.
- **Repro:** `./scripts/run.sh status --run <dir>` on a bf16+SFX+augment run — none of those words appear. Stop the run, upgrade the driver, re-run `status` — the `GPU:` line changes although the run never did. `RunState(run_id='x',fps=0,…)` validates; `status` math yields `0`, `validate` math yields `timeline_frames/24`.
- **Fix candidates:** Add Config section (quantization, SFX, augment floors, inspector, beats/BPM/drift, take-seconds/ahead); replace live GPU with manifest-recorded hardware + live probe labeled as such; add `resource_gauges` trend + restart/circuit-breaker counts; warn when `free < min_free_space_gib`; route `inspect metrics` through `_read_all_metric_events` (see 055); add a `validate_run` error for `fps<=0` (fail-loud) while keeping the 24-fallback for SFX math only after reporting.
- **Refs:** `docs/OPERATIONS.md:222-228`; `docs/STATE_AND_RECOVERY.md:40-42` (knobs `status` should echo); health-dashboard guidance (VRAM/disk-by-mount/service checks; alert >85% disk, VRAM pressure, temp).

### Resolution log (2026-09-30, Rank-2 batch)

- **Verdict: omissions LIVE and fixed; fps-divergence half already dead.**
  Re-verified in-container CPU-only 2026-09-30: all 11 keywords
  (sfx/augment/inspector/quantization/beats/drift/take/ahead/gauges/
  restart/circuit) absent from `cmd_status`; live `status` on a fake run
  showed none of them. The `validate` half of the fps divergence is
  already fail-loud (`voyage/cli.py` `validate_run` appends
  `state fps is corrupt` for `fps<=0`, probed live with `fps=0` → error);
  only the `status` display guard (`if state.fps else 0`) remained.
- **Fix (`voyage/cli.py:698` `cmd_status`, additive only — every
  pre-existing line kept for the 049-status substring tests):**
  - Video: `Quantization:` + `Device:`; corrupt-fps `WARN` line next to
    the Timeline (points at `voyage validate`).
  - `GPU (live probe):` label + `Hardware (recorded at init):` from the
    manifest (labeled; `cmd_init` still records only a note — enriching
    init-time hardware is a follow-up, not this issue).
  - New `Config` section: SFX backend/device/size, augment fps +
    resolution floors (or `disabled`), inspector on/off + model id,
    beats/segment + take/ahead values with the invariant verdict,
    drift cadence, director backend/device.
  - Workers: `Restarts:` per-worker counts + `circuit-breakers open:`
    from `_status_restart_counts()` (`:678`, over `_read_all_metric_events`).
  - `Gauges (last N)` trend via `bench.summarize_gauges` (RSS + disk
    first→last); Storage compares free space against
    `min_free_space_gib` (`doctor.meets_reserve`) with a below-reserve WARN.
- **Tests:** new rank-2 module (config section, recorded/live labels,
  forced-reserve WARN via `min_free_space_gib = 999999.0`, gauges +
  restarts after 1 committed segment); all related suites green
  (150 passed incl. the 049-status tests).
- **Residual/DESIGN proposal (text only):** `status` still cannot show
  live worker health (workers only run under `voyage run` — the static
  `idle` lines stay honest); `inspect metrics` reader unification stays
  with 055; recording real probe facts into the manifest at `init`
  (so `status` can diff recorded-vs-live hardware) is the natural next
  slice and touches `cmd_init` (outside this batch's owned regions).
