# 153 — SFX stem cache: exact-float hit check + pre-render unlink + truncate logging orphan retries

- **Severity:** MEDIUM (correctness-adjacent cache/ledger edge — wasted GPU renders and false-missing stems, never silent corruption)
- **File:line:** `Voyage/voyage/sfx_finalize.py:316-359` (`_render_one`), esp. `:326` (`record.get("duration") == window.duration`), `:330-331` (`stem.unlink()` pre-render), `:296` (`existing` read-once dict), `:350-357` (truncated-tail re-log), `:226-239` (order-sensitive `validate_sfx_ledger`), `:181-196` (`append_sfx_window`)
- **Area:** SFX stem reuse cache (below both previous windows; 054 covers the two-thread ledger race — this file is the single-worker cache/truncate/unlink logic)

## Description

Three independent edges in `_render_one`, all live in the same 44 lines:

1. **Exact-float hit check (`:326`).** A cached stem hits only when `record.get("duration") == window.duration` exactly. Planned durations are `min(start + 8.0, timeline) - start` floats; a re-finalize after any timeline perturbation (different probe rounding, `--sfx-caption` override changing nothing but re-planning) recomputes the same decimal through a different float path and misses. The miss is safe (re-render) but burns a full MMAudio window render (~1.23 s/8 s on small, full large cost on the default) for a byte-identical stem.
2. **Pre-render unlink (`:330-331`).** On a miss, `if backend == "mmaudio" and stem.exists(): stem.unlink()` deletes the old stem *before* `workers[slot].call("generate_sfx", …)` runs. If the worker raises, the process is killed, or the timeline changed so the old stem is still the valid predecessor, the window now has neither a new stem nor the old one — but the ledger still points at `audio/sfx/wXXXX.wav`, so `validate_sfx_ledger` (`:226-239`) reports `sfx wXXXX missing …` on a run that had a good stem seconds earlier. The delete-then-render ordering turns every transient render failure into a missing-stem validation failure.
3. **Truncate re-log orphans the plan (`:350-357`).** When the source yields a short tail, `resolved < window.duration` is logged as the record's duration (`logged = SfxWindow(…, float(resolved), …)` at `:351-357`, comment at `:348-349` says the ledger "records reality"). The next finalize re-plans the *full* `window.duration`, hits check (1) compares plan-full vs ledger-truncated, misses forever, re-renders the same short tail forever. Worse, `existing` (`:296`) is built once before the pool and keyed by `window_id` with last-write-wins dict semantics, while `validate_sfx_ledger` (`:228-239`) walks the raw ledger *in file order* with an overlap-tolerance cursor — a re-rendered window appends a second `wXXXX` line, and the validator sees the stale first line before the fresh one (coverage-gap false positive on the exact windows that truncated).

## Rationale

The stem cache exists to make re-finalize cheap (deterministic seeds at `plan_sfx_windows:128-130` say "re-finalize re-renders identical bytes" — the cache is what makes that free). Each edge defeats it in a different common case: float miss on any re-probe, unlink on any transient failure, truncate-miss on every short-tail timeline. None corrupts audio (failures are loud: re-render cost or `missing` validation errors), but together they make the "SFX re-finalize is incremental" claim false on exactly the timelines (short tails, re-probed finals) users iterate on.

## Live evidence

- `sed -n '316,359p' voyage/sfx_finalize.py` — `:326` exact `==` on duration; `:330-331` unlink before the `:332-345` worker call; `:350-357` `resolved` re-log with the "records reality" comment.
- `sed -n '296,296p' voyage/sfx_finalize.py` — `existing = {record["window_id"]: record …}` (last-wins dedupe, read once before the pool at `:361-365`; no re-read, so a retry appends a duplicate `window_id` line).
- `sed -n '226,239p' voyage/sfx_finalize.py` — validator walks `records` in file order with `cursor = covered_until - SFX_WINDOW_OVERLAP`; duplicate `window_id` lines are visited twice, the stale duration first.
- `sed -n '181,196p' voyage/sfx_finalize.py` — `append_sfx_window` is append-only (`open("a")`), no replace/dedupe, so every re-render adds a line the validator will walk.
- Overlap check: 054 is the concurrent-append race under `num_workers=2` (threading, needs `Lock`/serial-append); this file reproduces with `num_workers=1`, no threads, purely on cache-key/unlink/truncate logic.

## Repro

Static (deterministic, CPU-only): (1) render one window, then call `_render_one` with a `window.duration` differing by 1e-9 → miss + full re-render despite identical bytes; (2) place a valid stem, force `generate_sfx` to raise (bad `video_path`), observe the stem deleted and `validate_sfx_ledger` reporting `missing`; (3) finalize a timeline whose last window truncates (e.g. 8.4 s timeline → stub-merge + short tail), re-finalize unchanged → the truncated window re-renders every time and the ledger gains a duplicate `window_id` line per run.

## Fix candidates

1. Tolerance hit check: `abs(record.duration - window.duration) <= AV_ALIGNMENT_TOLERANCE_SECONDS` (the same 0.6 s budget the bed/timeline checks already use) plus caption/seed/model_size equality; log the near-miss at debug so real plan changes stay visible.
2. Render-to-temp + atomic replace: render to `wXXXX.wav.partial`, `os.replace` on success (the atomic rule §12) — a failed render leaves the old stem and the old ledger line intact, and `validate` never sees a half-written window.
3. Truncate-aware planning: when a ledger record's duration is shorter than the plan's, re-plan from the ledger (or key the cache on `(window_id, caption, seed, model_size)` and treat duration as observed, not as key) so truncated tails hit; alternatively make `validate_sfx_ledger` dedupe by `window_id` (last-wins, mirroring `:296`) and walk sorted by `start` so ledger order can never false-positive coverage (also fixes the 054-adjacent ordering fragility for the single-worker case).
4. Tests: float-perturbed duration hits; worker-failure leaves old stem + clean validate; truncate → re-finalize is a full cache hit with no ledger growth.

## Refs

- `Voyage/voyage/sfx_finalize.py:76-132,181-244,293-365`; `Voyage/voyage/media.py:441-446` (`AV_ALIGNMENT_TOLERANCE_SECONDS`); DESIGN three-caption doctrine / SFX slices.
- Adjacent, not overlapping: 054 (two-thread concurrent append race — threading, this file is single-worker cache logic); 101 (ledger fsync gaps — durability, not key semantics); 098 (orphan scan gaps — validator coverage, not the cache that feeds it).
