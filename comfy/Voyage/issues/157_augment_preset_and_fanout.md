# 157 — Augment chunk path: no preset knob, model reload per chunk, full-batch stack before OOM-halving

- **Severity:** MEDIUM (perf + quality-control — the finalize hot path pays per-chunk loads and holds the full batch it claims to halve)
- **File:line:** `Voyage/voyage/augment.py:178-213` (`ffmpeg_encode_chunk`: crf knob, no preset) + `:266-281` (`run_augment_chunks`), esp. `:280` (`ThreadPoolExecutor(max_workers=…)`) → `Voyage/voyage/workers/augment_worker.py:356-394` (`upscale_frames`), esp. `:376` (`_load_rrdb_net` per call), `:379` (`torch.stack(…all frames…).to(…)`), `:344-347` (OOM-halving with conditional `empty_cache`)
- **Area:** augment chunk execution (below both previous windows; 046 covers the linear rescan, 047 covers reload-per-call/stack-first at spike scope, 050 covers the finalize double-encode — this file is the preset gap + fan-out + batch-peak interaction)

## Description

Three edges that multiply on long sequences:

1. **No preset knob (`augment.py:178-213`).** `ffmpeg_encode_chunk` exposes `crf` (`:183`, validated `:187-189`) but hardcodes `-c:v libx264 -pix_fmt yuv420p` with no `-preset` (`:200-206`). Every chunk encodes at libx264's default (`medium`) while the rest of the pipeline standardizes on `veryfast` for intermediates (`media.py:1062-1063/1101-1102`, flagged by 050 as itself unknobbed). Chunk encodes are the per-chunk hot loop (one spawn per chunk plus the final encode); operators cannot trade preset for speed without editing the function, and a future preset unification (050's `FinalizeOptions` ask) will miss this second encode site.
2. **Model reload per chunk (`augment_worker.py:376`).** `upscale_frames` calls `_load_rrdb_net(weights_path)` on *every invocation* — and the orchestration calls it once per chunk (`run_augment_chunks` maps chunks → worker). A 284-frame sequence at 32-frame chunks pays 9 full RRDBNet builds + `torch.load` + `load_state_dict(strict=True)` + `.to(device).half()` for the identical weights file. `_load_film_net` (`:412`, `:440`) has the same per-call shape for interpolate paths. There is no session/resident-handle concept (compare the video/audio/SFX resident stacks with `init` + `_require_stack`).
3. **Full-batch stack before halving (`:379` vs `:344-347`).** `_run_stacked` halves on OOM down to singles (Comfy recipe, `:333-353`), but `upscale_frames` first materializes the *entire* chunk as one stacked tensor (`torch.stack([…for frame in frames]).to(device, dtype)` at `:379-381`) — the peak the halving is supposed to avoid is already held before `forward` runs. On a 32-frame 1536×1024 fp16 chunk the stacked input alone is ~600 MB plus the x4 activations; halving only shrinks the *forward* slices, never the input. `empty_cache` on split (`:346-347`) is additionally gated on `torch.cuda.is_available()`, so CPU-OOM paths recurse without Sector 7 relief.

## Rationale

Chunking exists so long sequences fit (the `video_export.json` 32-frame precedent in `augment.py:34-35`, the VHS_BatchManager trade-off documented at `augment_plan:106-112`). Per-chunk reloads + full-batch peaks + medium-preset encodes erase the margin chunking buys: the per-chunk fixed cost dominates at small chunk sizes (which memory pressure forces), and the peak that OOMs the container is allocated by our own stacking line, not the model. Each edge is individually small; together they set the ceiling on sequence length.

## Live evidence

- `sed -n '178,213p' voyage/augment.py` — argv at `:191-207` has `-crf` but no `-preset`; `rg -n "preset" voyage/augment.py voyage/workers/augment_worker.py` → no hits (vs 2 in `media.py`).
- `sed -n '356,394p' voyage/workers/augment_worker.py` — `_load_rrdb_net` at `:376` inside the per-call path; `torch.stack` at `:379-381` before `_run_stacked` at `:384`; `_run_stacked` at `:333-353` with the `is_available`-gated `empty_cache` at `:346-347`.
- `sed -n '266,281p' voyage/augment.py` — `:277-279` serial fast path, `:280-281` `ThreadPoolExecutor(2)` fan-out with `lambda chunk: worker(chunk, chunk.device)`; the worker closure re-enters `upscale_frames` (hence reload) per thread, with two threads sharing one CUDA context and no documented thread-safety for the per-call model (compare SFX's one-worker-per-process sharding at `sfx_finalize.py:301-314`).
- Overlap check: 046 is the chunk *decode* linear rescan (ffmpeg seeking, different function); 047 is the spike-scope reload/stack-first note (this file adds the preset gap, the fan-out interaction, and the stack-vs-halve peak ordering with line pins); 050 is the finalize parts+final double-encode (different stage, same missing-knob family).

## Repro

Static + measurement: (1) `rg -n "preset" voyage/augment.py` → empty (every chunk encodes `medium` regardless of CRF); (2) instrument `_load_rrdb_net` call count over a 3-chunk plan → 3 loads of identical bytes (9 on the 284-frame example); (3) trace peak allocation on a 32-frame chunk → the `:379` stacked tensor is live across the whole `_run_stacked` recursion (halving splits `stacked[0:half]` views of the same storage — peak never drops until the call returns).

## Fix candidates

1. Thread a `preset` parameter through `ffmpeg_encode_chunk` (default `veryfast` to match intermediates, validated against a small allow-list) and include it in the chunk metric so soak can trend preset-vs-wall like CRF.
2. Resident handles: load once per device (`functools.lru_cache` on weights mtime+device, or an explicit session object the chunk runner owns) so N chunks pay 1 load; evict between stages per the §40 discipline.
3. Stack lazily: move stacking *inside* `_run_stacked` (stack halves, not the whole list) or iterate singles into the halving runner, so peak follows the halved slice, not the chunk; call `empty_cache` unconditionally on split (it is a no-op without CUDA, the guard buys nothing).
4. Document the `:280` fan-out contract (one model per thread vs shared handle; SFX precedent is process-per-GPU) and test chunk-count determinism under both serial and parallel paths.
5. Tests: preset override reaches argv; 3-chunk run loads weights once (counter); peak-allocation test asserts halved input residency.

## Refs

 - `Voyage/voyage/augment.py:34-50,136-213,266-281`; `Voyage/voyage/workers/augment_worker.py:301-353,356-394`; `Voyage/voyage/media.py:1046-1110` (050's encode sites); DESIGN augment track + §40 (residency).
 - Adjacent, not overlapping: 046 (decode rescan); 047 (spike-scope reload/stack-first — this file is the preset + fan-out + peak-ordering remainder); 050 (finalize double-encode knobs).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise PARTLY SUPERSEDED: 047 already landed resident caches (`_RRDB_CACHE`/`_FILM_CACHE`), list-halving (`_run_frame_batches` — the full-batch stack never builds), and gc-before-empty_cache — so candidates 2 (resident handles) and the stack-vs-halve peak ordering are done. Remaining worker-owned: the two `if torch.cuda.is_available()` guards around `empty_cache` (`_run_stacked`, `_run_frame_batches`) — still present as-read. The preset knob + fan-out contract live in `augment.py` (E1's, out of scope) — residuals. TDD: `tests/test_e2_augment_worker_157_193.py` OOM-split leg failed pre-fix (no `empty_cache` call without CUDA), green post-fix. Note: `augment_worker.py` is not in the brief's OWN-FILES list but the brief explicitly assigns "the worker-owned half" here (and 193's file) to Group E2 — hunks kept to the cited lines only.

## Resolution

- Verdict: WORKER-OWNED REMAINDER FIXED; orchestration legs RESIDUAL (below).
- Changes (`voyage/workers/augment_worker.py` only): both OOM-split `empty_cache` calls unconditional with why-comments (`empty_cache` is a no-op without CUDA — the guard bought nothing and skipped relief on CPU-OOM recursion). Tested with stubbed torch (no real torch needed).
- Files changed: `voyage/workers/augment_worker.py` (+ OOM leg in `tests/test_e2_augment_worker_157_193.py`).
- Test evidence (in-container `voyage:latest`, CPU-only): new-file OOM leg green; all `test_augment_*` (125 passed, 3 skipped) green. Ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — augment track): "OOM-split paths call `empty_cache` unconditionally (no-op without CUDA) — allocator relief never depends on device presence."
- Residuals (out of scope, precise — for the E1 `augment.py` owner): (1) `voyage/augment.py:178-213` (`ffmpeg_encode_chunk`): thread a `preset` parameter (default `veryfast` to match intermediates, validated against an allow-list) + include it in the chunk metric — the `augment_benchmark_setup` shape in `voyage/bench.py` (154/163, this pass) already carries the field; (2) `:266-281` (`run_augment_chunks` `ThreadPoolExecutor(2)` fan-out): document the thread contract (one model per thread vs shared resident handle; SFX precedent is process-per-GPU) + chunk-count determinism under both paths.

## Progress log (2026-09-30, CLI track — orchestration half, this pass)

- Re-verified live first: `ffmpeg_encode_chunk` still had `crf` but no
  `preset` (`voyage/augment.py:233-268` as-read); `run_augment_chunks` fanned
  out immediately over `ThreadPoolExecutor(2)`; worker-side `_RRDB_CACHE` /
  `_FILM_CACHE` (`augment_worker.py:85/93`, keyed by weights+device) and
  `_run_frame_batches` list-halving confirmed present (batch-8/E2 halves
  landed — not re-done). "Halve the list before stacking" verified done;
  preset + fan-out gate done here. No `preset` in `media.py`'s chunked path
  needed touching (media.py out of scope).
- TDD: `tests/test_augment_preset_fanout.py` (5 tests) written first — all
  failed pre-fix in-container, green post-fix.

## Resolution (2026-09-30, CLI track — orchestration half, this pass)

- Verdict: ORCHESTRATION HALF FIXED in the owned file; worker/production-path
  legs RESIDUAL.
- Changes (`voyage/augment.py` only): `CHUNK_PRESET_DEFAULT = "veryfast"` +
  `CHUNK_PRESETS` (mirrors `media.FINALIZE_PRESETS` entry-for-entry, pinned by
  test — local copy because `media` imports `augment`, so reuse would cycle)
  + `validate_chunk_preset` + `preset` kwarg on `ffmpeg_encode_chunk`
  (default `veryfast` per the issue; behavior change from libx264-default
  `medium` — faster chunk encodes); `CHUNK_CRF_DEFAULT = 15` (value-unchanged
  extraction for the benchmark setup to reference); `run_augment_chunks`
  warm-first gate (chunk 0 serially populates any resident cache, remainder
  fans out; order contract preserved; first-chunk failure fails fast) +
  module-docstring contract note.
- Test evidence (in-container `voyage:latest`, CPU-only): new file 5 passed
  (incl. event-synced warm-first ordering, no timing margins); neighbors
  `test_augment_runner.py` + `test_augment_plan.py`-adjacent suites green.
  Gates on touched files: ruff check + format-check + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — augment track):
  "Chunk encodes thread a `preset` knob (default `veryfast`, validated
  against the shared x264 vocabulary) recorded in the chunk metric, and the
  two-device fan-out warms the first chunk serially so resident model caches
  populate before threads spawn."
- Residuals: worker-side single-flight lock (`_RRDB_CACHE`/`_FILM_CACHE` exist
  but unlocked — a cold double-miss still double-loads; warm-first mitigates
  from orchestration; worker-file owner); `preset` in the production finalize
  encode path (`voyage/media.py` concat/vf sites — media owner; this change
  covers the `augment.py` chunked orchestration only).
