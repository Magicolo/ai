# 215 — Partial-leg finalize still takes the legacy all-at-once `run_finalize_model_pass` path (HIGH)

## Technical description
`media.py:1393-1412` — full legs → durable sidecar; partial legs
(interp-only or ESRGAN-only) → legacy `run_finalize_model_pass`, which
decodes **all** segments to PNGs then loads **all** tensors at once
(`voyage/augment.py:1096-1120`):

```python
decoded_paths.extend(chunk_frames_found)   # every segment
source_tensors = load_png_frames_as_tensors(decoded_paths)  # all at once
```

## Rationale
`enhance_frames` explicitly supports half-provisioned stacks ("absent legs
skip … still runs the available leg"), so partial provisioning is a
supported state — and that state routes to the unbounded path. 256 segs ×
232f ≈ 60k frames × 6.75 MB (2048×1152 u8) ≈ 400 GB PNG staging + ~1.6 TB
f32 if materialized. This is the pre-cleanup OOM/disk-full shape the
per-chunk-mp4 work was built to kill, still reachable by config.

## Live evidence
```
$ sed -n '1393,1412p' voyage/media.py
 ...Partial legs keep the legacy all-or-nothing tmpdir flow...
 from voyage.augment import run_finalize_model_pass
$ grep -n "decoded_paths.extend\|load_png_frames_as_tensors(decoded" voyage/augment.py
1098, 1120
```

## Repro
`weights` with `rife=None, realesrgan=<real>` + 3 fake segments →
`finalize_run` selects the legacy branch (code-path assertion, no GPU).

## Source refs
- `Voyage/voyage/media.py:1393-1412`
- `Voyage/voyage/augment.py:1096-1120`

## Online sources
- In-tree `STAGE_CHUNK_MP4`/cleanup docs ("mp4 supersedes ~20x PNG weight …
  holds megabytes instead of gigabytes").

## Fix candidates
1. Route partial legs through the durable sidecar (it already supports
   per-leg skips).
2. Or chunk the legacy loader.
3. Or refuse partial legs fail-loud instead of silently taking the unbounded
   path.

## Log
- Track D sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07, live re-probe)
- **Confirmed.** `voyage/media.py:1393-1408` routed partial legs
  (interp leg xor ESRGAN leg) to legacy `run_finalize_model_pass`, whose
  loader (`voyage/augment.py:1096-1120`) decodes ALL segments to PNGs
  then `load_png_frames_as_tensors` on the whole list — unbounded, as
  filed. Repro shape verified by code path (partial `AugmentWeights` +
  demanded work selects the legacy branch; full legs select the durable
  sidecar; no-leg selects the ffmpeg fallback).
- **Candidate 1 (durable routing) deliberately NOT taken:** the durable
  entry's `weights_key_for` raises for partial stacks, `_poll_to_completion`'s
  `include_upscale/include_interp` skips are not plumbed through
  `run_durable_model_pass`, and drain expects interp-ledgered chunks —
  making partial durable-correct needs new key shapes, joint-unit
  semantics for missing legs, and drain fallback paths, none of which is
  verifiable without GPU work (explicitly out of scope). Candidate 3
  (fail loud) was chosen: it is small, CPU-testable, and satisfies "never
  silently unbounded".

## Progress log (2026-10-07)
- The legacy else-branch in `_do_model_pass` now raises `MediaError`
  naming the missing leg(s) (`<backend> interp weights` and/or
  `realesrgan weights`) with both remedies: provision the missing leg,
  or set `upscale=1/interpolate=1` for the bounded ffmpeg fallback.
  `run_finalize_model_pass` is no longer called from `media.py`
  (definition stays for other/historical callers).
- Stale `Partial legs keep the legacy ... flow` comments updated to the
  fail-loud contract (two sites).
- New `tests/test_partial_leg_fail_loud.py`: ESRGAN-only and interp-only
  partials both raise (durable + legacy passes stubbed forbidden to pin
  the code path), plus a message-remedies pin. Full-weight runs are
  unaffected (both-legs → durable, exercised by existing suites).
- Verification (in-container `voyage:latest`): ruff check + format clean,
  mypy clean on `voyage/media.py`; new file 3 passed; finalize/augment
  neighbors 112 passed (`test_finalize_av_stream/fastpath/gpu_defaults/
  model_music_parallel/encode_rank2`, `test_interp_backend`,
  `test_augment_finalize_wire`, `test_issue_166_model_pass_select`,
  `test_native_skips_model_pass`, `test_media_augment_unified_083`,
  `test_augment_background`, `test_sfx_caption_render_161`). Four
  failures in the wider sweep (`test_generate_skip_flags` x3,
  `test_generation_stack` x1) are foreign — concurrent agents'
  uncommitted `cli_core.py` skip-key/`_sfx_overrides` and `cli_generate.py`
  `mastering_enabled` work — untouched per §9.

## Resolution (2026-10-07)
- **RESOLVED (fail-loud).** Partial-leg finalize can no longer silently
  enter the unbounded legacy flow; durable routing for partial stacks
  stays open as future work (needs GPU-verified key/drain semantics).
  Files: `voyage/media.py`, `tests/test_partial_leg_fail_loud.py`
  (new). No commit (per mandate).
