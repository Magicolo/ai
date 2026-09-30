# 045 — SFX conditioning does 3 ffmpeg spawns/window + two full-tensor copies

**Severity:** HIGH

**File:line:** `voyage/workers/sfx_mmaudio.py:110-167` (`_extract_frames`), `170-189` (`_convert`); `voyage/audio/mmaudio_sfx.py:241-278` (`render_window`), `287-300` (`evict`)

**Description:**
Every 8 s SFX window spawns two full ffmpeg rawvideo decodes (`_grab` CLIP 8 fps@384 + sync 25 fps@224) plus a third for the FLAC→WAV `_convert` — ~3 processes × N windows, each `capture_output` holding the whole window bytes. `_grab` does `np.frombuffer(stdout)` → `reshape` → `.copy()` → `torch.from_numpy(...).float().div_(255)` → `permute` (line 158), then `render_window` does `list(clip_frames)[:expected]` + `torch.stack` again (lines 249-257), so window pixels live 3–4× (bytes + numpy + per-frame tensors + stacked batch). A 60 s `MAX_WINDOW_SECONDS` window stacks `1500×3×224×224×4 ≈ 0.84 GiB` in one `torch.stack` before the model runs; the 8 s nominal is ~115 MiB sync + ~27 MiB CLIP, still triple-held. No `non_blocking` on `.to(device)`, no `empty_cache` after `generate`, and `evict()` does `.to("cpu")` before `del` (extra D2H copy of a large stack).

**Rationale:**
The synchformer ≥16-frame floor is handled by fail-loud truncation (`resolved` + 0.6 s tolerance, lines 223-230), not by pad-to-17 tiling — short-tail windows raise `RuntimeError` → retryable restart instead of rendering. The triple-hold + triple-spawn is the SFX analogue of the single-pass finalize the codebase already banned for MMAudio past ~200 frames.

**Live evidence (current tree):**
```
sfx_mmaudio.py:127: def _grab(rate, size, count): ... subprocess.run(..., capture_output=True)
sfx_mmaudio.py:150: completed = subprocess.run(command, capture_output=True, check=False)
sfx_mmaudio.py:153: pixels = len(completed.stdout) // (size*size*3)
sfx_mmaudio.py:156-158: np.frombuffer(...) → reshape → frames.copy() → torch.from_numpy(...).float().div_(255).permute(...)
sfx_mmaudio.py:161-164: wanted_clip/wanted_sync; clip=_grab(...); sync=_grab(...)  # two spawns
sfx_mmaudio.py:170-187: def _convert(...): third ffmpeg spawn per window
mmaudio_sfx.py:256-257: sync_batch = torch.stack(sync_list).unsqueeze(0); clip_batch = torch.stack(clip_list).unsqueeze(0)
mmaudio_sfx.py:265-266: stack.feature_utils.to(torch_device); stack.model.to(torch_device)  # no non_blocking
mmaudio_sfx.py:293-294: stack.model.to("cpu") then del  # D2H copy before free
```
`python3 -c "print(1500*3*224*224*4/1024**3)"` → 0.84 GiB single-stack worst case.

**Repro (CPU):**
```python
print(1500*3*224*224*4/1024**3)  # → 0.84 GiB for a 60 s window sync tensor
```

**Fix candidates:**
- Single ffmpeg pass at the higher rate/size + `scale`/`fps` split via `split` filter, or decode once and interpolate CPU-side.
- Stream frames (`Popen` + chunked `read`) directly into preallocated tensors; `torch.stack` once, `pin_memory` + `to(device, non_blocking=True)`.
- Enforce planner-level `duration ≤ 8–10 s` (cap effective stacked bytes) and pad short tails to 16 sync frames instead of failing loud.
- `evict`: `del` first (or `to("cpu", non_blocking=True)` only when needed), then `gc.collect()` + `empty_cache`.

**Refs:** PyTorch "Understanding GPU Memory 2" (`del` alone frees nothing under cycles; `gc.collect()` before `empty_cache`); `sfx_finalize.py:36-46` floor discussion; `video_longlive.py:827-833` evict lesson.

**Overlaps with:** 043/044/048 (all-at-once RAM cluster); 154 (benchmark has no SFX targets — so this cost is unmeasured; not a duplicate).

## Progress log

- 2026-09-30: re-verified every premise against live code — all hold as-read.
  `sfx_mmaudio.py:127` `_grab`, `:150` `capture_output`, `:156-158`
  frombuffer/copy/tensor chain, `:161-164` two spawns, `:170-187` third
  `_convert` spawn, `mmaudio_sfx.py:256-257` double `torch.stack`,
  `:265-266` `.to(device)` without `non_blocking`, `:293-294` `.to("cpu")`
  before `del`, worker `:223-230` fail-loud truncation instead of pad-to-17 tiling.
- 2026-09-30 (TDD red): the 045 pure-helper tests failed first in-container
  (missing `window_frame_counts`, `MAX_STACKED_BYTES`, `tile_to_length`,
  `MIN_SYNC_FRAMES`/`pad_to_sync_floor`, `sfx_single_pass_argv`, `derive_clip_indices`).
- 2026-09-30 (implement): worker `_extract_frames` is now ONE ffmpeg spawn
  at the top geometry (25 fps @ 384 px, `sfx_single_pass_argv`), streamed
  frame-by-frame into a SINGLE `torch.stack` — no `capture_output` hold, no
  `_grab` (deleted), no numpy-copy fan-out (`from_numpy` borrows each fresh
  chunk). CLIP derives by even temporal subsample (`derive_clip_indices`,
  already 384 px — zero resize); sync derives by CPU bilinear downscale
  384 to 224 + normalize. `render_window` enforces `check_stacked_bytes`
  pre-stack and pads sub-floor tails via `pad_to_sync_floor` (whole-batch
  tiling to a duration-consistent 0.64 s window whose take the caller
  trims); only zero-frame windows fail loud. `evict` drops the pointless
  `.to("cpu")` device-to-host copy (`del` + `gc.collect()` + `empty_cache`).
  Worker validates the byte budget before spawning. The
  `validate_duration_seconds` reject beyond `MAX_WINDOW_SECONDS` is unchanged.
- 2026-09-30 (TDD green): 9 pure-helper tests CPU-green in the slim image
  (no torch needed — helpers are stdlib-only by construction); the one
  torch/ffmpeg integration test skips (`importorskip`, runs behind
  `VOYAGE_REQUIRE_TORCH_TESTS=1`).
- 2026-09-30 (gates): ruff + format-check + mypy strict clean on both files;
  `test_sfx_contract` + all audio suites pass (`MAX_WINDOW_SECONDS = 60`
  untouched, so the duration-bounds test still passes).

## Resolution

- Verdict: fixed in scope (conditioning 2 spawns to 1; triple-hold to
  stream + one stack; fail-loud tails to pad-to-16 with loud-only-unservable).
  Public signatures unchanged (`_extract_frames` same return triple;
  `render_window`/`evict` same shapes — callers adapt to nothing).
- Files changed: `Voyage/voyage/workers/sfx_mmaudio.py` (+`SINGLE_PASS_FPS/SIZE`,
  +`sfx_single_pass_argv`, +`derive_clip_indices`, +`_read_frame_bytes`,
  rewrote `_extract_frames`, +`check_stacked_bytes` pre-spawn),
  `Voyage/voyage/audio/mmaudio_sfx.py` (+`MIN_SYNC_FRAMES`,
  +`MAX_STACKED_BYTES`, +`window_frame_counts`, +`effective_stacked_bytes` /
  `check_stacked_bytes`, +`tile_to_length`, +`pad_to_sync_floor`, rewrote the
  starvation branch, `evict` without the device-to-host copy),
  `Voyage/tests/test_media_memory.py` (new: exact-count math, byte-budget
  arithmetic + geometry-monkeypatch rejection, tiling order, floor-pad
  shape/order/duration, healthy passthrough, unservable rejection,
  single-pass argv shape, index spacing, torch-gated integration).
- Test evidence: `test_window_frame_counts_are_exact` (8 s gives 64 + 200),
  `test_effective_stacked_bytes_and_cap` (8 s arithmetic + 60 s within budget
  + 61 s duration-reject), `test_pad_to_sync_floor_tiles_short_tails`
  (12 to 16 sync, 5 clip, 0.64 s, order-preserving),
  `test_single_pass_argv_is_one_ffmpeg_at_top_geometry`.
- DESIGN.md as-built proposal (not applied): in the SFX/three-caption
  sections, add "conditioning is one ffmpeg pass at 25 fps @ 384 px per
  window (CLIP = temporal subsample, sync = CPU downscale to 224), streamed
  into a single stack; window byte budget (`MAX_STACKED_BYTES = 2 GiB`)
  enforced pre-spawn; sub-0.64 s tails tile to the 16-frame sync floor
  (caller trims the 0.64 s render) and only zero-frame windows fail loud;
  `evict` frees without a device-to-host copy."
- Residuals (needs-GPU-box): torch execution itself is unverified CPU-only —
  `test_single_pass_extract_shapes_on_testsrc` runs only with torch
  (`VOYAGE_REQUIRE_TORCH_TESTS=1`), asserting 16 clip + 50 sync frames at
  about 2.0 s resolved on a synthetic clip. `_convert` (FLAC to WAV third
  spawn) deliberately retained: it streams one small final file with no byte
  capture — constant memory, not part of the conditioning triple-hold.
  Padded windows render 0.64 s of audio for a shorter window (worker 0.6 s
  tolerance + finalize minimum 1 s windows confine this to tail slivers;
  the join trims).
