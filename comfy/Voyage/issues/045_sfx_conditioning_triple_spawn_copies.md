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
