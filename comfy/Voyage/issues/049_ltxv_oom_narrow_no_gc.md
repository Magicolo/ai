# 049 — LTXV OOM fallback is narrow (`torch.OutOfMemoryError` only) + `empty_cache` without `gc`

**Severity:** MEDIUM

**File:line:** `voyage/workers/video_ltxv.py:407-416` (`_quantize_fp8_fallback`), `481-499` (fallback), `484` (`empty_cache`), `658` (`torch.cat`), `734-750` (`_save_mp4`)

**Description:**
`_generate_block` catches only `torch.OutOfMemoryError`. CUDA OOMs frequently surface as `RuntimeError: CUDA out of memory` (cublas/cudnn alloc sites, NCCL, `torch.cuda.OutOfMemoryError` alias gaps across torch 2.x builds) — those bypass the fp8 fallback and go straight to supervisor restart (full session rebuild, minutes lost). On the caught path, `torch.cuda.empty_cache()` runs without `gc.collect()` — the sibling (`video_longlive.py:874-879`) documents that this frees ~0 bytes under reference cycles. The retry also re-seeds the same generator (`generator.manual_seed(seed)`, line 486) but reuses already-moved `pos_embeds/pos_mask/conditioning` without re-validating freed state. Separately, `generate_blocks` does `torch.cat(novel_clips)` (658) + `_save_mp4` full-tensor `.cpu().numpy()` + `clip_array_to_uint8` copy (741-750) — same all-at-once fan-out as 048 at 121×768×512.

**Rationale:**
The fp8-fallback design is load-bearing for 16 GiB, so a missed string-match silently converts a 1-retry quant into a full-restart loop. The `gc`-before-`empty_cache` ordering is already measured in-tree.

**Live evidence (current tree):**
```python
video_ltxv.py:481: except torch.OutOfMemoryError:
video_ltxv.py:482-484:   if self._fp8_fallback: raise
video_ltxv.py:484:   torch.cuda.empty_cache()   # no gc.collect()
video_ltxv.py:486:   generator.manual_seed(seed)
video_ltxv.py:658: video = novel_clips[0] if len(novel_clips)==1 else torch.cat(novel_clips, dim=2)
video_ltxv.py:741-744: tail_numpy = ... / frames = video.permute(...).float().cpu().numpy()
```
`rg "out of memory" voyage/workers/video_ltxv.py` → only the `except` line; `augment_worker.py:344` matches on string `"out of memory"` instead — inconsistent taxonomy in the same tree. `video_ltxv.py:730-731` evict does have `gc.collect()` + `empty_cache` — the fallback path does not.

**Repro:**
```python
import torch
print(torch.OutOfMemoryError.__mro__[:3])
# vs: raise RuntimeError("CUDA out of memory. Tried to allocate ...")
# → second form escapes video_ltxv's except
```

**Fix candidates:**
- Catch `(torch.OutOfMemoryError, RuntimeError)` with an `is_oom(exc)` string predicate (`"out of memory" in str(exc).lower()`, the `augment_worker` idiom) shared in `video_common`.
- `gc.collect()` + `empty_cache()` + `torch.cuda.synchronize()` before `_quantize_fp8_fallback`, and log `memory_allocated/reserved` at fallback so the quant decision is visible in metrics.
- Chunk `_save_mp4`/concat path per 048.

**Refs:** PyTorch OOM-recovery FAQ (move recovery outside `except` — the wrapped exception pins frames); in-tree `augment_worker.py:343-352` string-match precedent.

**Overlaps with:** 048 (longlive copies — sibling worker-RAM defect); 052 (ACE OOM taxonomy — shared `is_oom` predicate would fix both; not duplicates).
