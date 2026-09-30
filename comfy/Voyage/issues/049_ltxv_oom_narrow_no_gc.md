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

## Progress log

- 2026-09-30 — Re-read the issue fully; re-verified every premise live
  against the current tree (file:line as-read 2026-09-30; the tree drifted
  from the issue's cites — `481-499` is now `528-546`, `484` is now `531`,
  `658` is now `705`, `734-750` is now `819-835`).
  - Narrow catch HOLDS: `video_ltxv.py:528` is still
    `except torch.OutOfMemoryError:` with no string predicate; `rg` for
    `out of memory` in `video_ltxv.py` hits nothing but that line.
  - Missing `gc` HOLDS: line `531` is a bare
    `torch.cuda.empty_cache()` with no `gc.collect()` on the fallback path,
    while `evict()` (`815-816`) does have the `gc.collect()` +
    `empty_cache` order — same asymmetry the issue reports.
  - Sibling order VERIFIED: `video_longlive.py:855-860` and `903-906`
    document that without a collection `empty_cache` frees ~0 bytes
    (`tensors alive past del` comments) — the in-tree measurement the fix
    copies.
  - No shared predicate EXISTS: `video_common.py` (342 lines, read fully)
    has no `is_oom`; `acestep.py:73-86` carries a local `is_oom`
    (torch-free, class-name + substring — added by the 052 pass, which
    explicitly leaves the 049 share as a future import); `augment_worker.py`
    uses inline `"out of memory"` substring matches (`500`, `540`), both
    preceded by `gc.collect()` + guarded `empty_cache()`.
  - Out of scope CONFIRMED UNTOUCHED: `torch.cat` fan-out (`705`) and
    `_save_mp4` (`819-835`) belong to 048's fix — not touched here.
  - VERDICT: issue fully live, all three legs (narrow catch, gc ordering,
    memory logging) need the fix. Sharing via `video_common` rejected for
    this pass — it would touch other groups' code outside the owned files;
    a local `is_oom` mirrors `acestep.is_oom` with a docstring merge pointer.
- 2026-09-30 — TDD: wrote `tests/test_ltxv_oom_fallback.py` (6 tests, stub
  session + fake torch + stubbed `ltx_video.inference`) and watched it fail
  in-container (`./scripts/test.sh`): 3 failed (`is_oom` missing,
  RuntimeError-OOM escapes the narrow catch, `gc.collect` absent) and 3
  passed as behavior guards (torch-OOM retry, non-OOM propagation,
  armed-fallback re-raise). Test artifact caught during red phase: each
  `type("OutOfMemoryError", ...)` call makes a distinct class object, so
  torch-shaped failures must be built from the fake namespace's own
  `OutOfMemoryError` — the helper takes a `fail_factory(torch_ns)`.
- 2026-09-30 — Implemented in `voyage/workers/video_ltxv.py` only (+40):
  local `is_oom()`; `except (torch.OutOfMemoryError, RuntimeError)` with
  `is_oom` re-raise guard (non-OOM RuntimeErrors propagate, one fallback
  only); `gc.collect()` before `empty_cache()`; guarded `synchronize()` +
  `memory_allocated/reserved` GiB figures logged to stderr at fallback.
- 2026-09-30 — Gates: `ruff check` + `ruff format --check` + `mypy`
  (strict, project config) on both touched files clean (two nits fixed:
  `list.append() or value` lambdas → named defs, one unused ignore
  removed); 6/6 new tests green; related suites green — 62 passed
  (`test_ltxv_oom_fallback`, `test_ltxv`, `test_ltxv_tensor_handoff`,
  `test_ltxv_failure_hygiene`, `test_precision`, `test_video_common`).
  `git diff` confirms scope: only `video_ltxv.py` (+40) modified among
  tracked files; other tree modifications present are concurrent agents'
  (bench.py, media.py, rank2 tests) — untouched. Issues/*.md never
  ruff-formatted.

## Resolution

FIXED (all three legs, `voyage/workers/video_ltxv.py` +40, no other
tracked file touched):

1. Broad catch: `except (torch.OutOfMemoryError, RuntimeError)` + local
   `is_oom()` predicate (`type(failure).__name__ == "OutOfMemoryError"`
   OR `"out of memory" in str(failure).lower()` — the `augment_worker`
   idiom, mirroring `acestep.is_oom`). RuntimeError-shaped CUDA OOMs now
   reach the one-shot dynamic-fp8 retry instead of forcing a full session
   rebuild; non-OOM RuntimeErrors re-raise untouched (tested).
2. `gc.collect()` before `torch.cuda.empty_cache()` on the fallback path
   (the `video_longlive`-documented order), plus `synchronize()` where
   `torch.cuda.is_available()`.
3. Fallback logs `allocated=/reserved= GiB` (or `cuda unavailable`) with
   the OOM text to stderr, so the quant decision is visible in metrics.

Deliberately NOT done (owned-scope boundaries): `torch.cat` + `_save_mp4`
fan-out stays for 048's fix; no `video_common` share (would touch other
groups — merge pointer left in the `is_oom` docstring, cf. 052); no
`000_INDEX.md` / `DESIGN.md` / `AGENTS.md` edits (proposal below, text
only).
