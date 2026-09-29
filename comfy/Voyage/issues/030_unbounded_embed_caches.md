# 030 — Unbounded GPU-resident embed caches (+ LTXV mask pinned to `"cuda"`)

- Status: resolved (fixed 2026-09-25: LRU-8 + CPU-side + device + tests)
- Severity: medium-high (slow VRAM leak over long voyages; breaks `cuda:1`)
- Area: performance/correctness — worker embed caches
- Rank rationale: every distinct prompt pins CUDA tensors forever; drift-every-N
  generates unbounded distinct prompts.

## Technical description

- `voyage/workers/video_ltxv.py:301-321` (`result = (embeds, mask.to("cuda"))`,
  cached dict never bounded; `.to("cuda")` hardcoded at line 319, ignoring
  `self._device` — breaks `cuda:1` configs, prevents CPU-side caching).
- `voyage/workers/video_longlive.py:353,479-487` (`_embed_cache: dict` never
  bounded, stores CUDA tensors).
- `voyage/workers/video_causvid.py:569-590` (per-`_infer` `_StubEncoder` class
  definition allocates a new `nn.Module` subclass per rollout; holds `precomputed`
  refs past the call if a traceback is cached).

## Why this is an issue

Every distinct prompt pins CUDA tensors forever, and drift-every-N guarantees a steady stream of distinct prompts — so long voyages leak VRAM monotonically until the card OOMs mid-run, the worst possible failure mode for an autonomous system. The hardcoded `"cuda"` additionally breaks any non-`cuda:0` device config, silently closing off multi-GPU setups. Both are slow-burn faults: invisible in short tests, fatal at production lengths.

## Evidence

`rg -n '_embed_cache|to\("cuda"\)|class _StubEncoder'
Voyage/voyage/workers/video_*.py`.

Verified live 2026-09-25:

```
video_ltxv.py:295: self._embed_cache ... 319: inputs.attention_mask.to("cuda")
video_ltxv.py:320: self._embed_cache[text] = result  605: self._embed_cache.clear()
video_longlive.py:353: self._embed_cache ... 480/486: get/store (never bounded)
video_causvid.py:543: pipeline.text_encoder.to("cuda")
video_causvid.py:569: class _StubEncoder(_module_base):  # defined per _infer call
```

## Reproduction

Long voyage with drift-every-N (distinct prompts); watch resident embed memory
grow monotonically; configure `cuda:1` and watch LTXV fail the mask placement.

## Source references

- Files/lines above.

## Resolution candidates

1. LRU (e.g. 8 entries) + store embeds on CPU, `.to(device)` on use.
2. Replace hardcoded `"cuda"` with `self._device`.
3. Hoist `_StubEncoder` to module level.

Payoff: flat VRAM over 100s of segments; unblocks non-`cuda:0` devices.

## Investigation / progress / resolution log

- 2026-09-25: found by perf sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; embed-cache /
  hardcoded-cuda / _StubEncoder refs re-verified live, current; pasted rg output
  into Evidence.
- Open: implement LRU + device fix.
- 2026-09-26 (resolution, FIXED): shared `EmbedCache` LRU
  (`EMBED_CACHE_CAPACITY = 8`,
  `voyage/workers/video_common.py:41-95`) plus torch-free
  `move_to_cpu` / `move_to_device` (`:98-132`, duck-typed recursion
  over dict/list/tuple, plain values pass through). Longlive
  (`video_longlive.py:566-581`) and LTXV (`video_ltxv.py:351-380`)
  store entries CPU-side and move to the session device on every use
  — including hits, so a cached mask never stays pinned to a stale
  device. Relevance check: the hardcoded `"cuda"` sub-item was
  already fixed (074 now reads `self._device` in LTXV) — adopted, not
  duplicated. CausVid `_StubEncoder` hoisted to a module-level
  cached-subclass factory (`_stub_encoder_class`,
  `video_causvid.py:389-421`; one class per torch module, Module
  subclassing preserved) used by `_infer` (`:630`).
  FOLLOW-UP FOR THE OWNING TRACK (not this scope):
  `tests/test_ltxv_failure_hygiene.py` pins the old private type
  (`_embed_cache={}` dict double, `:299`) — all 4 tests fail with
  `AttributeError: 'dict' object has no attribute 'put'` at
  `video_ltxv.py:377`. The LRU replacement inherently breaks that
  double; the fix is `_embed_cache={}` → `EmbedCache()` in that
  file. No production change needed.
  Tests: `tests/test_issue_030_embed_bounds.py` (11 tests: capacity,
  eviction, recency, overwrite/clear, move-helper recursion/shape
  preservation/copy semantics, LTXV CPU-side store + per-use device
  move + eviction, longlive CPU store + hit-without-reencode).
  Gates: ruff + format + mypy strict clean on all touched files;
  new tests pass in-container (see 014 log for full-tree gate state).
