# 047 — Augment worker reloads weights per call + stacks the full batch before OOM-halving

**Severity:** MEDIUM

**File:line:** `voyage/workers/augment_worker.py:141-150` (`_prepare_model`), `301-331` (loaders), `333-353` (`_run_stacked`), `356-394` (`upscale_frames`), `397-423` (`interpolate_pair`), `426-453` (`interpolate_triplet`)

**Description:**
Every `upscale_frames` / `interpolate_*` call rebuilds the net (`_build_rrdb_net` 23 RRDBs / `_build_film_net`) and `torch.load`s weights from disk with `strict=True`, then `.to(device)` + `.half()` (two placements: full fp32 move then in-place half — `_prepare_model` lines 146-149). No resident/cache path, so a 32-chunk augment pays 32× model construction + disk load + H2D. Worse, `upscale_frames` does `torch.stack(all frames).to(device, dtype)` (line 379) before `_run_stacked` halving — the all-at-once U4× tensor (the exact OOM the chunking was built to avoid: 32×768×512×3 fp16 → 4× ≈ 1.5 GiB + activations) is allocated first, then halved on failure. The halving only helps after the first OOM + `empty_cache`, without `gc.collect()` (line 346-347) — the codebase's own measured lesson (`del`/`empty_cache` without `gc` frees ~0 bytes, `video_longlive.py:827-833`).

**Rationale:**
Comfy's `FrameInterpolate` recipe halves the input list, not a pre-stacked tensor; stacking first defeats it. Per-call reload also hides slowness as "model time" in benchmarks with no load-vs-infer split (see 051).

**Live evidence (current tree):**
```
augment_worker.py:376: model = _load_rrdb_net(weights_path)   # per call
augment_worker.py:379: stacked = torch.stack([...]).to(torch_device, dtype=dtype)  # full batch first
augment_worker.py:384: for single in _run_stacked(model, stacked):  # halve only after OOM
augment_worker.py:346-347: if torch.cuda.is_available(): torch.cuda.empty_cache()  # no gc.collect()
augment_worker.py:146-149: model.to(torch_device); if cuda: model.half()
augment_worker.py:412,440: model = _load_film_net(weights_path)  # per interpolate call
```

**Repro (CPU, torch-free):**
```python
import inspect
from voyage.workers import augment_worker as a
assert "_load_rrdb_net" in inspect.getsource(a.upscale_frames)
assert "torch.stack" in inspect.getsource(a.upscale_frames)
```

**Fix candidates:**
- Session/resident-model cache keyed by `(weights path, device)` with `evict()`; load once per chunk-batch, not per call.
- Halve the list (`frames[i:j]`) and stack per half inside `_run_stacked`, or default to chunked singles with batched fast path.
- `model.half()` before `.to(device)` (or load fp16 directly), `gc.collect()` before `empty_cache` in the OOM branch, and record load-vs-infer ms separately.

**Refs:** PyTorch reference-cycle post (explicit `gc.collect()` necessity); diffusers VAE-tiling note (decode in patches, not whole-batch).

**Overlaps with:** 046 (augment chunk rescan — same path, complementary; not a duplicate).
