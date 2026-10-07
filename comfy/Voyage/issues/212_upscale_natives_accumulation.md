# 212 — Upscale direct path accumulates full-chunk 4x natives on GPU outside OOM-halving (HIGH)

## Technical description
`upscale_frames` direct (non-tiled) path appends every frame's native-x4
device tensor into `natives` and only post-processes after the loop
(`voyage/workers/augment_worker.py:1505-1522`):

```python
natives: list[Any] = []
for cpu_frame in cpu_frames:
    ...
    if tile_size is None:
        natives.extend(_run_frame_batches(model, [cpu_frame], torch_device, dtype))
    ...
for single in natives:
    results.append(_finish_upscaled(single, functional, target))
```

The tiled path finishes inline (`_finish_upscaled` immediately, comment
explicitly cites the 2060 OOM); the direct path does not.

## Rationale
1216×704 source → 4x native 4864×2816×3 fp16 ≈ 82 MB/frame. A 32-frame chunk
holds ≈ 2.6 GB of natives simultaneously, plus model + inputs + batch stack.
On the 6 GB 2060 this leaves ~3 GB headroom by itself, and
`_run_frame_batches`/`_run_stacked` halving cannot help because the
accumulation is outside the halving loop. Any larger geometry or concurrent
resident (llama sidecar ~3.2 GiB on the same card per `interp_pass_devices`
docs) tips it over.

## Live evidence
```
$ python3 -c "for w,h in [(1216,704),(2048,1152)]: print(w,h, round(w*4*h*4*3*2/2**30,2),'GiB per 32f chunk')"
1216x704 2.57 GiB per 32f chunk
2048x1152 7.05 GiB per 32f chunk
```

## Repro (CPU-only)
Stub `model` to return a recorded device tensor per call with 32 frames;
assert peak live tensors == 32 (direct) vs 1 (tiled). No GPU needed — count
allocations.

## Source refs
- `Voyage/voyage/workers/augment_worker.py:1505-1522`

## Online sources
- torch CUDA OOM/allocator docs (caching allocator; `empty_cache` doesn't
  create memory; fragmentation from live-held blocks).
- Real-ESRGAN tiling guidance (tile 256/400/512 to avoid OOM, overlap ≥48px).

## Fix candidates
1. Finish inline like the tiled path
   (`results.append(_finish_upscaled(single,...)); del single` per frame).
2. Or route all frames through the tiled postprocess.
3. Or cap the direct path to small frames and force tiling above budget
   (already computed per-frame — just extend to accumulation).

## Log
- Track D sweep, 2026-10-07. Read-only; nothing fixed.
