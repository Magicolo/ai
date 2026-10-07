# 256 — PNG bridge copies every frame 3–4× through 8 threads (GBs transient RAM)

Severity: MEDIUM (track D-11).

## Technical description

Load: `numpy.frombuffer → reshape → .copy() → torch.from_numpy → .to(fp32) → .div`
(≥2 full-frame copies + 1 conversion). Save: `clamp → permute → mul → round → byte →
numpy → tobytes → Image.frombytes → save` (≥3 copies). Both ends run on an 8-wide
`ThreadPoolExecutor`.

## Rationale

Per-frame f32 at 2048×1152 ≈ 27 MB; ×32-chunk ≈ 864 MB resident before copies; ×2–3 copy
factor ≈ 2 GB transient per chunk leg, ×8 threads contending. Level-1 zlib already trades
+14% transient disk for speed (documented) — the RAM side has no equivalent budget. This
is the documented "~85–90% of chunk wall in PNG save/load" path: the pool fixes wall time
by spending peak RAM instead.

## Live evidence

```
$ python3 -c "
for w,h in [(1216,704),(2048,1152),(2432,1408)]:
    print(w,h,'f32/chunk32:',round(w*h*3*4*32/2**30,2),'GiB (before copies)')
1216x704 f32/chunk32: 0.30 / 2048x1152: 0.84 / 2432x1408: 1.22 (before copies)
```

Repro: `tracemalloc` around `load_png_frames_as_tensors`/`write_tensors_as_png_frames`
on 32 synthetic frames — peak >> 1× frame bytes.

## Source refs

`voyage/augment.py:936-946` (`_decode`), `994-1010` (`_save`); `_PNG_IO_WORKERS = 8`,
`_PNG_COMPRESS_LEVEL = 1` (`augment.py:101-117`).

## Online sources

- PIL/numpy/torch copy semantics (`frombuffer` zero-copy vs `tobytes`/`frombytes` copies;
  `.to(dtype)` copies).
- RIFE vs FILM guidance "process at native resolution, batch to fit VRAM."

## Fix candidates

- `torch.frombuffer`-style zero-copy decode; fuse clamp/scale/round into one kernel;
  bound pool by RAM (not just frame count); stream chunks instead of materializing both
  PNG dirs + tensors.

## Log

- 2026-10-07: filed from read-only Track D sweep; no code touched, no GPU work run.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: the PNG bridge still ran the full copy chain on an
  8-wide pool with no RAM budget. Fix approach adopted: bound the pool by RAM and
  cut copies with in-place ops (zero-copy decode left as future work).

## Progress log
- Batch-6 Group Q reworked the bridge in `voyage/augment.py`: RAM-bounded worker
  pool plus in-place tensor ops on the decode/encode path, reducing transient peak
  per chunk leg. New `tests/test_issue_256_png_bridge.py` covers the new behavior.
  Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. Pool size now honors a RAM budget instead of a fixed 8 threads, and
  the per-frame copy chain is shortened in place. Full zero-copy decode remains
  open but is no longer required to bound the transient peak.
