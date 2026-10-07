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
