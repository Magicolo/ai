# 293 — LTXV tensor-handoff fallback catches everything, including OOM — real failures degrade with only an stderr line

Severity: LOW (pass-2 worker-tails sweep).

## Technical description

`_tail_clip_to_handoff_frames` wraps the GPU→CPU bottling (`permute().float().cpu().
numpy()`, `clip_array_to_uint8`) in `except Exception: ... return None`, falling back to
the chain-mp4 path with a `print(..., file=sys.stderr)`. That swallows
`torch.cuda.OutOfMemoryError`/`MemoryError` alongside the intended layout-drift cases,
converting a resource failure into a silent quality downgrade (lossy mp4 roundtrip on the
continuity-critical tail) that no metric records.

## Rationale

The comment says the fallback exists for "unexpected tensor API (e.g. stubbed sessions in
CPU tests)" — i.e., `AttributeError`/`TypeError`/`ValueError` territory. OOM is never a
"today's behavior" fallback; it should propagate to the worker's OOM→fp8 path.

## Live evidence (source, no GPU needed)

```
$ sed -n '174,198p' voyage/workers/video_ltxv.py
    try:
        tail_video = tail_clip[0]
        ...permute...cpu().numpy()...
        return tail_frames_for_conditioning(tail_clipped)
    except Exception as exc:  # noqa: BLE001 — fallback is today's mp4 path
        print(f"ltxv tensor handoff unavailable ({exc}); using chain mp4", file=sys.stderr)
        return None
```

Repro: stub a session whose `.cpu()` raises `MemoryError` (or run a genuinely pressured
GPU): next block conditions on the mp4 tail, continuity degrades, logs show one stderr
line, result payload reports nothing.

## Source refs

`voyage/workers/video_ltxv.py:174-198`; caller `:726-740`.

## Online sources

- ruff BLE001 ("blind except"); AGENTS.md §12 errors clause.

## Fix candidates

- Narrow to `(AttributeError, TypeError, ValueError)` (+ `ImportError` for the numpy
  path); let `MemoryError`/`torch.cuda.OutOfMemoryError` propagate. Optionally surface
  `handoff: "tensor"|"mp4"` in the block result for observability.

## Log

- 2026-10-07: filed from read-only pass-2 worker-tails sweep; no code touched.

## Evaluation (2026-10-07)
- Claim CURRENT on re-read: the `except Exception` still covered the whole
  GPU→CPU bottling, swallowing OOM into a silent mp4-tail downgrade. Fix
  candidate adopted as written.

## Progress log
- Batch-6 Group Q narrowed the handler in
  `voyage/workers/video_ltxv.py::_tail_clip_to_handoff_frames` to
  `(AttributeError, TypeError, ValueError)` — the layout-drift class the fallback
  exists for — so `MemoryError`/`torch.cuda.OutOfMemoryError` now propagate to
  the worker's OOM path instead of degrading silently. New
  `tests/test_issue_293_ltxv_handoff_errors.py` pins narrow-fallback vs OOM
  propagation. Scoped gates green (ruff + format + mypy strict + pytest).

## Resolution (2026-10-07)
- RESOLVED. Resource failures are failures again; the mp4 fallback only covers
  the tensor-API drift it was built for.
