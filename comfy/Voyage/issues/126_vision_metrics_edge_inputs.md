# 126 — Vision metrics edge inputs escape as raw exceptions or duplicate frames

- **Severity:** Low (inspector robustness — hostile/short clips crash or dilute the metrics the director trusts)
- **File:line:** `Voyage/voyage/vision/metrics.py:57-82` (`estimate_frame_total`: `round()` at `:81` outside the `try`, so `inf` → raw `OverflowError`, `nan` → raw `ValueError`); `:27-39` (`select_frame_indices`: duplicates when `count > total`, plus banker's-rounding asymmetry); `sample_frames` `:131-175` (no `width`/`count` sanity vs the decoded reality)
- **Area:** workers-internals tail — `voyage/vision/metrics.py` edge cases (pass 1 covered rotation-blindness (055) and fps divergence (029); these are input-validation edges underneath)

## Description

Three small boundary gaps, all verified live (evidence below):

1. **`estimate_frame_total` leaks raw numerics.** The `try` at `:75-80` covers parsing, but `round(duration * rate)` at `:81` sits outside it. A probe reporting `duration="inf"` (seen in the wild on broken moov atoms) raises `OverflowError: cannot convert float infinity to integer`; `"nan"` raises `ValueError: cannot convert float NaN to integer`. Neither is `MediaError`, so the inspector path (`sample_frames → estimate_frame_total`) crashes with an unclassified exception instead of taking the documented full-decode fallback (`:169-175`, which only triggers when the estimate returns `None`).
2. **`select_frame_indices` duplicates frames when `count > total`.** `select_frame_indices(2, 5)` → `[0, 0, 0, 1, 1]`; `(1, 3)` → `[0, 0, 0]`. `sample_frames` never clamps `count` to the decoded total, so a short clip yields a "3-frame" sample of 1-2 distinct frames. `motion_energy` over `[f, f, f]` is 0.0 and `visual_complexity` triple-counts one frame — the six §43 metrics silently misdescribe the segment. (Banker's `round` also skews spacing: `(4, 3)` → `[0, 2, 3]`, asymmetric.)
3. **`sample_frames(width=0)` → raw `ZeroDivisionError`.** `height` is guarded (`max(2, ...)` at `:155`) but `width` is not: `stride = 0` at `_decode_frames:119` → `divmod(len, 0)` raises `ZeroDivisionError: integer division or modulo by zero`, not `MediaError`. The width comes from a module constant today, but the function's contract promises `MediaError` for bad inputs and this input escapes it. (Negative widths fail later as ffmpeg `MediaError` — only zero takes the `divmod` path.)

## Rationale

The inspector is advisory-by-design (retry→skip), so none of these stop a voyage — but they corrupt or crash exactly the measurement the director's MEASURED context and style-similarity gate consume. An `OverflowError` from a damaged input file shouldn't need a new taxonomy entry; it should be `None` (→ full decode) or `MediaError`. And duplicated-frame metrics are worse than an error: they report confident zeros.

## Evidence (verified live 2026-09-30, `voyage:latest` + real ffmpeg)

```
select_frame_indices(2, 5) → [0, 0, 0, 1, 1]
select_frame_indices(1, 3) → [0, 0, 0]
select_frame_indices(4, 3) → [0, 2, 3]
estimate_frame_total({avg_frame_rate: 24/1, duration: 'inf'}) → OverflowError: cannot convert float infinity to integer
estimate_frame_total({..., duration: 'nan'}) → ValueError: cannot convert float NaN to integer
sample_frames(tiny.mp4, 1, width=0) → ZeroDivisionError: integer division or modulo by zero
sample_frames(tiny.mp4, 8) → 8 frames from a ~5-frame clip (duplicates confirmed end-to-end)
```

The last two used a real `testsrc` mp4 rendered with the image's ffmpeg.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app -e PYTHONPATH=/app/Voyage voyage:latest python3 -c "
from voyage.vision.metrics import select_frame_indices as s, estimate_frame_total as e
print(s(2, 5)); print(s(1, 3))
print(e({'avg_frame_rate':'24/1','duration':'inf'}, {}))  # OverflowError, should be None
"
```

## Fix candidates

1. Move `:81-82` inside the `try` (or pre-check `math.isfinite(duration * rate)` → `None`): non-finite probe data takes the full-decode fallback, matching the docstring ("Returns None when neither yields a positive count").
2. Clamp in `sample_frames`: `count = min(count, total)` after the total is known (both the estimate branch `:160` and the fallback `:174`); dedupe `picks` defensively in `select_frame_indices` or document the `count <= total` precondition with a `MediaError` when violated.
3. Validate `width >= 1` (and `count >= 1` — already done at `:144`) at the top of `sample_frames` → `MediaError`, closing the `divmod` hole.
4. Tests: inf/nan probe dicts → fallback path; `count > total` → distinct frames only; `width=0` → `MediaError`.

## Refs

 - `Voyage/voyage/vision/metrics.py:27-39,57-82,90-128,131-175`.
 - Adjacent, not overlapping: 055 (rotation-blind metrics — *what* is measured); 029 (rotation/fps divergence); 017 (validate_run hostile inputs — different function); 044 (decode copies — performance, not these branches).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. All three premises CONFIRMED as-read (`round()` outside the try, `select_frame_indices(2, 5) → [0, 0, 0, 1, 1]`, unguarded `width` into `divmod`). TDD: `tests/test_e2_vision_edges_126.py` written first — 6 failed pre-fix (inf/nan/inf-rate estimates raised raw `OverflowError`/`ValueError`, over-requests duplicated, `width=0` raised raw `ZeroDivisionError`), 8/8 green post-fix.

## Resolution

- Verdict: FIXED in `voyage/vision/metrics.py` (all three legs, minimal hunks).
- Changes: (1) `estimate_frame_total` computes `round(duration * rate)` INSIDE the try and catches `OverflowError` too — non-finite probe data returns None → full-decode fallback, matching the docstring; (2) `select_frame_indices` clamps `count` to `total_frames` (single home — both `sample_frames` branches inherit; documented in the docstring); (3) `sample_frames` validates `width >= 1` → `MediaError` before any probe/ffmpeg work. Banker's-rounding spacing asymmetry noted but deliberately untouched (no caller depends on it, out of this file's scope).
- Files changed: `voyage/vision/metrics.py` (+ new `tests/test_e2_vision_edges_126.py`). Verified the clamp changes none of the existing pins (`test_media_memory`, `test_perf_regressions` use count <= total).
- Test evidence (in-container `voyage:latest`, real ffmpeg): new file 8 passed (inf/nan/inf-rate → None, healthy estimates unchanged, over-requests serve distinct frames, `width=0` → MediaError without ffmpeg, 5-frame testsrc clip over-requested at 8 serves exactly 5 distinct frames); `test_vision_metrics` + `test_media_memory` + `test_perf_regressions` green (98 passed, 1 skipped). Ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — §§43-44 inspector): "Probe-derived frame estimates are best-effort: non-finite probe data yields no estimate (full-decode fallback), over-requested frame counts clamp to the decoded total (metrics never describe duplicated frames), and degenerate sample geometry fails as `MediaError`."
- Residuals: none in this file.
