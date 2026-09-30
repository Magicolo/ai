# 152 — SFX bed joins with a left-fold ffmpeg re-encode per window (O(N²) reads, accum re-encoded N−1 times)

- **Severity:** MEDIUM (perf + generational cost on exactly the long runs the SFX pass exists for)
- **File:line:** `Voyage/voyage/sfx_finalize.py:390-394` (left-fold `accum → _blend_pair → step`) → `Voyage/voyage/media.py:449-496` (`_blend_pair` manual-fade graph), esp. `:464-465` (durations + `fade = min(overlap, …/2, …/2)`), `:269` (same probe-per-blend pattern in `assemble_segment_audio`), `:624` (`probed_take_seconds` clamp precedent) and `:643-647` (identical left-fold in `build_final_audio`)
- **Area:** SFX finalize bed assembly (below both previous windows; 050 covers the video double-encode, 043 covers whole-MP4 RAM, 095 covers fade-absorption compensation — none covers the SFX left-fold re-encode)

## Description

`render_sfx_bed` joins stems pairwise left-to-right:

```python
# sfx_finalize.py:390-394
accum = stems[0]
for index in range(1, len(stems)):
    step = tmpdir / f"sfx_blend_{index:02d}.wav"
    _blend_pair(accum, stems[index], step, SFX_WINDOW_OVERLAP)
    accum = step
```

Each `_blend_pair` (`media.py:449-496`) probes both inputs (`:464-465`), builds a 2-input afade/adelay/amix graph, and writes a fresh `pcm_s32le` intermediate. Window 0 is therefore decoded/re-encoded N−1 times on an N-window timeline (8 s windows, 7 s step: a 10-minute final ≈ 85 windows ≈ 84 blends, the accum growing to ~600 s by the last blend). Intermediates stay s32le so there is no 16-bit generational loss (the docstring's explicit claim at `:460-462`), but every blend still pays a full ffmpeg spawn + full-accum read/write: total I/O is O(N²) in timeline length, and a failure at blend k discards k−1 good intermediates (no resume — the bed is rebuilt from scratch on retry). The music path (`build_final_audio`, `:643-647`) shares the identical fold, so a music+SFX finalize pays the quadratic twice.

## Rationale

The SFX pass is finalize-time and timeline-proportional: short demos never notice, long runs (the ones that need chunking elsewhere — 046/048) pay superlinearly. The single-stem fast path (`len(stems) == 1`, `:373-389`) already proves the codebase knows how to skip the fold when it is unnecessary; the multi-stem path has no equivalent short-circuit (e.g. concat when overlap is 0, single ffmpeg filter chain, or streaming concat). A benchmark/soak trend on finalize wall-clock will attribute the cost to "ffmpeg" rather than the fold shape without this file.

## Live evidence

- `sed -n '390,394p' voyage/sfx_finalize.py` — the accum loop; `len(stems) == 1` returns early at `:373-389`, everything else folds.
- `sed -n '449,496p' voyage/media.py` — `_blend_pair` probes both inputs (`:464-465`), `afade out + afade in + adelay + amix` at `:471-475`, writes `pcm_s32le` at `:489-491`; no streaming/batch variant exists.
- `sed -n '643,647p' voyage/media.py` — `build_final_audio` repeats the same `accum/windows[index]` fold, so the two finalize audio stages stack.
- `rg -n "_blend_pair|blend_" voyage/sfx_finalize.py voyage/media.py` → only the two left-fold call sites plus the `fade <= 0.1` concat fallback at `media.py:277-287` (SFX never takes it — `_blend_pair` raises at `:467-468` on non-positive overlap instead).
- Overlap check: 050 is the video parts+final double-encode (different module, different codec); 043 is whole-file RAM; 095 is fade-absorption length compensation (correctness of the same blends, not their count). None names the N−1 re-encode fold.

## Repro

Static (deterministic): count ffmpeg spawns for an N-window bed — `render_sfx_bed` with `len(stems) == N` runs exactly N−1 `_blend_pair` spawns plus the final bed convert (`:395-407`), each reading the full accum so far. Instrument with `strace -f -e execve` or log each `_blend_pair` call: wall time grows ~quadratically with N (double the timeline → ~4× the blend I/O). A 2-window bed runs 1 blend; an 85-window bed runs 84, the last reading ~600 s of accum to append 8 s.

## Fix candidates

1. Single-graph join: build one ffmpeg filter chain (N inputs, chained adelay+amix, or the `fade < 0.1` concat precedent at `media.py:277-287` generalized) so each stem is read once — O(N) I/O, one spawn.
2. Keep the fold but checkpoint: reuse the per-index `sfx_blend_*.wav` intermediates across retries (skip blends whose inputs are unchanged), mirroring the stem ledger cache (`:296-329`).
3. Emit per-blend wall seconds into `metrics.jsonl` (like the `stage_ms` 050 asks for) so soak can trend fold cost vs timeline length before restructuring.
4. Test: 3-window bed asserts byte-identical output between fold and single-graph paths; N-window test pins spawn count ≤ 2.

## Refs

- `Voyage/voyage/sfx_finalize.py:372-413`; `Voyage/voyage/media.py:249-312,441-496,643-647`; DESIGN §56 (finalize), §40 (SFX pass).
- Adjacent, not overlapping: 050 (video double-encode — different stage); 043 (whole-MP4 publish RAM); 095 (fade absorption — same blends, length correctness not count); 031 (slice memo — take slicing, not bed joins).
