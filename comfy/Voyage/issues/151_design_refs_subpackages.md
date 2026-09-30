# 151 — DESIGN refs missing across `workers/` + `audio/` subpackages (037's scan never looked there)

- Severity: LOW
- Files: `Voyage/voyage/workers/video_common.py:1-18`, `Voyage/voyage/workers/augment_worker.py:1-12`, `Voyage/voyage/workers/sfx.py:1-25`, `Voyage/voyage/workers/sfx_mmaudio.py:1-10,72`, `Voyage/voyage/workers/video_longlive.py:1-19` (+ body `:226,275,399,497,895,1196`), `Voyage/voyage/audio/planner.py:1`
- Area: standards / docs

## Description

AGENTS.md §12 requires every module docstring to state purpose + DESIGN §
ref. Issue 037 found 7 violations — but its sweep loop only scanned the
top level (`for f in voyage/*.py`), so the subpackages never got checked.
Six more modules fail the same rule: three with no `DESIGN` token anywhere
in the header (`video_common`, `augment_worker`, `planner`), two with bare
`§NN` tokens but no `DESIGN` prefix (`sfx`, `sfx_mmaudio` header), and one
(`video_longlive` header) with no `DESIGN` token up top while the body
carries six. Two of the six (`sfx_mmaudio:72`, `video_longlive` body) prove
the authors know the convention — the docstring just doesn't follow it.

## Rationale

037's rationale holds here unchanged: DESIGN refs are the traceability
mechanism (gates comments cite §83/§87/§92; worker bodies cite §5.3/§5.4/
§27/§40). Unref'd headers rot first — reviewers cannot tell which spec
section governs the module without reading the whole body. The subpackage
gap is systematic, not incidental: sibling modules in the same directories
(`workers/video.py`, `workers/loop.py`, `workers/audio.py`,
`workers/video_causvid.py:1`, `workers/video_ltxv.py:10`,
`workers/director.py:4`, `workers/audio_acestep.py:4`,
`audio/acestep.py:1`, `audio/beat.py:1`, `audio/mmaudio_sfx.py:1`,
`audio/__init__.py:1`) all carry `DESIGN` refs, so the six below are
outliers in otherwise-conforming packages. Fix cost is six one-line
docstring amendments; the risk of leaving them is the next sweep
re-filing the same class a third time.

## Live evidence (re-verified 2026-09-30, host reads)

`voyage/workers/video_common.py:1-18` — full header, no `DESIGN` token:

```python
"""Shared video-worker scaffolding (issue 019).

Extracted from `video_longlive.py` / `video_ltxv.py` / `video_causvid.py`,
...
Deliberate non-shares (documented, not migrated): ...
...
Top level is numpy + stdlib only ...
"""
```

`rg DESIGN voyage/workers/video_common.py` → zero hits in header (only
match in file, if any, is absent — the `(issue 019)` pointer is not a
DESIGN ref).

`voyage/workers/augment_worker.py:1-12` — no `DESIGN` token:

```python
"""GPU augment runner: Real-ESRGAN upscale + FILM interpolate (Track D spike).

`torch` loads only inside functions (behind a `find_spec` guard) — never
at module scope (supervisor section 12 GPU ban) — ...
...
"""
```

Says "supervisor section 12 GPU ban" — prose, not a `DESIGN §` ref.

`voyage/workers/sfx.py:1-25` — bare `§12`, no `DESIGN` prefix. Live `:1-8`:

```python
"""SFX worker: `python -m voyage.workers.sfx`.

Fake backend behind the same `generate_sfx` contract as the real
MMAudio worker (`sfx_mmaudio`): ...
...
the hard GPU ban (§12) holds trivially here.
"""
```

`rg DESIGN voyage/workers/sfx.py` → zero hits.

`voyage/workers/sfx_mmaudio.py:1-10` + `:72` — header fails, body passes.
Live `:1-12`:

```python
"""MMAudio SFX worker: `python -m voyage.workers.sfx_mmaudio`.
...
benchmark with VRAM peaks for the 2060 ladder (§104).

GPU ban (§12): `torch`/`mmaudio` only load inside functions ...
"""
```

Bare `(§104)` / `(§12)` — no `DESIGN` token in the docstring. But live
`:70-72` carries it in the body:

```python
        # first window (sequential residency, DESIGN §40).
```

`rg DESIGN voyage/workers/sfx_mmaudio.py` → `:72` only.

`voyage/workers/video_longlive.py:1-19` — header fails, body passes six
times. Live `:1-19`:

```python
"""LongLive 2.0 video worker: `python -m voyage.workers.video_longlive`.

Runs ONLY in the CUDA worker image ... Phase 1 scope: ...
16 GB VRAM design (RTX 4060 Ti): ...
Typing note: GPU-only imports carry `# type: ignore[import-not-found]` ...
"""
```

`rg DESIGN voyage/workers/video_longlive.py` → zero hits in `:1-19`,
then six body hits, all live-verified:

```
  Line 226:     16 GB VRAM sizing (RTX 4060 Ti, DESIGN §140): ...
  Line 275:         # Relative RoPE (DESIGN Phase 2 remainder): ...
  Line 399:     """Persistent causal stream (DESIGN §22): ...
  Line 497:         """Rebuild causal context after restart (DESIGN §27.1).
  Line 895:         # Recovery tail (DESIGN §27): ...
  Line 1196:     """Rebuild causal context from a recovery.pt tape (DESIGN §27.1)."""
```

`voyage/audio/planner.py:1` — bare `§§`, no `DESIGN` prefix. Live `:1`:

```python
"""Audio slow-loop planner (§35/§40): 30-60s music takes covering the video timeline.
```

`rg DESIGN voyage/audio/planner.py` → zero hits. Siblings conform —
`audio/__init__.py:1` `(DESIGN §37)`, `audio/acestep.py:1` `(DESIGN §37)`,
`audio/beat.py:1` `(DESIGN §35)`, `audio/mmaudio_sfx.py:1`
`(SFX slice 2, DESIGN three-caption doctrine)` — so `planner.py` is the
one outlier in its package.

Sweep that finds them (037's loop, extended to subpackages):

```bash
for f in voyage/*.py voyage/workers/*.py voyage/audio/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done
```

Live output includes exactly the six files above (plus 037's original
seven at top level — no overlap, no double-count).

## Repro

```bash
head -n 18 Voyage/voyage/workers/video_common.py
head -n 12 Voyage/voyage/workers/augment_worker.py
head -n 25 Voyage/voyage/workers/sfx.py
head -n 10 Voyage/voyage/workers/sfx_mmaudio.py
sed -n '1,19p' Voyage/voyage/workers/video_longlive.py
head -n 1 Voyage/voyage/audio/planner.py
rg -n "DESIGN" Voyage/voyage/workers/video_common.py Voyage/voyage/workers/augment_worker.py Voyage/voyage/workers/sfx.py Voyage/voyage/workers/sfx_mmaudio.py Voyage/voyage/workers/video_longlive.py Voyage/voyage/audio/planner.py
for f in voyage/*.py voyage/workers/*.py voyage/audio/*.py; do head -n 15 "$f" | grep -q DESIGN || echo "NO-DESIGN-REF: $f"; done
```

## Fix candidates

1. One-line docstring amendments: `video_common` → DESIGN §19 (its own
   header cites issue 019; confirm section); `augment_worker` → DESIGN
   §5.x/augment section; `sfx` + `sfx_mmaudio` header → DESIGN §40
   (three-caption/finalize-time SFX, matching `:72`); `video_longlive`
   header → DESIGN §§22/27 (+ §140 for the VRAM note); `planner` →
   `DESIGN §§35/40` (add the prefix, keep the numbers).
2. Extend 037's sweep to the subpackages (the loop above) and re-run after
   the fix — expect zero `NO-DESIGN-REF` lines outside 037's tracked seven.
3. Optional: normalize bare-`§NN` style (`sfx`, `sfx_mmaudio` header,
   `planner`) to the `DESIGN §NN` form in one pass so `grep -q DESIGN`
   stays a valid gate.

## Refs

- `Voyage/voyage/workers/video_common.py:1-18`
- `Voyage/voyage/workers/augment_worker.py:1-12`
- `Voyage/voyage/workers/sfx.py:1-25`
- `Voyage/voyage/workers/sfx_mmaudio.py:1-10,72`
- `Voyage/voyage/workers/video_longlive.py:1-19,226,275,399,497,895,1196`
- `Voyage/voyage/audio/planner.py:1`
- Issue 037 (top-level DESIGN-refs sweep — same rule, narrower glob)
- AGENTS.md §12 ("every module docstring states purpose + DESIGN § ref")
- Preserved track result: `ses_f0fbea3bbffe0HlTFo2fwHJi5B`, TRACK C NEW FINDING 2.
