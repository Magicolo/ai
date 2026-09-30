# 120 — `beats_per_segment` has no upper bound: plausible values produce grid BPMs the ACE worker rejects, failing takes fatally

- **Severity:** Medium (cross-layer contract break — a documented user knob reliably kills music takes at render time)
- **File:line:** `Voyage/voyage/config.py:312-317` (`positive_beats`: `>0` only); `Voyage/voyage/audio/beat.py:36-58` (`beats_for_segment`: doubling floor, no ceiling); `Voyage/voyage/supervisor.py:1179-1180,1206` (raw `take_bpm` into the ACE payload); `Voyage/voyage/audio/acestep.py:35-43` (`validate_bpm`: `1..300`, ValueError → INVALID_PAYLOAD fatal)
- **Area:** workers-internals tail — `voyage/audio/beat.py` BPM math + its config/worker contract (pass 1 covered NaN/inf *inputs* (100, config→RPC); this is finite, in-range-at-every-layer inputs that still break the downstream cap)

## Description

The beat grid only floors tempo (`while bpm < min_bpm: beats *= 2`) and never ceilings it, while the ACE worker hard-rejects `bpm > 300`. Grid BPM is `beats × 60 / segment_seconds`, so short segments × fine grids exceed the cap with ordinary settings:

- `longlive2` preset: `segment_frames=29 @ 24fps` (`config.py:150-155`) = **1.208 s/segment**. Default `beats_per_segment=4` → 199 BPM (fits). But `beats_per_segment=8` — the natural "finer grid" choice the TOML comment *invites* ("each committed segment spans `beats_per_segment` beats") → **397 BPM** → `validate_bpm` raises `ValueError` → worker loop maps to INVALID_PAYLOAD (fatal, no restart) → the take render dies and the segment with it.
- No layer bridges the gap: config accepts any `beats_per_segment > 0`; `beats_for_segment` returns any finite BPM; the supervisor passes `int(round(grid_bpm))` straight into `"bpm": take_bpm` with no clamp; the worker rejects. The failure surfaces minutes into a GPU run (post video-evict, ACE resident) as a fatal payload error for what is really a config error.
- Secondary: `min_bpm <= 0` is silently accepted (`beats_for_segment(4.0, min_bpm=0)` → `(4, 60.0)` — the doubling loop is dead code, the "minimum tempo" contract vacuous); and supervisor `:1819` reports raw float `grid_bpm` in progress events while `:1180` rounds for the render payload, so displayed BPM and rendered BPM can differ by a rounding step.

## Rationale

 Tempo is the one audio parameter that crosses three layers (config knob → beat math → ACE validator), and each layer validated only its own half: positivity at config, finiteness in beat math, range at ACE. The composition was never checked, so the first user to tune the grid on the noisiest (shortest-segment) backend gets a fatal mid-run crash instead of a config-time error. Failing at `init`/config-load is one line; failing after a 12 GB ACE load + video evict is a wasted GPU cycle plus a confusing error (`bpm must be None or 1..300 (got 397)` names the symptom, not the knob).

## Evidence (verified live 2026-09-30, host stdlib + tree reads)

```
longlive2 29f@24 beats=4: (4, 198.6206896551724)   → take_bpm 199 ✓
longlive2 29f@24 beats=8: (8, 397.2413793103448)   → take_bpm 397 ✗ (> 300 cap)
min_bpm=0  -> (4, 60.0)    # degenerate floor accepted
min_bpm=-5 -> (4, 60.0)    # degenerate floor accepted
```

- `config.py:312-317`: `if value <= 0: raise` — no upper bound, no BPM-compatibility check.
- `beat.py:55-57`: doubling loop only; no `while bpm > max_bpm: beats //= 2` counterpart.
- `supervisor.py:1180/1206`: `take_bpm = int(round(grid_bpm))` → `"bpm": take_bpm`, unclamped.
- `acestep.py:40-43`: `if bpm < 1 or bpm > 300: raise ValueError`.

## Repro

1. `beats_for_segment(29/24, 8)` → `(8, 397.24)` (any Python with the tree on `PYTHONPATH`; no GPU).
2. End-to-end: set `beats_per_segment = 8` in `voyage.toml` with `video.backend = "longlive2"`, `audio.backend = "acestep"` → first take render fails `INVALID_PAYLOAD: bpm must be None or 1..300 (got 397)`.

## Fix candidates

1. Cap the grid: in `beats_for_segment`, halve (or refuse) when the implied BPM exceeds the renderer's ceiling — needs the ACE `MAX_BPM=300` visible to beat math (import, don't duplicate the constant).
2. Or validate at config load: reject `beats_per_segment` whose implied BPM for the configured backend geometry exceeds 300 (geometry-aware check in `AudioConfig`/`ProjectConfig` model validator — fails before any GPU work).
3. At minimum, clamp + warn at the supervisor call site (`:1180`) so an out-of-range grid degrades to the ACE ceiling instead of a fatal.
4. Require `min_bpm > 0` in `beats_for_segment` (kill the degenerate floor); unify `:1180` round vs `:1819` raw-float reporting.

## Refs

- `Voyage/voyage/audio/beat.py:20-58`; `Voyage/voyage/config.py:150-155` (29-frame preset), `:312-317`; `Voyage/voyage/supervisor.py:1174-1180,1195-1207,1819`; `Voyage/voyage/audio/acestep.py:35-43`.
- Adjacent, not overlapping: 100 (NaN/inf *config* values into RPC timeouts — different layer, non-finite inputs); 013-era `take_seconds >> ahead_seconds` invariant (same file family, duration axis not tempo axis).

## Progress log

- 2026-09-30 (Group B): re-verified live first (host stdlib + tree reads):
  `beats_for_segment(29/24, 8)` → `(8, 397.24)` > ACE 300 cap; `min_bpm=0`
  accepted with a dead doubling loop; supervisor passes `take_bpm` raw into
  the ACE payload. Premise confirmed.
- Constraint found live: `tests/test_rhythm.py:34-40` deliberately pins
  the raw doubling math (`(0.5, 4) -> (4, 480)`, "degenerate short: math
  holds") — a default ceiling would undo that characterization, so the
  ceiling is opt-in and the three production call sites opt in explicitly.
- TDD: wrote `tests/test_beats_bpm_cap_120.py` first — 5 failed / 2 passed
  (default-no-ceiling + fitting-grid green) before the fix.
- Fix: `beats_for_segment(..., max_bpm=None)` — when set, halve toward one
  beat (cuts stay on an integer grid, only coarser); when even one beat
  exceeds, `ValueError` naming `beats_per_segment` + segment seconds +
  ceiling (fails before GPU work via the 002 backstop, actionable message).
  `min_bpm <= 0` now rejected (dead-floor guard); non-finite/non-positive
  `max_bpm` rejected. Supervisor's three call sites (take render, console
  plan, progress display) pass `max_bpm=MAX_BPM` (imported from
  `voyage.audio.acestep`, not duplicated — stdlib-only module, no cycle).
  The plan-display and render calls now share identical inputs, so displayed
  BPM and rendered BPM agree by construction (secondary nit closed).
- Gates (in-container): new tests + `test_rhythm` + `test_beat_properties`
  + `test_audio_planner` = 50 passed; `ruff check` + `ruff format --check`
  + `mypy` on `voyage/audio/beat.py` + `voyage/supervisor.py` clean.
  Supervisor diff re-checked minimal (3 disjoint hunks only, 15+/4-).

## Resolution

- Verdict: FIXED on the production path. Files changed:
  `voyage/audio/beat.py` (opt-in `max_bpm` + `min_bpm > 0`),
  `voyage/supervisor.py` (3 call sites opt into the ACE ceiling),
  `tests/test_beats_bpm_cap_120.py` (new: no-ceiling default, halve,
  passthrough, impossible-grid error, degenerate/non-finite guards).
- DESIGN proposal (quoted text only, for the DESIGN owner): in §35, after
  the adaptive-k description, add: "The beat grid honors the renderer's
  tempo ceiling: the supervisor passes the ACE `MAX_BPM` into the grid, so
  an over-fine `beats_per_segment` degrades to a coarser integer grid and
  an impossible one fails the commit naming `beats_per_segment` — before
  any GPU work, never as a fatal payload error after the ACE load."
- Residuals (other files, not touched per scope): `config.py:362-367`
  `positive_beats` still accepts any `beats_per_segment > 0` with no
  BPM-compatibility check — a geometry-aware validator in `AudioConfig` /
  `ProjectConfig` (candidate 2) would fail at `init` instead of at first
  commit.
