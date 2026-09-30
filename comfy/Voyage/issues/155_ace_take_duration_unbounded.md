# 155 — ACE take duration has no upper bound while the SFX sibling caps at 60 s (unbounded GPU/disk before any side effect)

- **Severity:** MEDIUM (validation gap — a single absurd `duration_seconds` passes every gate and dies deep in GPU/ffmpeg)
- **File:line:** `Voyage/voyage/audio/acestep.py:46-54` (`validate_duration_seconds`: finite + `> 0` only) vs `Voyage/voyage/audio/mmaudio_sfx.py:60-62` (`MAX_WINDOW_SECONDS = 60.0` ceiling) → `Voyage/voyage/config.py:305-310` (`take_seconds`/`ahead_seconds`/`crossfade_seconds` non-negative, no max) + `:340-357` (`take_covers_ahead`: only the `take > ahead` floor) → `Voyage/voyage/workers/audio_acestep.py:194-214` (`handle_benchmark` probe `duration_seconds` default 15.0, unbounded override)
- **Area:** audio duration validation (below both previous windows; 063-class validators are the precedent — this file is the missing upper half)

## Description

The ACE path rejects only the bottom: `validate_duration_seconds` (`acestep.py:46-54`) fails zero/negative/NaN/inf, and the docstring explicitly blesses "positive-but-short" through to the 1.0 s upstream floor. There is no top: `duration_seconds = 3600.0` (or `1e9`, still finite) passes `acestep.validate_duration_seconds`, passes `AudioConfig.non_negative` (`config.py:305-310`, `value < 0` only), passes `take_covers_ahead` (`:340-357`, only checks `take > ahead`), and reaches `render_take` → ACE DiT alloc + ffmpeg convert before anything complains — on a 16 GiB card the failure mode is a multi-minute swap/OOM deep in the music stack, not a parse-adjacent `ValueError`. The SFX sibling already solved this: `mmaudio_sfx.py:60-62` caps windows at `MAX_WINDOW_SECONDS = 60.0` ("windows are 8 s native; the bound only rejects caller bugs … before any GPU side effect"), and both workers' `handle_benchmark` paths accept arbitrary `duration_seconds` overrides (`audio_acestep.py:207-214` builds the probe from `payload.get("duration_seconds", 15.0)` with no cap; `sfx_mmaudio.py:269-270` at least inherits the 60 s cap via `validate_duration_seconds` at `:270`). A benchmark caller can therefore request a 2-hour ACE "take benchmark" that the SFX benchmark would reject.

## Rationale

Bounds belong at the boundary before GPU side effects (§12 worker discipline; issue 063 established the bottom half). Takes are 30-60 s by design (`AudioPlanner` slow loop, beat-grid quantization); anything orders of magnitude above that is a caller bug (misplaced milliseconds, swapped args, benchmark typo), and failing it in 1 ms of validation instead of 10 minutes of DiT render is the entire point of `validate_*`. The asymmetry is the tell: two sibling validators for the same physical quantity (audio seconds) disagree on whether "absurd" exists.

## Live evidence

- `sed -n '46,54p' voyage/audio/acestep.py` — `if not math.isfinite(…) or … <= 0.0` only; compare `sed -n '60,62p;76,86p' voyage/audio/mmaudio_sfx.py` — `MAX_WINDOW_SECONDS = 60.0` + three-way check (`not finite or <= 0 or > MAX`).
- `sed -n '305,310p;340,357p' voyage/config.py` — `non_negative` allows `+inf`-minus-NaN (finite check present) but no max; `take_covers_ahead` only floors `take > ahead`.
- `sed -n '194,214p' voyage/workers/audio_acestep.py` — benchmark probe takes `float(payload.get("duration_seconds", 15.0))` unbounded; the subsequent `validate_duration_seconds` at `:152` (via `handle_generate_audio`) still has no ceiling.
- `python3 -c` mental model (no host torch needed): `validate_duration_seconds(3600.0)` returns `None` on the ACE path, raises on the SFX path — same input, opposite verdicts.
- Overlap check: 063-class work built the bottom validators (`validate_bpm/task_type/reference_audio` in the same files); 100 covers NaN/inf into `select`; 104 covers unbounded slice loops (ffmpeg-spawn count, not take length). None caps ACE duration.

## Repro

CPU-only (no GPU needed): `from voyage.audio.acestep import validate_duration_seconds; validate_duration_seconds(3600.0)` → returns silently; `from voyage.audio.mmaudio_sfx import validate_duration_seconds as sfx_v; sfx_v(3600.0)` → `ValueError`. Config-level: `AudioConfig(take_seconds=1e6, ahead_seconds=20.0, …)` validates (only the `take > ahead` invariant fires). Worker-level: `handle_benchmark({"duration_seconds": 7200.0, …})` on the ACE worker passes validation and starts a 2-hour render.

## Fix candidates

1. Mirror the sibling: `MAX_TAKE_SECONDS` in `audio/acestep.py` (e.g. 120.0 — 2× the largest legitimate 60 s take, well under the SFX 60 s window cap × takes-per-render reality) enforced in `validate_duration_seconds`, with the same "rejects caller bugs before GPU" comment.
2. Cap `AudioConfig.take_seconds` (and `ahead_seconds`/`crossfade_seconds` by the same token) with a `field_validator` upper bound, so TOML-level typos fail at `load_config`, not at render.
3. Cap benchmark overrides: `handle_benchmark` clamps/probe-validates `duration_seconds` against the same constant (both audio workers), so a benchmark typo cannot become a multi-hour GPU hold.
4. Tests: `3600 s` raises on both audio validators; boundary test pins `MAX == accepted, MAX + epsilon == rejected`; config test pins `take_seconds=1e6` → `ValidationError`.

## Refs

 - `Voyage/voyage/audio/acestep.py:31-60,126-155`; `Voyage/voyage/audio/mmaudio_sfx.py:60-86`; `Voyage/voyage/config.py:290-357`; `Voyage/voyage/workers/audio_acestep.py:134-214`; DESIGN §37 (ACE), three-caption doctrine (SFX windows).
 - Adjacent, not overlapping: 063-class validators (bottom half — this file is the missing top); 100 (NaN/inf into select); 104 (slice-loop spawn storm); 013 (take/ahead swap-per-segment floor).

## Progress log

- 2026-09-30 (Group E2): evaluated live first. Premise CONFIRMED as-read: ACE `validate_duration_seconds` rejected only bottom (finite + > 0) while the SFX sibling caps at 60 s; `AudioConfig.non_negative` (`config.py:348-353`) has no max; both workers' benchmark probes take unbounded overrides. No test in the tree used ≥120 s durations (grep over acestep/planner/worker/test files — empty), so a 120 s ceiling (2× the largest legitimate 60 s take, per fix candidate 1) has no false-positive surface. TDD: `tests/test_e2_ace_ceiling_155.py` — collection failed pre-implementation (`MAX_TAKE_SECONDS` missing), 5/5 green post-fix.

## Resolution

- Verdict: FIXED at the validator (`voyage/audio/acestep.py` — the audio half of this group); config-level ceiling RESIDUAL (below). The worker benchmark leg (candidate 3) is inherited, not separate: both `handle_benchmark` paths funnel through `validate_duration_seconds` (`audio_acestep` via `handle_generate_audio`, `sfx_mmaudio` directly), so a 2-hour benchmark override now fails in validation.
- Changes: `MAX_TAKE_SECONDS = 120.0` + three-way check (`not finite or <= 0 or > MAX`) in `validate_duration_seconds`, same message shape as the SFX sibling.
- Files changed: `voyage/audio/acestep.py` (+ new `tests/test_e2_ace_ceiling_155.py`: absurd rejected, boundary MAX/MAX+ε, legitimate 1-60 s pass, bottom half unchanged, SFX parity).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 5 passed; `test_acestep_contract` + `test_audio_workers` + `test_audio_request_validation` + `test_audio_planner` + `test_audio_take_ahead_guard` green (51 passed, 1 skipped). Ruff + format + mypy strict clean.
- DESIGN proposal (quoted text only, for the DESIGN owner — §37): "ACE take durations validate both halves: finite within `(0, MAX_TAKE_SECONDS]` (120 s, 2× the largest legitimate 60 s take) — absurd values fail in validation, never after minutes of DiT render. Mirrors the SFX `MAX_WINDOW_SECONDS` bound."
- Residuals (out of scope, precise): `voyage/config.py:348-353` (`AudioConfig.non_negative` covering `take_seconds`/`ahead_seconds`/`crossfade_seconds`, plus `DraftConfig.non_negative` at `:594-599`) still has no upper bound — a `take_seconds=1e6` TOML typo passes `load_config` and only fails at render. Needs the config owner to add a `field_validator` upper bound (same constant family).
