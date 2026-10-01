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

## Progress log

- 2026-09-30 (Group E1): live re-verified premise FIRST per contract —
  `render_sfx_bed` still left-folds (`sfx_finalize.py:487-491`,
  `stems[0]` accum + `sfx_blend_*.wav` steps, single-stem fast path at
  `:470-486`) and `build_final_audio` still folds identically
  (`voyage/media.py` accum/`final_blend_*.wav` loop); no batch-5/6 or
  concurrent change to either fold (concurrent `sfx_finalize.py` hunks are
  manifest-caption reads only — disjoint but the file is hot). Checked for
  an ownable `media.py` slice: the two folds' restructure (single-graph
  join or checkpointed intermediates) changes blend topology/timeline
  exactness and the primary site (`sfx_finalize.py:487-491`) is explicitly
  out of scope (concurrent hot). The safe media-side micro-wins
  (duration-memo through `_blend_pair`, intermediate reuse) would touch
  the foreign call signature or tmpdir lifecycle for a LOW-teeth gain on
  a MEDIUM perf issue — wrong trade. Verdict: CONFIRMED, no ownable
  media.py leg; logged as precise residual per contract. No code changed.

## Resolution

- Residual for the SFX-pass owner — fix spec (issue candidates 1 + 3):
  replace the `accum → _blend_pair → step` left-fold in `render_sfx_bed`
  (`sfx_finalize.py:487-491`) with a single ffmpeg filter chain over the
  N stems (chained adelay + amix with the same manual-fade recipe, or the
  `fade < 0.1` concat precedent generalized) so each stem is read once —
  O(N) I/O, one spawn — with a 3-window byte-parity test (fold vs
  single-graph) and an N-window spawn-count pin (≤ 2); emit per-blend
  wall seconds so soak trends fold cost vs timeline length. Same shape
  applies to `build_final_audio`'s fold (`voyage/media.py` accum loop) —
  coordinate both, since a music+SFX finalize pays the quadratic twice.
  Quantified scale (unchanged from the issue): 10-min final ≈ 85 windows
  ≈ 84 blends, last blend re-reading ~600 s of accum to append 8 s.
- DESIGN proposal (quoted text only, not applied — DESIGN.md untouched):
  "> Finalize audio joins are single-graph: each stem/window is read once
  > through one chained adelay+amix filter invocation (O(N) I/O, one
  > spawn), never re-encoded N−1 times through a left fold. Soak trends
  > join wall-clock vs timeline length to prove the scaling."
- Files changed: this issue file only (log appended; original above intact).

## Progress log (2026-09-30, this pass — owned files: media.py + sfx_finalize.py)

- Re-verified live FIRST: `render_sfx_bed` still left-folds, `build_final_audio`
  still folds, `_blend_pair` still probes both inputs per blend. CONFIRMED.
- Fold-restructure verdict: BEHAVIOR-RISKY, did not land. Decisive blocker:
  `tests/test_final_blend_scale.py::test_final_blend_never_spawns_wide_acrossfade_graph`
  pins EVERY ffmpeg call in the final blend to at most 2 audio inputs
  (live 31-segment acrossfade deadlock) — a single-graph N-way join
  violates it by construction, and that file is another group's (this
  pass is new-tests-only). Per the issue's own fallback, landed the
  probe-memo half + per-blend timing instead; fold topology untouched.
- TDD: new `tests/test_issue_152_blend_probe_memo.py` (7 tests) failed
  first in-container (TypeError on the new kwargs, probe-count
  assertions), green after the fix.

## Resolution (this pass)

- Verdict: PARTIAL — probe-memo + timing landed; single-graph join
  recorded as residual below.
- Files changed: `voyage/media.py` (`_blend_fade_seconds` pure helper,
  `_blend_pair` gains keyword-only `first_seconds`/`second_seconds`
  (probe fallback kept when None) + `timing_ms` out-list;
  `assemble_segment_audio` + `build_final_audio` gain keyword-only
  `blend_timings` and thread per-input durations probed once, accum
  tracked arithmetically with the shared fade formula),
  `voyage/sfx_finalize.py` (`render_sfx_bed` gains keyword-only
  `blend_timings`, same threading over stems), new tests only.
- Test evidence: new file 7/7 green; related suites green
  (`test_sfx_finalize`, `test_final_blend_scale`, `test_audio_accounting`,
  `test_generation_stack` — 68 passed; `test_finalize_fastpath`,
  `test_state_integrity`, `test_cli_split`, `test_sfx_contract`,
  `test_commit_side_integrity_095_101_104` — 66 passed). Gates on touched
  files: `ruff check` + `ruff format --check` + `ruff check --select
  PLR2004` + `mypy` (5 source files) all green.
- DESIGN proposal (quoted text only, not applied — DESIGN.md untouched):
  "> Blend folds thread known durations: each stem/window is probed once
  > and the accum tracks arithmetically through one shared fade formula
  > (O(N) probes, one timing entry per pair); the pairwise topology stays
  > until the wide-graph deadlock class is re-proven safe. Soak trends
  > per-blend wall milliseconds vs timeline length."
- Residual (exact handoff): the single-graph N-way join from the issue
  log (one `-filter_complex` with chained afade/adelay/amix, each stem
  decoded once) is unblocked only by relaxing
  `tests/test_final_blend_scale.py:107-136` (the ≤2-input pin) with a
  deadlock-class proof for wide MANUAL-fade graphs (the incident was
  acrossfade-specific) — owner: that file's track + a GPU long-run;
  do not attempt from the sfx/media side alone.

## Progress log (2026-09-30, batch 12)

- Re-read `voyage/sfx_finalize.py:513-529` (probe-memo fold intact) and
  `voyage/media.py:600-684` (`_blend_fade_seconds` + `_blend_pair` with
  `first_seconds`/`second_seconds`/`timing_ms`) before testing; `git diff
  HEAD --` on both files was EMPTY (no concurrent hunks), and both stay
  UNCHANGED by this leg (proof ran in a new test file only).
- Deadlock-class proof executed live (in-container `voyage:latest`, CPU
  ffmpeg, new `tests/test_issue_152_wide_manual_join_proof.py`): one
  `-filter_complex` over N inputs with chained manual fades only
  (`afade` out/in + `adelay` + `amix inputs=N … normalize=0`, NEVER
  acrossfade), fold arithmetic mirrored exactly (same
  `_blend_fade_seconds` per pair, same `%.3f` fades, same integer-ms
  delays):
  - No-hang: PASSED — N=8 × 4 s sine stems complete in ~1 s under the
    180 s `subprocess` timeout guard, output non-empty, duration exact
    (25.0 s ≈ 8×4 − 7×1 within 0.15).
  - Bit-parity: FAILED — N=4 sizes (4992102 B) and durations exact, but
    first byte diff at index 2304110. PCM diagnosis (stdlib
    wave+array): 624000 frames each, max 65536 s32 units (= 1 s16 LSB,
    ~−69 dBFS peak against signal peak 189792256), mean ~5041
    (~−91 dBFS), 95860/624000 samples differ, confined to
    second-and-later overlap regions (sec6/sec9 full-second diffs,
    all other seconds zero). N=2 wide vs `_blend_pair` IS byte-identical
    (336000 frames, 0 diffs), so the recipe matches and the gap is
    generational ordering across chained blends (pairwise re-quantizes
    the accum through s32le files per blend; wide holds one float
    chain). A forced-`aformat=s32` diagnostic did NOT rescue parity
    (max 129600, 143720 diffs — worse), so no format-pin recipe exists
    from this pass; exact filter-rounding mechanism undetermined.
- TDD shape: the parity test was written strict (byte-equal) first and
  failed as above; the committed file pins the measured outcome instead
  (durations + sizes + closeness bounds ~2x headroom, labeled
  characterization) plus the passing no-hang and N=2-identity tests.

## Resolution (2026-09-30, batch 12)

- Verdict: BLOCKED (proof split: hang PASS, parity FAIL) — probe-memo
  path kept, NO behavior change to `voyage/sfx_finalize.py` or
  `voyage/media.py` (both files untouched; `git diff` clean on them).
- Files changed: new `tests/test_issue_152_wide_manual_join_proof.py`
  only (3 tests: N=8 no-hang, N=4 closeness characterization, N=2
  byte-identity recipe check). `tests/test_final_blend_scale.py`
  untouched (foreign file — the ≤2-input pin still holds and still
  passes).
- Test evidence (in-container `voyage:latest`, CPU-only): new file 3/3
  green; related suites green — proof + probe-memo + scale +
  `test_sfx_finalize` = 31 passed.
- Per-file gates: `ruff check` + `ruff format --check` clean on the new
  file (B905 `strict=True` + I001 import order fixed during the pass);
  no mypy scope change (untracked test files stay out of `gates.sh`
  mypy list until committed, per the script's contract).
- DESIGN proposal (quoted text only, not applied — DESIGN.md untouched):
  "> Blend folds stay pairwise: a wide MANUAL-fade single graph completes
  > without hanging but is not bit-identical to the fold (generational
  > ordering, ≤1 s16 LSB in post-first overlaps), so the O(N) probe-memo
  > fold remains until parity is proven. Soak trends per-blend wall
  > milliseconds vs timeline length."
- Residuals (exact handoff): landing the join needs BOTH (1) a parity
  rescue (exact rounding mechanism + a graph that reproduces fold bytes
  — the `aformat=s32` attempt failed, so this is research, not wiring)
  AND (2) the owner of `tests/test_final_blend_scale.py:107-136` to
  relax the ≤2-input pin with a 31-input-scale proof (this pass proved
  N=8 CPU-only; incident scale + GPU long-run remain open) — owner:
  that file's track; do not attempt from the sfx/media side alone.

## Progress log (2026-10-01, batch 13 — PIN CONFIRMATION ONLY)

- Premise re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip): the ≤2-input pin still holds —
  `tests/test_final_blend_scale.py::test_final_blend_never_spawns_wide_acrossfade_graph`
  passes unmodified, as do the probe-memo suite
  (`tests/test_issue_152_blend_probe_memo.py`) and the wide-join proof
  (`tests/test_issue_152_wide_manual_join_proof.py`). `git diff HEAD --`
  on `voyage/sfx_finalize.py` + `voyage/media.py` is EMPTY (no concurrent
  hunks in this leg) and both files stay UNCHANGED by this leg.
- Test evidence: `test_final_blend_scale.py` +
  `test_issue_152_blend_probe_memo.py` + `test_registry_pins.py` = 25
  passed in-container (shared run); the wider batch-13 neighbor set =
  117 passed, 8 skipped, with 6 FOREIGN failures in
  `test_augment_models.py` (all one root: `ImportError: cannot import
  name 'warn_if_deprecated_backend' from 'voyage.config'` — another
  group's in-flight config/cli split, zero lines mine).
- Verdict: PROVEN-BLOCKED, unchanged — hang PASS / parity FAIL stands,
  pairwise probe-memo fold remains, no behavior change. Nothing to do;
  moving on per the batch brief.

## Resolution (2026-10-01, batch 13)

- Verdict: confirmed blocked (pin holds, proof split stands). Files
  changed: none (this issue file only). DESIGN proposals: none (the
  batch-12 pairwise-fold proposal stands as quoted).
- Residuals: unchanged — (1) parity rescue research + (2) the
  `test_final_blend_scale.py:107-136` owner's 31-input-scale proof
  before any single-graph join lands.

## Progress log (2026-10-01, verification-only pass — PIN CONFIRMATION ONLY)

- Premise re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip): the ≤2-input pin still holds —
  `tests/test_final_blend_scale.py` +
  `tests/test_issue_152_blend_probe_memo.py` +
  `tests/test_issue_152_wide_manual_join_proof.py` = **14 passed**
  unmodified. `git diff HEAD --` on `voyage/sfx_finalize.py` +
  `voyage/media.py` is EMPTY and both files stay UNCHANGED by this leg.
- Verdict: PROVEN-BLOCKED, unchanged — hang PASS / parity FAIL stands
  (batch-12 proof), pairwise probe-memo fold remains, no behavior
  change. Nothing to do.

## Resolution (2026-10-01, verification-only pass)

- Verdict: confirmed blocked (pin holds, proof split stands). Files
  changed: none (this issue file only). DESIGN proposals: none (the
  batch-12 pairwise-fold proposal stands as quoted).
- Residuals: unchanged — (1) parity rescue research + (2) the
  `test_final_blend_scale.py` owner's 31-input-scale proof before any
  single-graph join lands.

## Progress log (2026-10-01, re-probe pass — PIN CONFIRMATION ONLY)

- Premise re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip): the ≤2-input pin still holds —
  `tests/test_final_blend_scale.py::test_final_blend_never_spawns_wide_acrossfade_graph`
  (asserts EVERY ffmpeg call in the final blend takes at most 2 audio
  inputs) passes unmodified, as do the probe-memo suite
  (`tests/test_issue_152_blend_probe_memo.py`) and the wide-join proof
  (`tests/test_issue_152_wide_manual_join_proof.py`). Pin text read
  live (`:107-136`, docstring + `wide == []` assertion intact) — the
  owner has NOT relaxed it, so no join landing was evaluated.
- `git diff --name-only` on `voyage/sfx_finalize.py` +
  `voyage/media.py` is EMPTY (no concurrent hunks) and both files stay
  UNCHANGED by this leg.
- Test evidence: `test_final_blend_scale.py` +
  `test_issue_152_blend_probe_memo.py` +
  `test_issue_152_wide_manual_join_proof.py` = **14 passed** in 8.08 s
  unmodified (shared run).
- Verdict: PROVEN-BLOCKED, unchanged — hang PASS / parity FAIL stands
  (batch-12 proof), pairwise probe-memo fold remains, no behavior
  change. Nothing to do.

## Resolution (2026-10-01, re-probe pass)

- Verdict: confirmed blocked (pin holds, proof split stands). Files
  changed: none (this issue file only). DESIGN proposals: none (the
  batch-12 pairwise-fold proposal stands as quoted).
- Residuals: unchanged — (1) parity rescue research + (2) the
  `test_final_blend_scale.py` owner's 31-input-scale proof before any
  single-graph join lands.

## Progress log (2026-10-01, record-only maintenance pass — PIN CONFIRMATION ONLY)

- Premise re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip): the ≤2-input pin still holds —
  `tests/test_final_blend_scale.py::test_final_blend_never_spawns_wide_acrossfade_graph`
  (`:107`, asserts EVERY ffmpeg call in the final blend takes at most 2
  audio inputs) passes unmodified, as do the probe-memo suite
  (`tests/test_issue_152_blend_probe_memo.py`) and the wide-join proof
  (`tests/test_issue_152_wide_manual_join_proof.py`). Pin text read live
  (`:107-136`, docstring + `wide == []` assertion intact) — the owner
  has NOT relaxed it, so no join landing was evaluated.
- `git diff --name-only` on `voyage/sfx_finalize.py` + `voyage/media.py`
  is EMPTY (no concurrent hunks) and both files stay UNCHANGED by this
  leg (no code — per brief).
- Test evidence: `test_final_blend_scale.py` +
  `test_issue_152_blend_probe_memo.py` +
  `test_issue_152_wide_manual_join_proof.py` = **14 passed** in 7.85 s
  unmodified (shared run).
- Verdict: PROVEN-BLOCKED, unchanged — hang PASS / parity FAIL stands
  (batch-12 proof), pairwise probe-memo fold remains, no behavior change.
  Nothing to do.

## Resolution (2026-10-01, record-only maintenance pass)

- Verdict: confirmed blocked (pin holds, proof split stands). Files
  changed: none (this issue file only). DESIGN proposals: none (the
  batch-12 pairwise-fold proposal stands as quoted).
- Residuals: unchanged — (1) parity rescue research + (2) the
  `test_final_blend_scale.py` owner's 31-input-scale proof before any
  single-graph join lands.

## Progress log (2026-10-01, parity-rescue research — user-ordered, read-first)

- Premise re-verified live FIRST (in-container `voyage:latest`,
  CPU-only, no host pip, scratch in /tmp only): the pin now lives at
  `tests/test_finalize_fastpath.py:275`
  (`test_final_blend_never_spawns_wide_acrossfade_graph` — moved from
  the deleted `test_final_blend_scale.py`, same <=2-input assertion,
  green unmodified); production fold vs flat wide still diverges exactly
  as batch-12 measured (N=4 sine: sizes 4992102 B both, first diff frame
  288001, max 65536 s32 units, mean ~5041, dirty secs [6,9]).
- MECHANISM FOUND (exact): `afade` runs in its input's NATIVE sample
  format (ffmpeg 7.1.5 filter negotiation, proven by `-v verbose`
  auto-insert dumps) — s16-fed pair graph inserts NO converter before
  `afade` (`s16 -> fltp` lands AFTER it, before `adelay`/`amix`), so the
  fade is s16-INTEGER math truncating every faded sample to the s16
  grid; s32-fed pair graph (fold blends 2+) likewise keeps `afade` in
  s32-INTEGER (full precision). Unit pins: s16-fed `afade`-alone fade
  region 100% grid with truncation signature (DC 13902: sample-1 ideal
  18981 -> 0, sample-47999 ideal 13901.71 -> 13901); s32-fed
  `afade`-alone fade region grid fraction 0.0014 (chance level). So
  blend 1 fades in s16 exactly like every wide chain (overlap-1/pure
  regions identical), while blends 2+ fade in s32 — same ideal gains,
  finer truncation grid — differing by <=1 s16 LSB (65536 s32 units),
  confined to second-and-later overlaps. `amix`/`adelay` always run
  fltp (exact for grid values: <=16 significant bits < 24-bit
  mantissa); the terminal fltp->s32 quantization is shared and
  innocent.
- Decisive probe matrix (all CPU-only `voyage:latest`, N=4 4 s sine):
  R1 fold-twice byte-identical (max=0 — no random dither stage);
  R2 intermediate ladder vs wide: s16 -> max=0 BIT-IDENTICAL,
  s32/f32/f64 -> max=65536/65536/65535 at the same first-diff frame
  (file width is irrelevant — the negotiated filter domain decides);
  R3 chained single-graph without barriers vs fold max=16 (pure fltp
  rounding-order residue) vs wide max=65536 (files incidental);
  R4 `aformat=dbl` wide diverges from BOTH flat-wide and fold from
  overlap-1 (float gains everywhere — also explains why batch-12's
  forced-s32 was worse: it moved stage 1 off the s16 grid too).
  Sine-sample forensics first suggested impossible gains (implied
  g=1/6) — a cross-domain comparison artifact; DC fixtures (exact
  gain solve) + converter dumps gave the true story.
- CONSTRUCTIONS (both bit-identical, NEITHER landed — recorded only):
  (1) single-spawn rescue — chained pairwise stages with an
  `aformat=sample_fmts=s32` barrier per stage == production fold
  byte-for-byte (N=4: 624000 frames, max=0, sizes 4992102 B; each
  stem decoded once, one spawn); (2) other direction —
  s16-intermediate fold == flat wide byte-for-byte (N=4 AND N=8:
  1200000 frames, max=0; both fade every stage in s16 — not a
  production option, it reintroduces the 16-bit generational loss the
  s32 intermediates were chosen to avoid).
- Closed legs: dither (R1 determinism + converter dump shows format
  converters only — the integer-trunc path has no dither stage);
  `aresample` precision (no resampling: every converter 48000->48000);
  `aformat` (s32-barrier rescues, dbl/s32-everywhere diverge).

## Resolution (2026-10-01, parity-rescue research)

- Verdict: MECHANISM NAILED, CONSTRUCTION PROVEN, STILL BLOCKED ON
  LANDING. No behavior change (`voyage/media.py`,
  `voyage/sfx_finalize.py`, the pin test, and all other tests
  untouched — `git diff` clean on them).
- Files changed: new `tests/test_issue_152_parity_research.py` only
  (6 tests, all green in-container: s16-truncation unit pin,
  s32-precision contrast, s16-fold==wide, staged-s32==fold,
  bounded-divergence characterization, determinism; `ruff check` +
  `ruff format --check` clean).
- Test evidence: new file 6/6 in ~2 s; neighbor run green — new +
  `test_finalize_fastpath` + probe-memo + wide-proof +
  `test_lock_manifest_agreement` = 31 passed in 17.17 s.
- DESIGN proposals: none (behavior unchanged; the batch-12
  pairwise-fold proposal stands as quoted).
- Residuals: landing construction (1) needs the pin owner's
  31-input-scale no-hang proof + <=2-input pin relaxation first
  (N=8 proven here CPU-only; incident scale + GPU long-run remain
  theirs) — do not wire from the sfx/media side alone.
