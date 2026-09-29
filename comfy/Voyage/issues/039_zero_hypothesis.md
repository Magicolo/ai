# 039 — Zero Hypothesis: 361 tests, 0 properties; `nan`/`inf` slip through (`inf` → 5.6e162 beats)

- Status: resolved (fixed 2026-09-25: hypothesis pin + conftest profile + 9 properties + pins)
- Severity: medium-high (entire input-domain class untested; live `nan` leak +
  astronomic `inf` blowup)
- Area: tests — property-based testing
- Rank rationale: beat math guards `<=0` but never `nan/inf`; both verified live
  to produce garbage that flows into segment planning.

## Technical description

`Voyage/pyproject.toml:6-20` has no hypothesis dep (Zoomy pins
`hypothesis==6.168.0` + `tests/conftest.py` profile +
`HYPOTHESIS_STORAGE_DIRECTORY`); `rg -n "hypothesis|given|example" Voyage/tests`
→ 0 code hits (verified live 2026-09-25: `no-hypothesis-hits`; only
`pytest.approx` + 6 `parametrize` sites in `test_rhythm.py:15,45`,
`test_generate.py:53,69,75`, `test_tui_state.py:130`). Consequences:

- `beats_for_segment`/`quantize_take_seconds` (`voyage/audio/beat.py:22-58`) guard
  `<=0` with `ValueError`, tested at `0.0/-2.0` (`test_rhythm.py:36-58`), but never
  `nan/inf/-0.0/1e-300/1e308`: `nan <= 0` is False → `nan*60/nan` → `nan < 60`
  False → returns `(4, nan)` silently.
- No property `0<=sim<=1` symmetric/idempotent for `token_set_similarity`
  (`""`/`""`==1.0 pinned in `test_unit.py:94` only as an example).
- No stateful filesystem properties (numbering/contiguity/orphan scan
  `media.py:434-437` tested only with hand-built fixtures).
- No determinism test (fakes ignore seed — see 040).

## Why this is an issue

Example tests pin the cases authors thought of; the `nan`/`inf` leaks prove the interesting cases are the ones nobody thought of — a `nan` BPM and a ~5.6e162 beat count flowing silently into segment planning, where the latter is a credible loop-bound DoS. Without property tests, every numeric ingress carries the same unexamined tail of non-finite, negative, and extreme inputs, and each needs its own hand-written pin discovered the hard way. Hypothesis buys the whole input-domain class at once instead of one anecdote per bug report.

## Evidence (code experiments, host, verified by orchestrator 2026-09-25)

```
$ python3 -c "...beats_for_segment(float('nan'),4) / (float('inf'),4)..."
nan-> (4, nan)
inf-> (5617791046444737211654078721215702292556178059194708039794690036179146118921905097897139916325246669527046414884444362538940441232908842252656430276192208823201965046059784704400851161354703458893321819998351435577491134526104885300757004288, nan)
```

`nan` leaks as BPM; `inf` yields a ~5.6e162 beat count (downstream loop-bound
DoS — pairs with 006's unbounded-`frames` concern).

## Reproduction

Commands above; `rg -n "hypothesis|given" Voyage/tests | wc -l` → 0.

## Source references

- `voyage/audio/beat.py:22-58`; `Voyage/tests/test_rhythm.py:15-58`;
  `Voyage/pyproject.toml:6-20`.

## Resolution candidates

1. Add `hypothesis` to `dev`; `tests/conftest.py` NUL-free/whole-second/
   bounded-count strategies per Zoomy; properties for beat math/seed
   derivation/similarity bounds.
2. Deterministic example pins for `nan/inf/0/negative` on every numeric ingress
   (`beats_for_segment` first — reject non-finite with `ValueError`).

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep; `nan`/`inf` probes executed live by
  orchestrator (the `inf` blowup is worse than the sweep reported).
- 2026-09-25: repair pass — added `## Why this is an issue`; re-ran the beat
  probes live (`nan->(4, nan)`; `inf`→~5.6e162 beats, `bpm: nan` — current);
  `rg -n "hypothesis|given" tests/` still 0 code hits.
- 2026-09-26 (resolution, FIXED): `hypothesis==6.168.0` pinned in
  `[project.optional-dependencies] dev` (Zoomy parity pin); `conftest.py`
  carries the container profile (`database=None`) + two shared strategies
  (`bounded_counts`, NUL-free `short_texts`) behind a `find_spec` guard so
  the suite still collects where the package is absent (property modules
  skip via `importorskip` — verified: 22 passed / 3 skipped without it).
  New: `tests/test_beat_properties.py` (BPM identity, minimal doubling,
  determinism, grid alignment, sub-segment floor + `-0.0`/extreme-finite/
  negative-base pins) and `tests/test_similarity_properties.py`
  (bounded+symmetric, reflexive, canonicalize idempotence, cosine
  symmetry/self-unit + empty/case/order/underflow/mismatch pins) — 27
  tests, all green with Hypothesis (49 passed incl. seeds properties).
  - Live discoveries pinned, not just fixed: the cosine self-unit property
    found float-underflow (`[4.4e-197]` squares to sub-denormal → norm 0.0
    → returns 0.0 for a nonzero vector; pinned as
    `test_cosine_underflow_reads_as_zero`, policy call logged for the
    concepts owner); `characters(blacklist_characters=…)` still draws lone
    surrogates (probed `\ud800`) — conftest blacklists `Cs` explicitly
    with a Literal-typed tuple (mypy-strict clean).
  - Adopted (already in tree): nan/inf rejection lives in `beat.py`
    `_require_finite` + `test_rhythm.py` pins (batch-1 work, not duplicated).
  - Follow-ups: `test_seeds_properties.py` `short_labels` uses bare
    `st.text()` — if that shares the `characters()` surrogate default,
    tighten to conftest `short_texts` (owning pass, one line + rerun);
    seed-determinism of fakes is issue 040's, not this one.
  - Hook (lock owner): `requirements.lock` needs the matching lines
    (`hypothesis==6.168.0` + `sortedcontainers` transitive) — the slim
    image installs from the lock with `--no-deps -e .`, so the gate image
    does not carry Hypothesis until the lock is regenerated per its header
    procedure; gates were verified green via ephemeral install meanwhile.
- Open: none in this slice.
