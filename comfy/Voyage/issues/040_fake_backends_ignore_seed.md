# 040 — Fake backends ignore `seed`/`prompt`: seed-determinism claims untestable

- Status: open
- Severity: medium-high (test fidelity — "deterministic seed" tests through fakes
  are vacuous)
- Area: tests — `voyage/fake_backends.py:35,72-73`
- Rank rationale: two runs with different seeds produce byte-identical media, so
  the GPU determinism story (`±5 cross-process floor`) has no fake-model
  characterization.

## Technical description

`fake_backends.py:35` (`del prompt, seed`), `:72` (`del style, seed`); frequency
derives from `energy` only (`:73`), video from geometry only. `test_unit.py:34-40`
pins `derive_seed` stability/separation but never that the seed reaches bytes.

## Why this is an issue

Every "deterministic seed" test that runs through the fakes is vacuous: two runs with different seeds produce byte-identical media, so the test asserts determinism of a constant. That leaves the GPU determinism story — the ±5 cross-process floor the project actually relies on — with no fake-model characterization and no cheap CI proxy, forcing all determinism validation onto scarce idle-GPU time. Either the fakes should honor seeds or the suite should stop claiming they test seeding.

## Evidence

By construction: render twice with seeds 1 vs 9999 to same-size paths, `cmp` →
identical.

Verified live 2026-09-25:

```
$ rg -n "del prompt, seed|del style, seed" voyage/fake_backends.py
35:        del prompt, seed
72:        del style, seed
```
(`:73` derives tone frequency from `energy` only; video geometry-only. Refs current.)

## Reproduction

Two fake renders differing only in seed; `cmp` the outputs.

## Source references

- `Voyage/voyage/fake_backends.py:35,72-73`; `Voyage/tests/test_unit.py:34-40`.

## Resolution candidates

1. Mix seed into fakes (e.g. `hue=seed%360` overlay or `sine` phase), OR
2. Document fakes as seed-insensitive + add a determinism test asserting exactly
   that + a GPU-only determinism test behind the `gpu` marker (see 041).

## Investigation / progress / resolution log

- 2026-09-25: found by standards sweep.
- 2026-09-25: repair pass — added `## Why this is an issue`; `del` lines
  re-verified live (`fake_backends.py:35,72`, current); pasted rg output into
  Evidence.
- Open: pick (1) or (2); GPU determinism test needs idle-GPU time.
