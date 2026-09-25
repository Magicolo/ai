# 070 — `derive_seed` label-join ambiguity: distinct label tuples collide

- Status: fixed
- Severity: medium (silent stream merging — the exact failure the module
  docstring claims to prevent)
- Area: seeds — `Voyage/voyage/seeds.py:14-17`
- Rank rationale: pass-2 finding; current call sites use single-token labels so
  nothing collides today, but any future `:`-containing label silently merges
  streams.

## Technical description

`key = (str(run_seed) + ":" + ":".join(...)).encode()` — `("a:b","c")` and
`("a","b:c")` produce the identical key string.

## Why this is an issue

The module docstring promises stream separation ("a change to director logging
can never silently alter video randomness"), but colon-joining breaks that exact
guarantee for any label containing a colon. Nothing collides today only because
current call sites use single-token labels — the first concept name or profile
string with a `:` silently merges two RNG streams with no error to catch it.

## Evidence (code experiment, host stdlib-only, verified by orchestrator 2026-09-25)

```
$ python3 -c "...derive_seed(7,'a:b','c')==derive_seed(7,'a','b:c')..."
True 952123984
```

## Reproduction

One-liner above. Future trigger: a concept name or profile string containing
`:` passed as a label.

## Source references

- `Voyage/voyage/seeds.py:14-17`.

## Resolution candidates

Length-prefix or `repr()` each label, e.g.
`":".join(f"{len(str(l))}:{l}" ...)`; add a two-line regression test. (Distinct
from 040's fake-seed issue, which is worker behavior, not derivation.)

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep; collision reproduced live.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`seeds.py:14-17` — match); re-ran probe
  (`derive_seed(7,'a:b','c')==derive_seed(7,'a','b:c')` → `True 952123984`,
  confirmed).
- 2026-09-25 (fix): FIXED — `derive_seed` in `voyage/seeds.py:14-25` now
  length-prefixes each label (`"<len>:<label>"` joined with `":"`), so
  `("a:b","c")` and `("a","b:c")` encode distinctly. Verified live: the
  colliding pair now separates while identical inputs stay stable. Note:
  derived values change for all existing label tuples (new encoding); no
  golden values are pinned in tests or code (checked), and current call
  sites use single-token labels. Tests:
  `test_derive_seed_separates_colon_labels` in `tests/test_unit.py`
  (extends the existing stability/separation test). Gates: ruff + format +
  mypy strict clean on all scope files, 424 pytest passed in-container.
- Resolution: fixed as above; label tuples are unambiguously encoded.
