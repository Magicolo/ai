# 082 — Vacuous self-comparison assert in `test_causvid_worker.py:317`

- Status: open
- Severity: low (redundant line; cannot fail)
- Area: tests — `Voyage/tests/test_causvid_worker.py:306-319`
- Rank rationale: pass-2 test-hygiene finding; AST self-compare scan over all
  test files flags exactly two `x == x` asserts, one legitimate.

## Technical description

`assert video_causvid.split_tail_novel(81, 3) ==
video_causvid.split_tail_novel(81, 3)` — identical call on both sides; cannot
fail unless the pure function is nondeterministic. Surrounding lines (:308-311,
:318-319) already assert the concrete values `(9,72)`, `dropped+novel==81`,
`novel==72`. The other flagged site (`test_unit.py:35`, determinism + paired
inequality separation) is a legitimate determinism test.

## Why this is an issue

A tautological assert can never fail, so it contributes nothing but false
coverage confidence — a future regression in `split_tail_novel` would still
show this line "passing". The surrounding lines already pin the real values,
so deleting it loses zero signal and removes noise from the gate.

## Evidence

AST self-compare scan (re-run 2026-09-25, stdlib only):

```
$ python3 -c "...walk test_causvid_worker.py for assert-Compare with identical sides..."
self-compare at line 317
```

Only hit in that file; `test_unit.py:35` is the legitimate determinism case
(paired with an inequality separation). Read `:306-319` to confirm the
surrounding concrete asserts.

## Reproduction

Read the lines.

## Source references

- Files/lines above.

## Resolution candidates

Delete line 317 (coverage already provided by :311/:318/:319).

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  cited lines live (`test_causvid_worker.py:306-319`, self-compare at `:317`
  — match); re-ran AST scan (single hit at 317, confirmed).
- Open: delete the line.
