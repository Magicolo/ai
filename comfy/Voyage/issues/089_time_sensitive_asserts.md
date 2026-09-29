# 089 — Time/environment-sensitive asserts (sleep race, wall-clock bound, RSS monotonicity, always-on endurance)

- Status: resolved 2026-09-25 (this track: all four hardened + gating documented)
- Severity: low (flaky-gate class; endurance marker runs on every gate)
- Area: tests — `test_recovery.py:90`, `test_failure_policy.py:147-150`,
  `test_benchmark.py:149,161-171`
- Rank rationale: pass-2 finding; 041 covered the dead *gpu* marker — this is the
  *endurance* marker (always-on, not dead) plus four wall-clock/RSS asserts.

## Technical description

- `test_recovery.py:90` (`time.sleep(0.05)` pacing a pause-at-boundary race).
- `test_failure_policy.py:147-150`
  (`assert time.monotonic() - started < 10.0` wall bound).
- `test_benchmark.py:149` (`rss_delta_mb >= 0` — RSS can legitimately *fall*
  between segments after GC).
- `test_benchmark.py:161-171` (`@pytest.mark.endurance`, `rss_delta_mb < 500`,
  always executed — neither `gates.sh` nor `test.sh` passes `-m`, so the
  "long-running stability" marker runs on every gate).

`rg -n "pytest.mark" tests/*.py` shows only the one endurance marker and zero
`gpu` uses (sweep output).

## Why this is an issue

Wall-clock bounds (`< 10.0 s`), sleep-paced races (`0.05 s` polling a thread
boundary), RSS monotonicity (memory can legitimately fall after GC), and an
always-on endurance test make gates flaky under load — failures that blame the
code for the machine's mood. The endurance marker running on every gate also
slows each run with no opt-out, so flakiness and slowness compound.

## Evidence

All four sites verified live (re-run 2026-09-25):

```
$ rg -n "sleep\(0\.05\)|monotonic\(\).*10\.0|rss_delta_mb.*>= 0|mark\.endurance" Voyage/tests/test_recovery.py Voyage/tests/test_failure_policy.py Voyage/tests/test_benchmark.py
Voyage/tests/test_recovery.py:90:            time.sleep(0.05)
Voyage/tests/test_failure_policy.py:150:        assert time.monotonic() - started < 10.0
Voyage/tests/test_benchmark.py:149:    assert summary["rss_delta_mb"] >= 0
Voyage/tests/test_benchmark.py:161:@pytest.mark.endurance
```

## Reproduction

Read the lines; run the endurance test under memory pressure to observe
`rss_delta_mb` variance.

## Source references

- Files/lines above.

## Resolution candidates

Replace sleep-pacing with event polling; drop or widen the 10 s wall bound;
assert `abs()`/trend-tolerance on RSS; either exclude `endurance` from default
gates (`-m "not endurance"`) or document it as always-on.

## Investigation / progress / resolution log

- 2026-09-25: found by pass-2 tests sweep.
- 2026-09-25: issue-file repair — added `## Why this is an issue`; re-verified
  all four sites live (outputs pasted above — all match).
- 2026-09-25: resolved — (a) recovery: DONE poll keeps the 0.05 s cadence
  but gains a 120 s deadline + loud `assert` when the first segment never
  commits (no silent fallthrough to a confusing zero-segment failure);
  (b) failure-policy: 10 s → 60 s wall bound with comment (the call resolves
  in ~0.2 s; the budget only guards the hang class on loaded machines);
  (c) benchmark gauges: one-sided `>= 0` → symmetric `abs(delta) < 500`
  (same flatness budget as endurance; GC dips no longer flake);
  (d) endurance gating DECIDED as always-on: documented in a comment above
  the marker (gates/test scripts pass no `-m` by design; opt out with
  `-m "not endurance"`). Scripts untouched (out of scope).
- 2026-09-29: verification (this track) — re-read live:
  `test_recovery.py:74-81` (deadline + assert), `test_failure_policy.py:135-137`
  (60 s bound + comment), `test_benchmark.py:134-136` (`abs(delta) < 500`),
  `:148-152` (always-on comment + marker) all present; scope pytest passes.
  No edit needed (already fixed).
