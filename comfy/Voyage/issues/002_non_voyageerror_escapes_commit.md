# 002 — Non-`VoyageError` escapes `commit_one_segment` → run rests at stale RUNNING, never FAILED

- Status: resolved 2026-09-25 (supervisor track; ingress-site moves deferred, see log)
- Severity: critical (state integrity / crash recovery)
- Area: correctness — error taxonomy at persistence/media ingress
- Rank rationale: any torn ledger line, corrupt concept line, or weird video
  leaves a RUNNING corpse that resume logic misreads; the exact failures a
  SIGKILL-heavy system must survive.

## Technical description

`voyage/supervisor.py:418-433` only transitions to FAILED/PAUSED on `VoyageError`:

```python
except VoyageError as exc:
    failed = read_state(self._run_dir)
    ...
    failed.status = "FAILED"   # only on VoyageError
```

But several commit-path ingress points raise plain stdlib/pydantic errors:

```python
# voyage/audio/planner.py:188-191
takes.append(AudioTake.from_dict(json.loads(line)))  # json.JSONDecodeError, KeyError, ValueError
# voyage/concepts.py:95-98
self._records.append(ConceptRecord.model_validate_json(line))  # pydantic ValidationError
# voyage/media.py:81-83
actual_fps = float(num) / float(den or 1) if num else 0.0  # ZeroDivisionError on "0/0","24/0"
```

Note `den or 1` only guards the empty string — `"0"` is truthy, so `24/0` divides by
zero for real.

## Why this is an issue

- SIGKILL mid-`append_take` leaves a torn last ledger line; a hand edit/disk error
  corrupts `concepts.jsonl`; a truncated/corrupt mp4 yields `avg_frame_rate="0/0"`.
- All three propagate out of `commit_one_segment` uncaught by `run_segments`, so
  `state.json` stays RUNNING forever. Operators and `voyage run` resume key off
  terminal statuses; a RUNNING corpse misleads.
- The `errors.py` contract ("branch on class, never message") is violated at every
  persistence boundary.

## Evidence

(Code experiments — `PYTHONPATH=Voyage`, host.)

Sub-agent ran live (orchestrator reviewed the sites):

```
load_takes raised JSONDecodeError: Expecting value (is VoyageError? False)
ConceptStore raised ValidationError (VoyageError? False)
0/0 RAISES ZeroDivisionError float division by zero
24/0 RAISES ZeroDivisionError float division by zero
```

Orchestrator verified the fps site reads `num, _, den = rate.partition("/")` at
`voyage/media.py:82-83` with the `den or 1` guard that misses `"0"`.

## Reproduction

1. `echo TRUNCATED >> <run>/audio/takes.jsonl` (or `kill -9` mid-append), then
   `commit_one_segment` → `JSONDecodeError` escapes, status stays RUNNING.
2. `echo NOT_JSON >> <run>/novelty/concepts.jsonl` → `ValidationError` escapes.
3. Craft a video whose `avg_frame_rate` probes as `0/0` → `ZeroDivisionError` escapes.

## Source references

- `voyage/supervisor.py:418-433` (except-VoyageError-only transition).
- `voyage/audio/planner.py:188-191`, `voyage/concepts.py:95-98`,
  `voyage/media.py:81-83`.
- Audit `voyage/atomic.py:read_json`, `read_state`, `probe` callers for the same class.

## Resolution candidates

1. Wrap all persistence/media ingress in `VoyageError`: `load_takes`
   try/except `(ValueError, KeyError, TypeError)` → `StateError`;
   `ConceptStore.__init__` per-line try → `StateError`; fps parse guard
   `den in ("", "0")` → `MediaError`.
2. Belt-and-braces: catch `Exception` at the `commit_one_segment` boundary,
   transition to FAILED, and re-raise as `FatalWorkerError` (never let RUNNING
   survive an exception). Prefer (1) for precise classes, keep (2) as a backstop.
3. Add crash-matrix tests: torn `takes.jsonl`, corrupt `concepts.jsonl`, `0/0`
   fps — each must end FAILED, and `validate_run` must report the cause.

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep; live exception-class probes executed.
- Open: implement wraps + tests; re-run gates.
- 2026-09-25 (repair pass): refs verified current (`supervisor.py:420`
  except-block; `planner.py:191`; `concepts.py:98`; `media.py:82`);
  ZeroDivisionError probes re-run live; `## Why this is an issue` already
  present, no change.
- 2026-09-25 (RESOLVED, supervisor track): candidates 1 (in-scope parts) +
  2 (backstop). `planner.py`/`concepts.py`/`media.py` are outside this
  track's scope (untouched), so the wraps land at the supervisor-owned
  boundary instead: takes-ledger load → `StateError`
  (`voyage/supervisor.py:1017-1029`), `ConceptStore` construction →
  `StateError` (`voyage/supervisor.py:1295-1308`), manifest read →
  `StateError` (`voyage/persistence.py:63-70`; `read_state` already wrapped
  everything). Plus the belt-and-braces backstop in `run_segments`
  (`voyage/supervisor.py:542-575`): any non-`VoyageError` (torn JSON,
  `ValidationError`, media `ZeroDivisionError`) rests the run at FAILED and
  re-raises as `FatalWorkerError` — RUNNING can no longer survive an
  exception. All three repros verified FAILED via
  `Voyage/tests/test_commit_hardening.py` (`test_torn_takes_ledger_rests_failed`,
  `test_corrupt_concept_history_rests_failed`,
  `test_unexpected_exception_backstops_to_failed` with an injected
  `ZeroDivisionError`). FOLLOW-UP for the ingress-owning track: move the
  wraps to `load_takes`/`ConceptStore.__init__`/fps-parse proper (precise
  classes at the source); behavior is already correct through the boundary.
  Gates: full `Voyage/scripts/gates.sh` green (626 passed).
