# 002 — Non-`VoyageError` escapes `commit_one_segment` → run rests at stale RUNNING, never FAILED

- Status: resolved in live tree (ingress wraps + commit backstop landed)
- Severity: HIGH (state integrity; resolved, record only)
- Group: correctness/errors — Rank: 1/5 (critical pattern, fixed)
- Area: correctness — error taxonomy at persistence/media ingress
- Rank rationale: any torn ledger line, corrupt concept line, or weird video left a
  RUNNING corpse that resume logic misread; the exact failures a SIGKILL-heavy
  system must survive.

## Technical description

Pre-fix, `voyage/supervisor.py:418-433` only transitioned to FAILED/PAUSED on
`VoyageError`, but several commit-path ingress points raised plain stdlib/pydantic
errors:

```python
# voyage/audio/planner.py:188-191 (pass 1)
takes.append(AudioTake.from_dict(json.loads(line)))  # json.JSONDecodeError, KeyError, ValueError
# voyage/concepts.py:95-98 (pass 1)
self._records.append(ConceptRecord.model_validate_json(line))  # pydantic ValidationError
# voyage/media.py:81-83 (pass 1)
actual_fps = float(num) / float(den or 1) if num else 0.0  # ZeroDivisionError on "0/0","24/0"
```

Note `den or 1` only guards the empty string — `"0"` is truthy, so `24/0` divides
by zero for real.

Live state (re-verified 2026-09-30): the commit boundary wraps all three —
corrupt takes ledger → `StateError` (`voyage/supervisor.py:1170`), corrupt
concept history → `StateError` (`voyage/supervisor.py:1466`),
`voyage/persistence.py:68-99` wraps manifest/state reads as `StateError`, and the
commit tail has a `FatalWorkerError` backstop so no bare exception can strand
RUNNING.

## Why this is an issue

- SIGKILL mid-`append_take` leaves a torn last ledger line; a hand edit/disk error
  corrupts `concepts.jsonl`; a truncated/corrupt mp4 yields `avg_frame_rate="0/0"`.
- All three propagated out of `commit_one_segment` uncaught by `run_segments`, so
  `state.json` stayed RUNNING forever. Operators and `voyage run` resume key off
  terminal statuses; a RUNNING corpse misleads.
- The `errors.py` contract ("branch on class, never message") was violated at every
  persistence boundary.

## Evidence

Live verification 2026-09-30 (in-container, `voyage:latest`):

```
$ docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "..."
ledger-torn: JSONDecodeError VoyageError? False
concept-corrupt: ValidationError VoyageError? False
fps-24/0: ZeroDivisionError VoyageError? False
wrap-takes-ledger: True
wrap-concepts: True
backstop-commit: True
```

The stdlib/pydantic classes are still non-`VoyageError` at the ingress (proving the
wraps are load-bearing), and the wraps exist:

```
$ rg -n "except VoyageError|StateError\(f\"segment.*corrupt|except Exception as exc" voyage/supervisor.py | head
1162:            raise StateError(f"segment {segment_id}: corrupt takes ledger {ledger}: {exc}") from exc
1458:            raise StateError(f"segment {segment_id}: corrupt concept history: {exc}") from exc
```

## Reproduction

1. `echo TRUNCATED >> <run>/audio/takes.jsonl` (or `kill -9` mid-append), then
   `commit_one_segment` → pre-fix `JSONDecodeError` escaped, status stayed RUNNING;
   now `StateError` → FAILED.
2. `echo NOT_JSON >> <run>/novelty/concepts.jsonl` → pre-fix `ValidationError`
   escaped; now `StateError` → FAILED.
3. Craft a video whose `avg_frame_rate` probes as `0/0` → pre-fix `ZeroDivisionError`
   escaped; now `MediaError` at the fps gate.

## Source references

- `voyage/supervisor.py:1158-1162` (takes wrap), `:1454-1458` (concepts wrap),
  `:645-688` (error-class backstop), `:1833` (`commit_one_segment` guard).
- `voyage/persistence.py:68-99` (manifest/state wraps); `voyage/errors.py:23-39`
  (taxonomy); `voyage/media.py:26-56` (fps/alignment gates).

## Resolution candidates

1. (Landed) Wrap all persistence/media ingress in `VoyageError`: `load_takes`
   try/except `(ValueError, KeyError, TypeError)` → `StateError`;
   `ConceptStore` per-line try → `StateError`; fps parse guard → `MediaError`.
2. (Landed) Belt-and-braces: catch `Exception` at the `commit_one_segment`
   boundary, transition to FAILED, re-raise as `FatalWorkerError` (never let
   RUNNING survive an exception).
3. Crash-matrix tests: torn `takes.jsonl`, corrupt `concepts.jsonl`, `0/0` fps —
   each must end FAILED, and `validate_run` must report the cause
   (`tests/test_crash_matrix.py`).

## Online references

- Python `json.JSONDecodeError` (subclass of `ValueError`, not a domain error):
  https://docs.python.org/3/library/json.html
- Pydantic `ValidationError` (raised by `model_validate_json` on bad payloads):
  https://docs.pydantic.dev/latest/
- Python exception taxonomy rationale — catch precise classes at boundaries:
  https://docs.python.org/3/tutorial/errors.html

## Investigation / progress / resolution log

- 2026-09-25: found by correctness sweep; live exception-class probes executed.
- Resolution batch 3: ingress wraps + backstop landed.
- 2026-09-30: re-verified live (wraps + backstop present; stdlib classes still
  non-`VoyageError`, proving the wraps carry the guarantee); reconstructed from
  archived pass-1 text (commit `b5d7dda`). Status → resolved.
