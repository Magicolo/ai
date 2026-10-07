# 225 — `_embed_texts` catches only `VoyageError`, so transport bugs crash the commit instead of falling back (MEDIUM)

## Technical description
`supervisor.py:1703-1736` docstring promises "None when unavailable
(fallback)". The `try` covers only `self._director.call(...)` with
`except VoyageError`. But `call()` can raise non-`VoyageError`: `OSError`
from a respawn path (see #219), `ValueError`/`OverflowError` from a poisoned
timeout reaching `select`, or any future transport bug. Those propagate out
of `_embed_texts` into `_accept_director_decision` → commit → `FAILED`,
instead of the token-set fallback the novelty path was designed for.

## Rationale
Embeddings are explicitly advisory ("a slow director must degrade to
fallback"). A transport hiccup upgrading an advisory probe into a
run-killing `FAILED` inverts the design priority. `_sample_gauges` (same
file, `:1354`) already uses bare `except Exception` for exactly this reason.

## Live evidence
Source shape + contrast (`inspect.getsource`):
`except VoyageError: return None` vs `_sample_gauges: except Exception:
continue  # best-effort`.

## Repro
Stub `director.call` to raise `OSError("select boom")`, call
`sup._embed_texts(["x"])` → raises instead of `None`.

## Source refs
- `Voyage/voyage/supervisor.py:1703-1736`, `:1348-1354`
- `Voyage/voyage/rpc.py:113-132`

## Online sources
- Same RPC framing refs as #200 — advisory probes must be total functions.

## Fix candidates
`except (VoyageError, OSError)` → `None` (keep
`MemoryError`/`KeyboardInterrupt` propagating); log a
`director_embed_fallback` metric with the exception class so the degradation
stays visible.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07)
Re-verified live against the current tree before fixing — every
load-bearing claim still holds: `Supervisor._embed_texts`
(`voyage/supervisor.py:1703-1717`) catches only `VoyageError` around
`self._director.call(...)`; `SubprocessWorker.call` (`voyage/rpc.py`)
raises `RecoverableWorkerError`/`FatalWorkerError` (both `VoyageError`
subclasses) on the handled paths, but raw `OSError` can still escape —
e.g. `os.set_blocking`/`os.read`/`select` failures outside the guarded
regions, or a stubbed/test-double `call` raising transport errors — and
those propagate through `_accept_director_decision` into a failed
commit instead of the documented `None` fallback. The
`_sample_gauges` contrast (`except Exception`, same file) is confirmed.
The issue's `ValueError`/`OverflowError`-from-poisoned-timeout example
is already closed upstream (`_require_finite_positive_timeout` raises
`RecoverableWorkerError`, a `VoyageError`, before any deadline math) —
so the fix narrows to the issue's own candidate: `(VoyageError,
OSError)`. `MemoryError` inherits `Exception`, not `OSError`;
`KeyboardInterrupt` inherits `BaseException` — both still propagate, as
required. No staleness found; no adjustment to the fix direction.

## Progress
- Broadened the catch in `Supervisor._embed_texts` to
  `except (VoyageError, OSError)`, emitting a `director_embed_fallback`
  metric (`error_class` + truncated `error`) on every fallback so the
  degradation stays visible; docstring updated. The only caller
  (`_accept_director_decision`) is already `None`-safe
  (`vectors[0] if vectors else None`).
- New `tests/test_issue_201_225_recoverable_embed.py` pins: stubbed
  `director.call` raising `OSError` → `None` + one
  `director_embed_fallback` event with `error_class == "OSError"`;
  `VoyageError` path still `None`, now metered; `MemoryError`
  propagates.
- Scoped verify in-container (`voyage:latest`, bind mount): `ruff check`
  + `ruff format --check` clean on all touched files; `mypy`
  `voyage/supervisor.py` clean; pytest 61 passed (7 new + neighbors —
  incl. `test_director_embed_degrades_after_crash`, which now also
  exercises the metered path).
- Changes left uncommitted for orchestrator review, per task scope.

## Resolution (2026-10-07)
Fixed as per the issue's candidate (`except (VoyageError, OSError) →
None`, `MemoryError`/`KeyboardInterrupt` propagating) plus the
`director_embed_fallback` metric with the exception class. No `errors.py`
taxonomy change was needed.
