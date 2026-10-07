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
