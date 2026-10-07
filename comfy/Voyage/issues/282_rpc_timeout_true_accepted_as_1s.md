# 282 — `timeout=True` accepted as 1 s; `timeout=1.0` vs `True` indistinguishable

Severity: LOW (track A-11).

## Technical description

`_require_finite_positive_timeout` (`rpc.py:113-132`) tests
`isinstance(v,(int,float))` — `bool` is an `int` subclass, so `True` (== 1) passes as a
1-second deadline and `False` (== 0) is rejected as non-positive. The `checked_request`
worker side explicitly rejects `bool`-for-`int`; the RPC timeout side does not.

## Rationale

A caller passing a flag where a duration belongs (`timeout=done`) silently gets a 1 s
deadline → spurious `Recoverable` → burns restart budget. Same class as the `seed=True`
bug already fixed worker-side (test pins `seed=True → TypeError`).

## Live evidence

```
command: docker run --rm -v /home/goulade/Projects/ai/comfy/Voyage:/app -w /app voyage:latest python3 -c "
from voyage.rpc import _require_finite_positive_timeout
from voyage.errors import RecoverableWorkerError
for v in [True, False, 1, 0]:
    try:
        _require_finite_positive_timeout('m','op', v)
        print(repr(v), 'ACCEPTED')
    except RecoverableWorkerError:
        print(repr(v), 'rejected')
"
output:
True ACCEPTED
False rejected
1 ACCEPTED
0 rejected
```

Repro: `worker.call("health", {}, timeout=True)` → 1 s deadline instead of `Recoverable`
refusal.

## Source refs

`voyage/rpc.py:113-132`; `voyage/workers/loop.py:154-167` (bool guard precedent).

## Online sources

- Python docs on `bool`/`int` subclassing; issue-007 precedent in-tree.

## Fix candidates

- Mirror the loop guard: `if isinstance(v,bool): raise RecoverableWorkerError(...)`; add
  `True/False` pins to `test_rpc_deadline_finite.py`.

## Log

- 2026-10-07: filed from read-only Track A sweep; no code touched.
