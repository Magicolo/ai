# `_safe_system_statistics` catches only `ZoomyError`

- Severity: medium (robustness — one backend hiccup kills the stats UI).
- Status: FIXED. `_safe_system_statistics` now catches `Exception`
  (never `BaseException` — documented in the docstring) and returns `None`,
  matching the "engine offline" contract. Test:
  `test_safe_system_statistics_treats_backend_errors_as_offline` (a
  `RuntimeError`-raising engine double → `None` → "engine offline" line).
  `zoomy/interface.py:844-856`.

## Evidence

```python
def _safe_system_statistics(engine: EngineProtocol) -> EngineStatistics | None:
    try:
        return engine.engine_statistics()
    except ZoomyError:
        return None
```

`EngineProtocol` is structural — any implementation (including the real
`LocalEngine`, whose `engine_statistics` calls `torch.cuda.mem_get_info`)
may raise `RuntimeError`/`OSError`/torch errors that are not `ZoomyError`.
The name promises "safe", but a non-`ZoomyError` propagates out of the
10 s timer tick (`refresh_status`), breaking the stats line and health
badge until reload. Note the current `LocalEngine` masks its own torch
failures as `None` (`local_engine.py:197-198`), so this bites with any
*future* engine or a torch failure mode outside
`(ImportError, RuntimeError, ValueError)`.

## Fix

Catch `Exception` (narrow: `Exception`, never `BaseException` — interrupts
must propagate) and return `None`, matching the documented "engine
offline" contract. Test: fake engine raising `RuntimeError` →
statistics line renders "engine offline", no raise.

## Verification

- New test: `RuntimeError`-raising engine → `None`, badge shows offline.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
