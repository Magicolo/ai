# System-RAM failure reports `0.0 GiB`, not `n/a`

- Severity: low (misleading stats line on non-Linux / `/proc` failure).
- Status: verified open, narrowed (VRAM half already fixed).
  `zoomy/local_engine.py:1194-1208`, `zoomy/interface.py:834-838`.

## Evidence

```python
def _read_system_memory_bytes() -> tuple[int, int]:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return (0, 0)
```

The Linux path is hardcoded; any `/proc` failure (non-Linux dev,
sandboxed container) returns `(0, 0)`, and the stats line renders
`RAM 0.0 / 0.0 GiB` — indistinguishable from a real zero reading.
Contrast the VRAM path (`local_engine.py:186-204`), which reports `None`
and renders `n/a` via `_gibibytes`. Malformed `/proc` lines are handled
(`ValueError → continue`, missing keys → `.get(…, 0)`), so only the
total-failure branch lies.

## Fix

Return `(None, None)` (widen `EngineStatistics` RAM fields to
`int | None` like the VRAM fields) so the line shows `n/a`; keep the
per-line tolerance as-is. Update the stats-line test for the `None` RAM
case.

## Verification

- New test: missing `/proc/meminfo` → `n/a` in the rendered line.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
