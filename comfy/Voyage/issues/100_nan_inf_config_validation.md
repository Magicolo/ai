# 100 — NaN/inf slip through `positive_seconds` into the RPC deadline, surfacing as raw ValueError/OverflowError

**Severity:** MEDIUM

**File:line (verified live 2026-09-30):**
- `voyage/config.py:478-483` (`positive_seconds` validator: `if value <= 0: raise`)
- `voyage/config.py:458` (`rpc_timeout_seconds: float = 600.0`)
- `voyage/rpc.py:63` (`DEFAULT_RPC_TIMEOUT_SECONDS = 600.0`)
- `voyage/rpc.py:114` (`SubprocessWorker.__init__` stores `timeout` unvalidated)
- `voyage/rpc.py:198-240` (`_read_response_line`: `deadline = time.monotonic() + effective_timeout`; `select.select([readable], [], [], remaining)`)
- `voyage/rpc.py:282-295` (`call()`: `effective_timeout = self._timeout if timeout is None else timeout`, no validation at the top)
- `voyage/supervisor.py:301,315,325` (config `rpc_timeout_seconds` flows straight into worker `timeout=`)
- Precedent: `voyage/audio/beat.py:24-33` (`_require_finite` with `math.isfinite`, documented ValueError contract)

**Description:**
`positive_seconds` rejects only `value <= 0`. In IEEE 754 (which Python floats follow), `nan <= 0` is `False` and `inf <= 0` is `False`, so both `nan` and `inf` pass validation and land in `SubprocessWorker._timeout` via `supervisor.py:301/315/322`. From there:

- `nan`: `deadline = monotonic() + nan` → `nan`; `remaining = nan - monotonic()` → `nan`; the `remaining <= 0` guard (line 227) is `False` for NaN, so the loop proceeds to `select.select([readable], [], [], nan)`, which raises raw `ValueError: Invalid value NaN (not a number)` — not `RecoverableWorkerError`, so the supervisor's restart path (`_call_with_restart`, which branches on error class per `voyage/errors.py`) never engages and the run dies via the belt-and-braces `except Exception` at `supervisor.py:659-690` as a misleading `FAILED`.
- `inf`: same path reaches `select.select(..., inf)`, which raises raw `OverflowError: timestamp out of range for platform time_t` on Linux. Additionally `deadline = monotonic() + inf` is `inf`, so even a finite-remaining computation would never expire — the timeout is silently infinite until the select call itself blows up.

Neither exception is a `VoyageError` (`voyage/errors.py:1-48` taxonomy), so both bypass the recoverable/fatal branching the whole supervisor is built on. The in-tree precedent is `beat._require_finite` (lines 24-33), whose docstring already documents exactly this trap: "bare round() internals leak confusing messages (nan) or the wrong exception class (OverflowError for inf)". The config validator simply never adopted it.

**Rationale:**
`rpc_timeout_seconds` is operator-editable TOML (`config.py:685` default block) and also settable per-call via `call(..., timeout=...)`. A single `nan` (e.g. from a computed override, a corrupted TOML float like `nan`, or arithmetic upstream) turns every worker op into a non-restartable crash instead of a bounded timeout. Timeouts are the load-bearing reliability mechanism of the worker architecture (issue 001's non-blocking deadline read); a validator that admits the two values guaranteed to break the deadline math leaves the mechanism unguarded at its only choke point.

**Live evidence (current tree, host stdlib only — no voyage imports needed):**
```
$ python3 -c "
import math, select
nan, inf = float('nan'), float('inf')
print('nan<=0:', nan <= 0, '| inf<=0:', inf <= 0)
for label, t in [('nan', nan), ('inf', inf)]:
    try: select.select([], [], [], t)
    except (ValueError, OverflowError) as e: print(label, '->', type(e).__name__, e)"
nan<=0: False | inf<=0: False | -inf<=0: True
nan -> ValueError : Invalid value NaN (not a number)
inf -> OverflowError : timestamp out of range for platform time_t
```
Source chain (all live-read):
```
config.py:480-483:  def positive_seconds(cls, value: float) -> float:
                          if value <= 0:
                              raise ValueError("must be positive")
rpc.py:282:             effective_timeout = self._timeout if timeout is None else timeout
rpc.py:228:             deadline = time.monotonic() + effective_timeout
rpc.py:236:             ready, _, _ = select.select([readable], [], [], remaining)
```
No `isfinite` anywhere in `config.py` or `rpc.py` (verified by grep); the only `math.isfinite` gates in the tree are `audio/beat.py`, `audio/acestep.py:53`, and `audio/mmaudio_sfx.py:79`.

**Repro (CPU, no GPU):**
1. In-container: `ProjectConfig` (or bare `VoyageConfig`) with `rpc_timeout_seconds=float("nan")` — `model_validate` succeeds (validator passes).
2. Build a `SubprocessWorker(..., timeout=config.voyage.rpc_timeout_seconds)` and `call("health", {})` (or directly `_read_response_line` with the NaN timeout).
3. Observe raw `ValueError` (NaN) / `OverflowError` (inf) instead of `RecoverableWorkerError`; in a full run, the commit lands in the `except Exception` branch (`supervisor.py:659`) and rests `FAILED` with no restart attempted.
4. TOML variant: `rpc_timeout_seconds = nan` in `voyage.toml` (TOML floats admit `nan`/`inf`) → `load_config` succeeds → same crash at first worker op.

**Fix candidates:**
- Add `math.isfinite` to `positive_seconds` (`config.py:478-483`), mirroring `beat._require_finite`: `if not math.isfinite(value) or value <= 0: raise ValueError(...)`. One line, kills the TOML/config path.
- Validate at the top of `SubprocessWorker.call()` (`rpc.py:282`): reject non-finite/non-positive `effective_timeout` as `RecoverableWorkerError` (or `ConfigurationError` for the stored timeout vs caller-supplied timeout — caller-supplied garbage is arguably a programming error, fail-loud either way but as a `VoyageError`).
- Optionally clamp in `__init__` too (defense in depth for programmatically constructed workers that bypass config validation).
- Tests: `nan`/`inf`/negative/zero/timeout matrix for the validator; `call(timeout=nan)` raises a `VoyageError` subclass without spawning anything; TOML `rpc_timeout_seconds = nan` fails `load_config` with `ConfigurationError`.

**Refs:**
- Overlaps with 104 (NaN/inf validation family — 104 owns the ledger/walk leg; this issue owns the RPC-deadline leg).
- In-tree precedent with the exact contract to copy: `voyage/audio/beat.py:24-33` — "Reject nan/inf with the documented ValueError contract… an infinite segment silently yields an infinite take downstream (issue 068)."
- `voyage/errors.py:1-48` (restart decisions "branch on error class, never on message string matching" — raw `ValueError`/`OverflowError` defeat this).
- `select.select` timeout semantics: timeout must be a non-negative finite number; NaN → `ValueError`, inf → `OverflowError` (Python docs, https://docs.python.org/3/library/select.html; both confirmed live above on this host).
- TOML v1.0 permits `nan`, `inf`, `-inf` as float values — any float-typed TOML knob without an `isfinite` gate inherits this issue; `rpc_timeout_seconds` is the one wired to `select`.

## Progress log (resolution 2026-09-30)

- As-read re-verified live in-container: `VoyageConfig(rpc_timeout_seconds=nan/inf)` and `DraftConfig(take_seconds=nan)` all validated successfully (bug confirmed); `call()`/`_read_response_line` had no `isfinite` gate; `math` was already imported in `config.py` but not in `rpc.py`.
- TDD: wrote `tests/test_rpc_deadline_finite.py` first — 13 failed / 7 passed: every nan/inf rejection plus the TOML-nan case failed; `-inf` variants and the finite boundary already passed (`-inf <= 0` is True; `-inf` deadline trips the `remaining <= 0` guard into `RecoverableWorkerError` by accident).
- Fix: `config.py` `positive_seconds` now `if not math.isfinite(value) or value <= 0` ("must be a finite positive number", mirroring the `beat._require_finite` precedent); `DraftConfig.take_seconds` validator gained the same `isfinite` gate ("must be a finite non-negative number", mirroring `AudioConfig`); `rpc.py` gained `_require_finite_positive_timeout` (rejects non-`int`/`float`, non-finite, or non-positive as `RecoverableWorkerError` — the same class an expired deadline raises), called at the top of `call()` (after the not-running `Fatal` check, before any deadline math) and at the top of `_read_response_line` (defense past `call`). `__init__` deliberately untouched: construction never raises, every use validates.
- Sibling audit: `AudioConfig` take/ahead/crossfade/overlap-cap already `isfinite`-gated (pinned by `test_unit.py`); `backends.VideoSegmentRequest.positive_seconds` and `frames_for_segment_seconds` already gated (the pattern copied); `AudioConfig.energy` / `VoyageConfig.novelty_threshold` range checks reject nan/inf via comparison semantics (`not 0 <= nan <= 1` is True) — verified, no change. `fps` validators are int-typed (nan cannot pass the int wire check) — verified, no change. `validate_benchmark_counts` (`workers/loop.py`) and `DirectorConfig.temperature` / `VoyageConfig` transition-interval floats lack gates but live outside this contract's editable files — recorded as follow-ups below, not fixed here.

## Resolution

- Verdict: RESOLVED for the RPC-deadline leg. NaN/inf can no longer reach `select.select`: config/TOML NaN fails `load_config` with `ConfigurationError`; programmatic NaN/inf (stored or per-call) fails as `RecoverableWorkerError`, so the restart budget engages instead of the belt-and-braces `FAILED`.
- Files changed: `voyage/config.py` (2 validators); `voyage/rpc.py` (`import math`, helper + 2 call sites); new `tests/test_rpc_deadline_finite.py` (20 tests: validator matrix, per-call/stored/reader rejections, TOML-nan).
- Test evidence (in-container, CPU-only): new module 20/20 green (was 13 failed pre-fix); scoped regressions `test_rpc_timeout` (137 resync intact), `test_unit`, `test_draft`, `test_rhythm`, `test_config_resolution`, `test_failure_policy`, `test_benchmark_counts` all green; `ruff check` + `ruff format --check` clean; `mypy voyage` clean (49 files).
- Follow-ups (out-of-contract files, not fixed): `workers/loop.validate_benchmark_counts` should reject non-`int`/non-finite counts before `range()` (raw `TypeError` today); `DirectorConfig.temperature` and `VoyageConfig.major_transition_min/max_seconds` / `world_decision_interval_seconds` have no validators (no deadline/loop-bound feed found — low risk, but TOML `nan` survives into them).
