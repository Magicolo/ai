# 018 — `probe()` does not wrap `json.loads`; raw `JSONDecodeError` escapes the `MediaError` taxonomy

- Severity: MEDIUM
- Group: correctness/media — Rank: 3/5
- File:line: `voyage/media.py:79-100` (`probe`)

## Technical description

Live code (re-verified 2026-09-30):

```python
# voyage/media.py:79-98
def probe(path: Path) -> dict[str, Any]:
    proc = run_capture([... "ffprobe", ..., "-of", "json", str(path)])
    if proc.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {proc.stderr[-2000:]}")
    data: Any = json.loads(proc.stdout or "{}")   # <-- UNWRAPPED
    if not isinstance(data, dict):
        raise MediaError(f"ffprobe returned non-object for {path}")
    return data
```

`json.loads` raises `json.JSONDecodeError` (subclass of `ValueError`) on truncated / non-JSON stdout — reachable when ffprobe is killed mid-write, disk-full partial output, locale-injected warnings on stdout, or a wedged binary emitting garbage with exit 0. The function maps the *non-zero-exit* case and the *non-dict* case to `MediaError`, but the *malformed-JSON* case propagates raw.

`JSONDecodeError` is **not** a `MediaError` (`VoyageError` hierarchy). Callers that `except (MediaError, StateError, DiskSpaceError)` — notably `cmd_finalize` (`voyage/cli.py:972-974`):

```python
try:
    finalize_run(...)
except (MediaError, StateError, DiskSpaceError) as exc:
    print(f"finalize failed: {exc}", ...)
```

— do **not** catch it. The same holds for `run_segments`' `except VoyageError` (which *would* miss it, falling into the generic `except Exception → FatalWorkerError` belt-and-braces with a misleading "failed with JSONDecodeError" message instead of a media diagnostic). The taxonomy promise (issue 002/007: branch on class, media failures are `MediaError`) is broken at the most common media entry point.

## Why it matters

- Every commit probes video/audio (`probed_take_seconds`, `validate_video/audio`, assembly duration reads). One corrupt ffprobe stdout turns a routine media failure (retryable, `--skip-bad`-able) into an unclassified exception that rests the run at FAILED with a confusing message.
- `probed_take_seconds` (`voyage/media.py:211-225`) *does* wrap its `probe` call in `except (OSError, ValueError)` → `MediaError` — proving the codebase already knows this is needed everywhere else. `probe` itself is the hole.

## Live evidence

Host probe (stdlib only, `voyage.media` imports without pydantic for these symbols — verified live):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
import json, inspect
from voyage import media
print(inspect.getsource(media.probe)[400:700])
try: json.loads("{not json")
except Exception as e:
    print(type(e).__name__, "| ValueError?", isinstance(e, ValueError))
    from voyage.errors import MediaError
    print("MediaError?", isinstance(e, MediaError))
PY
data: Any = json.loads(proc.stdout or "{}")
...
JSONDecodeError | ValueError? True
MediaError? False
```

- `json.loads` failure is a `ValueError`, not a `MediaError`. `probe` has no `try` around it (confirmed by source: the only `raise MediaError` lines are the returncode and non-dict branches).
- `cmd_finalize` catches `(MediaError, StateError, DiskSpaceError)` — `JSONDecodeError` escapes to a traceback, exit code 1 via unhandled exception instead of the `finalize failed:` path.

## Repro steps

1. Stub `run_capture` to return `CompletedProcess(returncode=0, stdout="{truncated", stderr="")`.
2. Call `probe(Path("x.mp4"))` → raw `json.JSONDecodeError: Expecting property name...` instead of `MediaError`.
3. End-to-end: corrupt an ffprobe binary wrapper to emit garbage with exit 0; `voyage finalize` tracebacks instead of printing `finalize failed:`.

## Fix candidates

1. (Preferred, one-liner) Wrap in `probe`:
   ```python
   try:
       data = json.loads(proc.stdout or "{}")
   except ValueError as exc:
       raise MediaError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
   ```
   Fixes every caller at once (commit, finalize, validate-via-probe, assembly).
2. Keep `probed_take_seconds`'s existing wrap as-is (it becomes redundant but harmless) — or simplify it to rely on `probe`'s new guarantee.
3. Regression test: fake `run_capture` returning `("{oops", 0)` → assert `pytest.raises(MediaError)`; plus a `cmd_finalize` taxonomy test asserting `JSONDecodeError` can never escape `finalize_run`.

## References

- `voyage/media.py:79-98` (unwrapped `json.loads` at `:95`), contrast `:211-225` (`probed_take_seconds` wraps correctly).
- `voyage/cli.py:972-974` (`cmd_finalize` catches `MediaError`, misses `JSONDecodeError`); `voyage/supervisor.py:643-690` (commit `except VoyageError` vs generic `except Exception` — wrong branch taken).
- Python `json.JSONDecodeError` subclasses `ValueError`: https://docs.python.org/3/library/json.html#json.JSONDecodeError
- Issue 002 (non-`VoyageError` escapes) / 007 (taxonomy erased) — same family.

## Progress log

- 2026-09-30: re-verified live in-container before touching anything — holds as-read: `voyage/media.py:118` (`data: Any = json.loads(proc.stdout or "{}")`, zero `try` around it; only the returncode and non-dict branches raise `MediaError`). `JSONDecodeError` confirmed `ValueError`, not `MediaError`; `cmd_finalize` catches `(MediaError, StateError, DiskSpaceError)`, so the raw error escapes the `finalize failed:` path.
- 2026-09-30 (TDD red): wrote `Voyage/tests/test_media_robustness_rank2.py` first; `test_probe_invalid_json_raises_media_error` + `test_probe_garbage_stdout_raises_media_error` failed with raw `JSONDecodeError` as predicted.
- 2026-09-30 (implement, `voyage/media.py` only — candidate 1, the one-liner): `probe` wraps `json.loads` in `try/except ValueError → MediaError(f"ffprobe returned invalid JSON for {path}: {exc}")`. `probed_take_seconds`' existing `except (OSError, ValueError)` wrap kept as-is (redundant but harmless — a `MediaError` from `probe` propagates unchanged, which is already the taxonomy).
- 2026-09-30 (TDD green + gates): 18 passed in the new file; related suites 93 passed + 1 torch-gated skip; `ruff check` + `ruff format --check` + `mypy` (strict) clean on all touched files.

## Resolution

- Verdict: fixed in scope (one hunk in `probe`). Every caller (commit, finalize, validate-via-probe, assembly) inherits the taxonomy at once.
- Files changed: `Voyage/voyage/media.py` (`probe`), `Voyage/tests/test_media_robustness_rank2.py` (new: invalid/garbage-JSON tests + happy-path pin).
- Test evidence: stubbed `run_capture` returning `("{truncated", 0)` / `("not json at all", 0)` now raises `MediaError` (was `json.JSONDecodeError`); `test_probe_valid_json_still_parses` pins the happy path.
- DESIGN.md as-built proposal (not applied — DESIGN.md untouched per directive): in §§54-57 (media taxonomy), add "malformed ffprobe JSON (truncated stdout, wedged binary with exit 0) is a `MediaError`, retryable/`--skip-bad`-able like any media failure."
- Residuals: none in this leg — `probed_take_seconds`' wrap could be simplified to rely on `probe`'s guarantee, but it is harmless and out of the minimal-diff scope.
