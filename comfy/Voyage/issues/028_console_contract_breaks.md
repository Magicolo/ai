# 028 — Console layer breaks its own contracts: `error()` ignores the injected stream; spinner leaks on `BaseException`; tracker `KeyError` on unknown labels

- Severity: LOW (display layer: lost errors, leaked spinner, completion-path crash)
- Group: observability/console — Rank: 4/5
- File:line: `voyage/console.py:149-150` (`error`), `voyage/console.py:160-198` (`stage`), `voyage/console.py:361-373` (`ParallelDownloadTracker`)
- Overlaps: 063 (console) — same file/same stream-split root cause; recommend keep 063, fold 028 on fix.

## Description

(a) Every method honors the injected `stream` except `error()`, which prints to `sys.stderr`. Tests/embeds capturing `stream` lose errors. The only reason TUI errors appear is the TUI's `redirect_stdout/redirect_stderr(buffer)` merge — the console contract ("same words, either sink") is broken by one method.

(b) `stage()` stops the spinner/status only on `Exception`. `KeyboardInterrupt`/`CancelledError` (`BaseException`) skip `stop.set()`/`status.stop()` — the daemon ticker thread spins on and the rich `Status` is never stopped, corrupting terminal state after the very interrupt the user just issued.

(c) `ParallelDownloadTracker.succeed`/`fail` do `self._tasks[label]` with no guard — a typo'd or double-reported label raises `KeyError` inside the completion path, i.e. the success reporter crashes while reporting success. (`_started` already uses `.get()` — the inconsistency proves the intent.)

## Rationale

- A display layer's contract is "same words, either sink". One method breaking the sink rule plus unguarded dict indexing in the success path are exactly the faults that surface during the failure they're meant to report.
- The spinner leak is user-visible terminal corruption on Ctrl-C, the most common interruption of a long GPU render.
- The fix is three one-liners; the cost of leaving them is paid in confused debugging sessions.

## Live evidence (read, 2026-09-30)

`voyage/console.py:140-150` (every sibling uses `self._stream` / `self.styled` except `error`):

```python
    def info(self, message: str) -> None:
        self.styled("▸", message, "cyan")

    def ok(self, message: str) -> None:
        self.styled("✓", message, "green")

    def warn(self, message: str) -> None:
        self.styled("⚠", message, "yellow")

    def error(self, message: str) -> None:
        print(f"✗ {message}", file=sys.stderr)
```

`voyage/console.py:178-189` (rich branch — `except Exception` only):

```python
            try:
                yield
            except Exception:
                stop.set()
                status.stop()
                ...
                raise
```

`voyage/console.py:361-373`:

```python
    def succeed(self, label: str) -> None:
        elapsed = time.monotonic() - self._started.get(label, time.monotonic())
        if self._progress is not None:
            self._progress.update(self._tasks[label], completed=1)
```

Sweep probe (preserved Track B result): `VoyageConsole(no_color=True, stream=s).error('boom')` prints `✗ boom` to real stderr; `s.getvalue() == ''`.

## Repro

```bash
docker run --rm -v "$PWD:/app" -w /app voyage:latest python3 -c "
import io
from voyage.console import VoyageConsole
s = io.StringIO()
c = VoyageConsole(verbose=False, no_color=True, stream=s)
c.error('boom')
print('captured:', repr(s.getvalue()))"
# Buggy: captured == '' (went to real stderr)
grep -n "except Exception" voyage/console.py
```

## Fix candidates

1. `print(..., file=self._stream)` in `error` (or a dedicated error stream if stderr routing is intended — then document it and give tests a handle).
2. `except BaseException` for the spinner teardown with re-raise (both branches).
3. `.get()` + warning line for unknown tracker labels in `succeed`/`fail`.

## Refs (with links/quotes)

- Textual validation pattern (validators run on change/submit/blur; failures surface inline rather than as crashes) — the analogous discipline for console trackers is validate-then-report. — https://github.com/Textualize/textual/blob/main/docs/widgets/input.md

## Progress log

- 2026-09-30 re-verified live alongside 063 (same file, same probes): `error()` ignores the injected stream, `stage()` leaks the spinner on `BaseException` and drops elapsed on the plain failure path, tracker indexes `_tasks` directly. Premise CONFIRMED.

## Resolution

- FOLDED into 063 on fix (same file, same stream-split root cause). Fix lives in `voyage/console.py`, coverage in `tests/test_observability_rank2.py` (4 console tests) — see 063 Progress log / Resolution for verdict, files, test evidence, residuals, and DESIGN proposals. This file kept intact as the fold record; no separate implementation.
