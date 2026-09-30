# 016 — `_checked_tape_path` uses lexical `relative_to` + `exists()` (follows symlinks); dir passes; TOCTOU check-then-use

- Severity: HIGH
- Group: security/paths — Rank: 1/5
- File:line: `voyage/supervisor.py:403-415` (`_checked_tape_path`)

## Technical description

Live code (re-verified 2026-09-30):

```python
def _checked_tape_path(self, tape: str, segment_id: str) -> str:
    candidate = Path(tape)
    if not candidate.is_absolute():
        candidate = self._run_dir / candidate
    try:
        candidate.relative_to(self._run_dir)   # (1) lexical only — no normalization, no symlink resolution
    except ValueError:
        raise MediaError(...) from None
    if not candidate.exists():                 # (2) follows symlinks; True for dirs too
        raise MediaError(...)
    return str(candidate)
```

Three gaps:

1. **Symlink escape.** `relative_to` is purely lexical; `exists()` follows symlinks. `segments/evil.pt -> /etc/passwd` (or `/models/...`) passes both gates: lexically inside, physically outside. The supervisor then hands the outside path to the video worker `resume` call (read primitive; worker may exfiltrate content into metrics/logs) or probes it.
2. **Dir passes.** `exists()` is `True` for directories. A worker reporting a directory as `recovery_tape` passes validation; the downstream `resume` / `torch.load` / probe then fails with an unclassified `IsADirectoryError` / `OSError` instead of the intended `MediaError`, bypassing the error taxonomy (issue 007) and restart-budget branching.
3. **TOCTOU.** Check (`exists`) and use (worker `resume` reads the path later, possibly after restart) are separated in time. A path swapped between check and use (symlink planted after validation, file replaced) is not detected. Classic check-then-use race.

## Why it matters

Worker-reported paths are explicitly untrusted (issue 006 — this function *is* the 006 gate). A gate that can be bypassed with a one-line symlink defeats the whole validation layer. Combined with 015 (consumer side trusts even more), the run dir is not a containment boundary today.

## Live evidence

Stdlib probe (host, no container deps):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    run = Path(td)/"run"; run.mkdir(); (run/"segments").mkdir()
    outside = Path(td)/"outside.pt"; outside.write_text("x")
    link = run/"segments"/"evil.pt"; link.symlink_to(outside)
    candidate = run/"segments/evil.pt"
    print(candidate.relative_to(run))  # lexical: OK
    print(candidate.exists())          # follows symlink: True
    d = run/"segments"/"adir"; d.mkdir()
    print(d.exists())                  # dir: True -> would pass
PY
segments/evil.pt
True
True
```

- Lexical check passes, `exists()` is `True` while the real target (`.../outside.pt`) is outside the run.
- A directory also yields `exists() == True`, so it would pass the gate.

Grep confirming no `is_file` / `resolve` / `readlink` in the gate:

```
$ grep -n "_checked_tape_path\|relative_to\|is_file\|resolve()" comfy/Voyage/voyage/supervisor.py | head
396:    def _checked_tape_path(self, tape: str, segment_id: str) -> str:
407:            candidate.relative_to(self._run_dir)
```

## Repro steps

1. Init a run; plant `run/segments/evil.pt -> /tmp/outside.pt` (or any outside file).
2. Report `recovery_tape: "segments/evil.pt"` from the video worker (or inject via metrics).
3. Call `_checked_tape_path("segments/evil.pt", "000000")` → returns the path instead of raising `MediaError`.
4. Dir variant: `mkdir run/segments/adir`; report it as the tape → passes, downstream `resume` crashes with `IsADirectoryError` (not `MediaError`).
5. TOCTOU variant: validate a real file, replace it with a symlink before the worker `resume` call consumes it → use bypasses the check.

## Fix candidates

1. (Preferred) Resolve then contain:
   ```python
   resolved = candidate.resolve()  # follows symlinks
   try:
       resolved.relative_to(self._run_dir.resolve())
   except ValueError:
       raise MediaError(...)
   if not resolved.is_file():
       raise MediaError(...)
   ```
   Use `is_file()` (not `exists()`) so dirs/sockets/fifos fail with `MediaError`. Return the resolved absolute.
2. Close TOCTOU: open with `O_NOFOLLOW` where the consumer reads, or re-validate immediately before use (in `_resume_video_worker`, after restart, before `resume` RPC). Full elimination needs an fd-passing design; re-validation shrinks the window to unexploitable in practice for this threat model.
3. Regression tests: symlink-escape fixture (inside link → outside target must raise), dir-as-tape (must raise `MediaError`, not `IsADirectoryError`), `..` lexical escape (already covered), TOCTOU re-validation test.

## References

- `voyage/supervisor.py:396-414` (gate quoted above); callers: resume path `voyage/supervisor.py:539-555`, metrics tape check `voyage/cli.py:808-812`.
- Sibling `voyage/paths.py:62-86` (consumer side, issue 015) — fix both together so producer and consumer share containment.
- Python `Path.resolve` ("make the path absolute, resolving symlinks") vs `relative_to` (purely lexical): https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
- CWE-59 (link following) / CWE-367 (TOCTOU): https://cwe.mitre.org/data/definitions/59.html / https://cwe.mitre.org/data/definitions/367.html

## Progress log

- 2026-09-30: premise re-verified against live `voyage/supervisor.py` as read (`_checked_tape_path` at `:403-421`: lexical `relative_to` + `exists()`, no `resolve`/`is_file` anywhere in the gate). All three gaps held — symlink escape, dir-passes, TOCTOU. Re-ran the issue's stdlib reasoning against the live gate shape; premise confirmed, not invalidated by batch-1 (b40b052 never touched this region).
- 2026-09-30: failing tests first in `tests/test_supervisor_hardening.py` (new file): `test_tape_symlink_escape_rejected`, `test_tape_directory_rejected`, `test_latest_recovery_tape_skips_planted_entries`, and `test_latest_recovery_tape_skips_directory_tape` all failed pre-fix (gate returned the planted path; discovery returned the dir). `test_legitimate_tape_accepted` is the characterization guard (passes pre/post). Watched fail in-container via `scripts/test.sh`.
- 2026-09-30: fix implemented (see Resolution). Verification — all pass post-fix; existing tape tests green (`test_foreign_recovery_path_rejected`, `test_resume_failure_gets_second_chance`, `test_resume_failures_consume_the_same_budget` — the resume path still shares the budget through the hardened discovery). Related suites green (see 013 log for the one foreign failure). `ruff check` + `ruff format --check` + `mypy` (strict) green on touched files.

## Resolution

Resolve-then-contain plus regular-file enforcement, supervisor-side only (`paths.py` untouched — the concurrent 015 track owns the consumer side):

- `voyage/supervisor.py:449-481` (`_checked_tape_path`): `candidate.resolve()` first (symlink loops / unresolvable paths map to `MediaError`), containment checked on the resolved path against the resolved run dir (lexical bypass closed), then `is_file()` — not `exists()` — so directories/sockets/fifos fail here with `MediaError` instead of `IsADirectoryError` downstream. Returns the resolved absolute wire path, per the issue's preferred fix.
- `voyage/supervisor.py:652-693` (`_latest_recovery_tape`, the discovery site): mirrors the gate — each candidate tape is resolve-contained and `is_file()`-checked; non-conforming entries are skipped (resume degrades to a fresh stream, same as a missing tape) with a metric-visible `recovery_tape_skipped` event (`:677,686`, segment + reason), so a poisoned segment never hides silently and an older good tape still serves the resume.

Half left open (documented residual): check-then-use TOCTOU between validation and the worker's `resume` read (a symlink planted after the check still bypasses it). Shrunk in practice by re-validation at discovery on every resume; full elimination needs an fd-passing design (`O_NOFOLLOW` open at the consumer), which belongs to the paths.py/consumer track, not this gate.
