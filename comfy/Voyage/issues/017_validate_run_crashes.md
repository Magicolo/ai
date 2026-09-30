# 017 — Read-only `validate_run` tracebacks on dir-as-file, nested-metrics blowup, and `sha256_file` on dirs

- Severity: MEDIUM
- Group: correctness/tooling — Rank: 3/5
- File:line: `voyage/cli.py:781-815` (`_check_segment_checksums`), `voyage/cli.py:805-875` (`_check_segment_metrics`), `voyage/hashing.py:19-25` (`sha256_file`)

> Note on line numbers: the task mapping cites `voyage/cli.py:733-754,757-776`. A concurrent agent has uncommitted edits in `voyage/cli.py` (verified via `git diff HEAD -- comfy/Voyage/voyage/cli.py`: `--name` alias + `_effective_run_id`, shifting the file by ~+9 lines). All refs below cite **live** numbers observed 2026-09-30 via `grep -n`.

## Technical description

`validate_run` is documented "Read-only consistency check. Never mutates the run" (`voyage/cli.py:837-838`) and is the operator's first tool on a sick run. Three inputs crash it with a traceback instead of an `INVALID` line:

1. **Dir-as-file.** `_check_segment_checksums` (`:742-763`) calls `sha256_file(target)` when `target.exists()`. `sha256_file` (`voyage/hashing.py:19-25`):
   ```python
   with path.open("rb") as handle: ...
   ```
   On a directory this raises `IsADirectoryError` (an `OSError`, not caught — the function catches nothing, and `_check_segment_checksums` catches nothing). Same for `metrics.json` read in `_check_segment_metrics` (`:782`: `metrics_path.read_text(...)` inside `try: ... except (ValueError, KeyError, AttributeError, TypeError)` — `IsADirectoryError` is an `OSError`, **not** in the tuple, so it escapes). A run where `video.mp4` / `metrics.json` is a directory (failed `mkdir`/mount confusion, operator error, malicious worker) turns `voyage validate` into a traceback.
2. **Nested-metrics blowup.** `_check_segment_metrics` does `metrics.get("frames", 0)` then `int(...)`, and `float(block.get("duration", 0.0))` with `except (TypeError, ValueError)`. A `metrics.json` whose `frames`/`duration` nests adversarial structures (deep lists/dicts) can raise `RecursionError` during `json.loads` or `float()` coercion paths that are not in the caught tuple (`RecursionError` subclasses `RuntimeError`, not `ValueError`). The read-only tool then crashes on the exact runs that most need diagnosis.
3. **`sha256_file` on special files.** Same root cause as (1): no `is_file` pre-check, no `OSError` mapping to a validation error string. `validate` must translate *every* filesystem anomaly into `errors.append(...)`, never raise.

## Why it matters

- `validate` is the recovery entry point (`voyage run` → `validate` → `finalize --skip-bad`). If it tracebacks, the operator has no machine-readable verdict and scripts (`gates.sh`, soak) that parse `VALID:`/`INVALID:` break.
- Dir-as-file is not hypothetical: segment dirs are created with `mkdir exist_ok` (013), workers report paths (006), and mounts (`~/.cache/voyage-models`, `/output`) have all been confused before (see 053/077 history).
- The fix is cheap (widen `except` + pre-check `is_file`), and the contract ("never raises") is directly testable.

## Live evidence

Host stdlib probes (no container deps — `hashing` is stdlib-only):

```
$ PYTHONPATH=Voyage python3 - <<'PY'
from pathlib import Path
from voyage.hashing import sha256_file
import tempfile
with tempfile.TemporaryDirectory() as td:
    d = Path(td)/"seg"; d.mkdir(); (d/"video.mp4").mkdir()
    try: sha256_file(d/"video.mp4")
    except Exception as e: print(f"sha256_file(dir) -> {type(e).__name__}: {e}")
    try: (d/"metrics.json").mkdir(); (d/"metrics.json").read_text()
    except Exception as e: print(f"read_text(dir) -> {type(e).__name__}: {e}")
PY
sha256_file(dir) -> IsADirectoryError: [Errno 21] Is a directory: '/tmp/.../seg/video.mp4'
read_text(dir) -> IsADirectoryError: [Errno 21] Is a directory: '/tmp/.../seg/metrics.json'
```

- `IsADirectoryError` subclasses `OSError`, **not** `(ValueError, KeyError, AttributeError, TypeError)` caught at `cli.py:784`. Confirmed by hierarchy: `IsADirectoryError -> OSError -> Exception`.
- `sha256_file` catches nothing (`hashing.py:19-25` — five lines, zero `try`).

Grep (live):

```
$ grep -n "def _check_segment_checksums\|def _check_segment_metrics\|def sha256_file" comfy/Voyage/voyage/cli.py comfy/Voyage/voyage/hashing.py
cli.py:742:def _check_segment_checksums(
cli.py:766:def _check_segment_metrics(
hashing.py:19:def sha256_file(
```

## Repro steps

1. `init` a fake run; `mkdir segments/000000/video.mp4` (dir where the file belongs) + `touch segments/000000/DONE`.
2. `voyage validate --run <dir>` → traceback `IsADirectoryError` instead of `INVALID: ... missing/checksum ...`.
3. Variant: `mkdir segments/000000/metrics.json` → same class from `_check_segment_metrics`.
4. Variant: write `metrics.json` with `{"frames": [[[[...deeply nested...]]]]}` → observe unhandled `RecursionError` (or at minimum no `unreadable metrics.json` verdict).

## Fix candidates

1. (Preferred) Make `validate` total: in `_check_segment_checksums`, skip non-files with an error string (`if not target.is_file(): errors.append(...); continue`); in `_check_segment_metrics`, widen `except` to `(ValueError, KeyError, AttributeError, TypeError, OSError, RecursionError)` and map to `unreadable metrics.json`. Both keep the `VALID:`/`INVALID:` contract total.
2. Harden `sha256_file` callers (not the helper itself — it is correctly strict for the commit path where a dir *should* raise): gate on `is_file()` at the validate layer only.
3. Regression tests (tmp_path, no GPU): dir-as-`video.mp4`, dir-as-`metrics.json`, deeply-nested `metrics.json` → each yields `INVALID` entries, exit code 1, zero tracebacks.

## References

- `voyage/cli.py:742-763` (checksums, no `OSError` handling), `:766-813` (metrics, narrow `except` at `:784`), `:837-889` (`validate_run` — no outer guard).
- `voyage/hashing.py:19-25` (strict helper, correct for commit, wrong to call unchecked from validate).
- Python exception hierarchy — `IsADirectoryError → OSError`, `RecursionError → RuntimeError`: https://docs.python.org/3/library/exceptions.html#exception-hierarchy
- `json.loads` recursion limits: https://docs.python.org/3/library/json.html
