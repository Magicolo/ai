# 222 — `atomic_write_*` silently narrows file mode `0644 → 0600` (MEDIUM)

## Technical description
`atomic.py:40-42,79-81` stage via `tempfile.mkstemp` (always `0600`) then
`os.replace`. The destination inherits the temp's mode. Every
manifest/state/metrics rewrite flips the file from the scaffolded `0644`
to owner-only.

## Rationale
Multi-user/box-shared runs (bind-mounted `output/`, backup sidecars, status
readers as another uid) lose read access after the first supervisor write.
`go-atomic-write`/`atomicfile` preserve mode explicitly for this reason.
Also breaks the "readers never need the writer's uid" assumption in
`iter_metric_files`/`scoreboard`.

## Live evidence
```
command: docker run --rm -v /app:/app -w /app voyage:latest python3 -c "
... chmod 644, atomic_write_json, stat ..."
output:
before 0o644
after atomic_write_json 0o600
new file mode 0o600
```

## Repro
`chmod 644` a file, `atomic_write_json` it, `stat`.

## Source refs
- `Voyage/voyage/atomic.py:38-59,66-94`

## Online sources
- https://pkg.go.dev/github.com/larsartmann/go-atomic-write%40v0.3.0 —
  "File permissions are preserved from the existing file (defaults to 0644
  for new files)".
- https://github.com/nradawg/durable-replace — `fchmod(temp)` to exact
  final permissions before `fsync`+`rename`.

## Fix candidates
1. `os.fchmod(tmp_fd, dest.stat().st_mode & 0o777 if dest.exists() else
   0o644)` before `fsync`.
2. Or `shutil.copymode` fallback.
3. Pin mode in a test.

## Log
- Track A sweep, 2026-10-07. Read-only; nothing fixed.

## Evaluation (2026-10-07)
Re-verified live against the current tree before fixing — all
load-bearing claims hold, nothing stale:
- `tempfile.mkstemp` (always 0600) + `os.replace` with no chmod
  confirmed at `voyage/atomic.py:40-42` (`atomic_write_bytes`) and
  `:79-81` (`atomic_copy`) — both writers narrowed.
- Red-state proven in-container (`voyage:latest`, old logic replayed:
  chmod 644 → mkstemp/write/replace → `0o600`), so the new 0644
  pins genuinely fail pre-fix.

## Resolution (2026-10-07)
- Mode preservation implemented in the shared path
  (`voyage/atomic.py`): new `DEFAULT_FILE_MODE = 0o644` +
  `_replace_mode()` (existing destination keeps
  `stat.S_IMODE`, missing/failed stat races to 0644 — preserve,
  never widen), applied via `os.fchmod` on the open temp fd before
  fsync in both `atomic_write_bytes` and `atomic_copy` (same
  narrowing bug, same fix). `atomic_write_json` inherits it;
  `_merge_manifest_record` (issue 202) composes for free.
  Module + `atomic_copy` docstrings updated; `except OSError`
  shaped without `return`-in-`try` (TRY300).
- New `Voyage/tests/test_atomic_mode_222.py` (6 tests): json/bytes
  writers preserve 0644/0640, deliberate 0600 stays 0600, new files
  land 0644 deterministically (fchmod bypasses umask), `atomic_copy`
  preserve + default.
- Verify (in-container): scoped pytest 49 passed (12 new + 37
  `test_unit`), `ruff check` + `ruff format --check` clean on all
  touched files, `mypy` strict clean on `voyage/atomic.py` +
  `voyage/model_registry.py`.
- Uncommitted; left for orchestrator review.
