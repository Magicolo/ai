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
