# `pyproject.toml` is missing cheap enforcement

- Severity: medium (process — gates live only in prose and scripts).
- Status: FIXED. `warn_unused_ignores` + `addopts` landed earlier;
  this change adds the coverage gate (`pytest-cov==7.1.0` exact pin in
  `requirements-dev.txt`, `[tool.coverage.run]` with `source=["zoomy"]`
  and `/tmp` `data_file`, `[tool.coverage.report]` `fail_under = 67`
  ratcheting upward from the measured 68%) wired into `quality-gates.sh`
  only (targeted runs stay floor-free), plus the `scripts/` seam decision
  (stays unpackaged; redundant `sys.path` test inserts removed in favor of
  the `pythonpath` seam). Measured baseline: 174 passed, TOTAL 68%
  (interface 73%, local_engine 42% — the rest is GPU-live code by design).

## Evidence

1. **No `addopts`**: cache suppression relies on every caller remembering
   `-p no:cacheprovider` (AGENTS.md §10, both scripts). Bare `pytest`
   writes root-owned `.pytest_cache/` through the bind mount. Same for
   `--strict-markers` (no markers policy exists at all).
2. **No coverage gate**: no `[tool.coverage.*]`, no `--cov`, no
   `fail_under`. Handler coverage gaps (bind handlers, `_finalize_window`
   branching, `_frame_pipeline_for` eviction) fail nothing.
3. **No `warn_unused_ignores`**: mypy `strict=true` is neutered by 25+
   `ignore_missing_imports` entries across two override blocks; stale
   ignores accumulate silently. (Keep the ignores — the slim gate image
   genuinely lacks the GPU stack — but flag dead ones.)
4. `mypy_path=["scripts"]` + `sys.path.insert` hacks in
   `tests/test_download_*.py` instead of packaging `scripts/` — fragile
   import seam.

## Fix

```toml
[tool.mypy]
warn_unused_ignores = true
[tool.pytest.ini_options]
addopts = ["-p", "no:cacheprovider", "--strict-markers"]
```

plus a `[tool.coverage.run]`/`[tool.coverage.report]` section with
`fail_under` ratcheting upward, and a tracked decision on the
`scripts/` packaging seam. Add `pytest-cov` (exact pin) to
`requirements-dev.txt` if the coverage gate needs it — the gate script
must keep passing on the slim image, so verify there.

## Verification

- Bare in-container `pytest` writes no cache dirs; stale ignore →
  mypy error; coverage below the floor fails the gate.
- Gates: `Zoomy/scripts/quality-gates.sh` green.
