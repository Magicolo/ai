# 057 — Rotation is mtime-day-only: no size cap, no compression, no fsync, mtime-spoofable, retention unwired

**Severity:** MEDIUM

**File:line:** `voyage/logrotate.py:69-102` (`rotate_log`), `138-153` (`_prune_siblings`); call sites `voyage/supervisor.py:470`, `voyage/rpc.py:139`

**Overlaps with:** 056 (worker-log rotation cadence — sibling; not a duplicate)
- **Description:** `rotate_log()` decides solely on `st_mtime` day (UTC) vs today. There is no size trigger, no `.gz` compression, no `flush/fsync/fsync_dir` anywhere in the module, `_prune_siblings` deletes without a directory fsync, `keep_days` is always the default `30` at both call sites (never from TOML/CLI), and `iter_metric_files` glob `metrics-*{suffix}` plus `_ROTATED_SUFFIX` matching is the only guard against colliding with non-rotated files.
- **Rationale:** Contradicts the repo's own durability bar (`voyage/atomic.py` — temp + fsync + `os.replace` + `fsync_dir`, DESIGN §31) and standard rotation guidance (rotate by size *or* time at a line boundary; compress to `.jsonl.gz`; sortably-named siblings). A single-day flood (verbose worker stderr, hot loop) can still fill the disk despite "rotation"; `touch`/backup/rsync/chown changing mtime causes spurious or missed rotations; a power loss between `rename` and prune can leave the directory entry unpersisted while `validate_run` assumes atomicity elsewhere.
- **Evidence (re-verified live 2026-09-30):** `'st_size' in logrotate.py` → False; `'gzip'/'.gz'` → absent; `'fsync'` → absent; `'os.replace'` → absent; `grep keep_days voyage/` shows only the defaulted signatures, zero CLI/TOML/config plumbing. `rotate_log` renames via `path.rename(rotated)` (`logrotate.py:92`) with no fsync of file or parent.
- **Repro:**
```bash
python3 -c "import pathlib; print('size-guard:', 'st_size' in pathlib.Path('voyage/logrotate.py').read_text())"
grep -rn "keep_days" voyage/ --include='*.py'
```
- **Fix candidates:** Add size trigger (e.g. `max_bytes`, default ~10–50 MiB) OR keep time-only but document the single-day flood gap + add a disk-free alert (see 066); compress rotated siblings (`.jsonl.gz`, update `iter_metric_files`); fsync file + `fsync_dir` after rename/prune per `atomic.fsync_dir`; expose `keep_days`/`max_bytes` via `[voyage]` config + `status` display.
- **Refs:** `voyage/atomic.py:fsync_dir`; jsonlkit guide ("Rotate by size or time, and always at a line boundary … Compress rotated files to `.jsonl.gz` … Name files sortably"); `docs/OPERATIONS.md:229`.

## Progress log

- 2026-09-30: premise RE-VERIFIED live in-container (as-read
  2026-09-30): `rotate_log` still mtime-only (`stat().st_mtime`,
  no size branch), `append_line` still bare `write` (no
  flush/fsync), `rotate_log`/`_prune_siblings` still no `fsync_dir`,
  `gzip`/`.gz`/`os.replace` still absent from the module. The
  file-level `st_size` hit is `rotate_open_log` (issue 056,
  copytruncate mid-run rotation) — out of contract, left untouched
  (no regression). `keep_days` still defaulted at both call sites
  (`supervisor._log_metric` via `append_line`, `rpc.start` via
  `rotate_log`) with zero TOML/CLI plumbing — out-of-contract files,
  recorded as residual. TDD: new rotation tests watched fail, then
  fixed.

## Resolution (FIXED within owned files; compress + plumbing deferred)

- `voyage/logrotate.py::rotate_log(path, keep_days, max_bytes=None)`:
  size trigger added (`max_bytes=None` reads current
  `MAX_METRICS_BYTES` at call time, monkeypatch seam mirroring
  `rotate_open_log`); rotates on stale day OR size overflow; after
  rename, best-effort `fsync_dir(parent)` (live file stays
  appendable on failure).
- New `MAX_METRICS_BYTES = 10 MiB` (same default as worker logs;
  single-day floods now roll by size or time).
- `voyage/logrotate.py::append_line(..., max_bytes=None)`: durable
  tail — `flush` + `os.fsync` + `fsync_dir(parent)` (101 leg);
  docstring states the advisory-vs-takes contract (rotation stays
  best-effort; a failed append/sync raises so a failing disk fails
  loudly instead of losing forensics silently).
- `voyage/logrotate.py::_prune_siblings`: best-effort `fsync_dir`
  after any unlink batch (never raises).
- Untouched per contract: `rotate_open_log`,
  `rotate_worker_logs`, `MAX_WORKER_LOG_BYTES`, `_unique_rotated`,
  `WORKER_LOG_FILENAMES` (056).
- Tests: `test_057_append_line_flushes_and_syncs`,
  `test_057_size_trigger_rotates_same_day`,
  `test_057_rotate_syncs_directory_entry`,
  `test_057_prune_syncs_directory_entry`.
- Evidence: new file 13/13 pass; related suites 99 passed;
  `ruff check .` + `format --check` + `mypy voyage` clean.
- DESIGN proposals (text only, out-of-contract files — not
  implemented): (a) compress rotated siblings to `.jsonl.gz` +
  teach `iter_metric_files` + the `cli._read_all_metric_events` /
  `scoreboard._stages_by_segment` readers to consume gzip (cannot
  land in `logrotate.py` alone — readers do `read_text` today);
  (b) plumb `keep_days`/`max_bytes` via `[voyage]` config + `status`
  display (both call sites currently default).
