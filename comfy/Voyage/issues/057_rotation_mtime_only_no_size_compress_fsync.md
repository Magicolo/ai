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
