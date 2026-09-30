# 077 — Parallel `ensure_models` manifest repair is best-effort and can silently bless hash-less state

- Severity: MEDIUM
- File: `Voyage/voyage/models_ensure.py:16-21` (race docstring), `:142-167` (`_repair_manifest` swallows `OSError`), `:227-232` (repair-then-`console.ok("models ready…")`)
- Area: model registry / orchestration — `ensure_models` manifest race

## Description

Parallel `download_model` calls race on `manifest.json` read-modify-write (last writer wins, entries lost). The repair pass re-stats files to re-merge missing entries, but `except OSError: continue` (`:162-163`) means disk-full/permission/racing deletes are silently skipped — and success (`"models ready"`) is still reported because `verify_model` (presence + size floors, cf. issue 071) already passed. Net effect: a run can proceed with a manifest missing `checkpoint_sha256` entries, which downgrades every later `verify_checkpoint_against_manifest` to pass-through (issue 071's hole) with no warning.

## Rationale

Security bookkeeping must be fail-loud: a silent manifest gap converts all downstream hash gates into no-ops, and the operator sees "models ready".

## Live evidence

Re-verified 2026-09-30 live:

```
Voyage/voyage/models_ensure.py:16-21:
Manifest race note: `model_registry.download_model` bundles the hub fetch
with a read-modify-write of `manifest.json`, so parallel calls can drop
each other's manifest entries (last write wins). ... The
repair is best-effort — `verify_model` stays authoritative.
Voyage/voyage/models_ensure.py:38: _MANIFEST_LOCK = threading.Lock()
Voyage/voyage/models_ensure.py:142-167 (_repair_manifest):
    with _MANIFEST_LOCK:
        for entry in entries:
            try:
                ...
                model_registry._merge_manifest_record(...)
            except OSError:
                continue
Voyage/voyage/models_ensure.py:227-232:
    if failures:
        _repair_manifest([entry for entry, _ in missing if entry.spec not in failures])
        ...
        return 1
    _repair_manifest([entry for entry, _ in missing])
    console.ok(f"models ready: {', '.join(entry.spec for entry, _ in missing)}")
```

Docstring admits "best-effort — `verify_model` stays authoritative" while `verify_model` checks no hashes (`model_registry.py:1192-1198`).

Overlaps with 071 (the hash-less state blessed here downgrades 071's gate to pass-through) — ownership stays here (race/repair); 071 owns ingest verification.

## Repro

Force the race (4 parallel specs on a slow FS) or inject `OSError` (read-only `manifest.json` during repair in a unit test with `models_root` on a `tmp_path` + `chmod 444`) → `ensure_models` returns 0 with a manifest missing keys; subsequent `verify_checkpoint_against_manifest` returns without checking.

## Fix candidates

1. Serialize the manifest merge under `_MANIFEST_LOCK` around the whole fetch+merge (not just repair).
2. Atomic merge with retry + fsync (repo already has `voyage/atomic.py` + `fsync_dir` discipline per §12).
3. Verify repair outcome and fail loud when entries remain missing.
