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

## Resolution log 2026-09-30 (Rank-2 batch)

Re-verified live 2026-09-30 — CONFIRMED with two mechanism corrections,
code having evolved since filing: (a) `verify_model` is no longer
presence-only — it now checks recorded hashes via `_manifest_hash_mismatches`
(`model_registry.py:1299-1338`); but a race-dropped entry leaves NO baseline,
so it still passes hash-less exactly as described. (b)
`verify_checkpoint_against_manifest` now fails CLOSED by default (071:
raises `ValueError` on missing sha unless `allow_missing_manifest`/env
opt-in; both workers call it without opt-in) — so today's downstream
consequence is a loud late worker refusal, not silent pass-through
(pass-through needs explicit opt-in). The core defect stands as filed:
`except OSError: continue`, non-atomic `_merge_manifest_record`, no
post-repair verification, unlocked fetch+merge, `"models ready"` regardless.

Fix (`Voyage/voyage/models_ensure.py` only): `_repair_manifest` now returns
`list[RequiredModel]` still-missing — one atomic write per models dir
(`atomic_write_json` + `fsync_dir`, §12 rule) under `_MANIFEST_LOCK` with
`_REPAIR_ATTEMPTS = 3` on transient I/O; torn manifests are never
overwritten (entries stay missing; torn files are `validate_run`'s
territory). `ensure_models` fails loud (`return 1` + per-spec
`console.error`, never `"models ready"`) on unrepaired entries, both paths.
Documented carve-out: an ABSENT manifest is skipped, not repaired — real
`download_model` merges on every success (merge errors fail the download)
and last-writer-wins always leaves ≥1 record, so absent-after-success is
impossible outside custom downloaders; worker fail-closed still guards
loads. Module docstring updated; registry-side follow-up noted there.

TDD: 5 repair/ensure tests in `Voyage/tests/test_containers_rank2.py`
(11→10 failing pre-fix; 4 true drivers + 1 labeled characterization
guard for the happy path). Two pre-existing ensure tests initially broke
(their fakes download without merging under real spec names) — reconciled
via the absent-manifest carve-out, NOT by touching those tests (out of
scope); verified the failures were the repair path, not the fakes.

Evidence: batch 48 passed (`test_containers_rank2` + `test_generate_ensure`
+ `test_registry_pins`); full in-container suite 1388 passed, 5 skipped,
1 deselected; `ruff check` + `ruff format --check` + `mypy --strict` clean
on both touched py files. Never `pip install` on host; never formatted
`issues/*.md`.

Residuals: fetch+merge inside `download_model` still unlocked — a
concurrent downloader can clobber the repair write (narrow window, loud on
next verify). `_merge_manifest_record` (non-atomic) still serves direct
`voyage models download` paths; repair no longer uses it.

DESIGN proposal (text only, not implemented): module-level manifest lock in
`model_registry.py` held around the fetch+merge in `download_model` (repair
reuses it); bigger alternative is moving the merge out of `download_model`
into a single serialized commit step in the ensure caller.
